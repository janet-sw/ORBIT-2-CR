"""Test evaluation script for HRRR composite reflectivity prediction + super-resolution.

Runs the trained ORBIT-2 model on the full test set and reports:
  - ACC  (Anomaly Correlation Coefficient, latitude-weighted)
  - RMSE (Root Mean Squared Error, latitude-weighted)
  - lat_mse (Latitude-weighted MSE, in normalized space)
  - SSIM (Structural Similarity Index, computed in physical/denormalized space)

Usage:
    srun -n <NUM_GPUS> python hrrr_test.py ../configs/hrrr_forecasting.yaml \
        --checkpoint /path/to/checkpoint_epoch_0029.pt

Example (4 nodes, 32 GPUs on Frontier):
    srun -N4 -n32 python hrrr_test.py ../configs/hrrr_forecasting.yaml \
        --checkpoint /lustre/orion/csc662/proj-shared/janet/checkpoints/hrrr_refc/checkpoint_epoch_0029.pt
"""

import os
import sys
import yaml
import time
import json
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from argparse import ArgumentParser
from datetime import timedelta
import functools

import climate_learn as cl
from climate_learn.data.processing.era5_constants import (
    PRESSURE_LEVEL_VARS,
    DEFAULT_PRESSURE_LEVELS,
    CONSTANTS,
)
from climate_learn.metrics import MetricsMetaInfo
from climate_learn.utils.loaders import get_climatology
from climate_learn.models.hub.components.vit_blocks import Block
from climate_learn.utils.fused_attn import FusedAttn
from utils import seed_everything

# SSIM: try torch-based first, fall back to skimage
try:
    from skimage.metrics import structural_similarity as sk_ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("WARNING: scikit-image not available, SSIM will be skipped", flush=True)


def clip_replace_constant(y, yhat, out_variables):
    """Post-process predictions (same as era5_forecasting.py)."""
    if "total_precipitation_24hr" in out_variables:
        prcp_index = out_variables.index("total_precipitation_24hr")
        torch.clamp_(yhat[:, prcp_index, :, :], min=0.0)
    for i in range(yhat.shape[1]):
        if out_variables[i] in CONSTANTS:
            yhat[:, i] = y[:, i]
    return yhat


def create_model(config, device, world_rank, in_vars, out_vars):
    """Create the Res_Slim_ViT model."""
    from climate_learn.models.hub import Res_Slim_ViT

    lr_h, lr_w = config["model"]["img_size"]
    superres_factor = config["model"]["superres_factor"]

    constant_vars = ["land_sea_mask", "orography", "lattitude", "landcover"]
    num_constant_vars = sum(1 for var in constant_vars if var in in_vars)

    all_vars = []
    seen = set()
    for var in in_vars + out_vars:
        if var not in seen:
            all_vars.append(var)
            seen.add(var)

    model = Res_Slim_ViT(
        default_vars=all_vars,
        img_size=(lr_h, lr_w),
        in_channels=len(in_vars),
        out_channels=len(out_vars),
        superres_mag=superres_factor,
        history=config["model"].get("history", 1),
        patch_size=config["model"]["patch_size"],
        cnn_ratio=config["model"]["cnn_ratio"],
        learn_pos_emb=config["model"]["learn_pos_emb"],
        embed_dim=config["model"]["embed_dim"],
        depth=config["model"]["depth"],
        decoder_depth=config["model"]["decoder_depth"],
        num_heads=config["model"]["num_heads"],
        mlp_ratio=config["model"].get("mlp_ratio", 4.0),
        drop_path=config["model"].get("drop_path", 0.1),
        drop_rate=config["model"].get("drop_rate", 0.0),
        num_constant_vars=num_constant_vars,
        FusedAttn_option=FusedAttn.DEFAULT,
        input_refine_cnn=config["model"].get("input_refine_cnn", False),
        output_refine_cnn=config["model"].get("output_refine_cnn", False),
    )

    if world_rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params / 1e6:.2f}M", flush=True)

    return model


def create_data_module(config, world_rank, world_size):
    """Create data module for downscaling."""
    variables = config["data"]["variables"]
    in_vars = []
    for var in variables:
        if var in PRESSURE_LEVEL_VARS:
            for level in DEFAULT_PRESSURE_LEVELS:
                in_vars.append(var + "_" + str(level))
        else:
            in_vars.append(var)

    out_variables = config["data"]["out_variables"]
    out_vars = []
    for var in out_variables:
        if var in PRESSURE_LEVEL_VARS:
            for level in DEFAULT_PRESSURE_LEVELS:
                out_vars.append(var + "_" + str(level))
        else:
            out_vars.append(var)

    low_res_dir = config["data"]["low_res_dir"]
    high_res_dir = config["data"]["high_res_dir"]
    batch_size = config["trainer"]["batch_size"]
    num_workers = config["trainer"]["num_workers"]
    buffer_size = config["trainer"]["buffer_size"]

    data_module = cl.data.IterDataModule(
        "downscaling",
        low_res_dir,
        high_res_dir,
        in_vars,
        out_vars,
        data_par_size=world_size,
        data_par_group=None,
        subsample=1,
        batch_size=batch_size,
        num_workers=num_workers,
        buffer_size=buffer_size,
    )
    data_module.resize_config = None
    data_module.setup()

    return data_module, in_vars, out_vars


def compute_ssim_batch(pred_denorm, target_denorm):
    """Compute mean SSIM over a batch of 2D images (per output channel).

    Args:
        pred_denorm: [B, C, H, W] denormalized predictions (float32 numpy or tensor)
        target_denorm: [B, C, H, W] denormalized ground truth

    Returns:
        float: mean SSIM across batch and channels
    """
    if not HAS_SKIMAGE:
        return float("nan")

    # Convert to numpy float32
    if torch.is_tensor(pred_denorm):
        pred_np = pred_denorm.cpu().float().numpy()
        target_np = target_denorm.cpu().float().numpy()
    else:
        pred_np = pred_denorm.astype(np.float32)
        target_np = target_denorm.astype(np.float32)

    B, C, H, W = pred_np.shape
    ssim_sum = 0.0
    count = 0

    for b in range(B):
        for c in range(C):
            p = pred_np[b, c]
            t = target_np[b, c]
            data_range = t.max() - t.min()
            if data_range < 1e-8:
                # Skip constant fields
                continue
            ssim_val = sk_ssim(t, p, data_range=data_range)
            ssim_sum += ssim_val
            count += 1

    return ssim_sum / max(count, 1)


def evaluate_test_set(
    model, test_dl, device, world_rank, world_size,
    val_loss_metrics, val_transforms, denorm_transform,
    in_vars, out_vars,
):
    """Run full evaluation on the test set.

    Returns:
        dict: {metric_name: value} averaged over all test batches
    """
    model.eval()

    # Accumulators
    loss_dict_sum = {}
    ssim_sum = 0.0
    ssim_count = 0
    num_batches = 0

    if world_rank == 0:
        print(f"\nEvaluating on test set...", flush=True)
        t0 = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dl):
            x, y, in_variables, out_variables = batch
            x = x.to(device)
            y = y.to(device)

            # Flatten 5D [B, C, T, H, W] to 4D [B, C*T, H, W]
            if x.dim() == 5:
                x = x.flatten(1, 2)
            if y.dim() == 5:
                y = y.flatten(1, 2)

            # Forward pass
            yhat = model.forward(x, in_variables, out_variables)
            yhat = clip_replace_constant(y, yhat, out_variables)

            # ---- Compute lat_rmse, lat_acc, lat_mse via existing metrics ----
            for i, lf in enumerate(val_loss_metrics):
                if val_transforms is not None and val_transforms[i] is not None:
                    yhat_ = val_transforms[i](yhat)
                    y_ = val_transforms[i](y)
                else:
                    yhat_ = yhat
                    y_ = y

                if y_.size(2) != yhat_.size(2) or y_.size(3) != yhat_.size(3):
                    losses = lf(yhat_, y_[:, :, :yhat_.size(2), :yhat_.size(3)])
                else:
                    losses = lf(yhat_, y_)

                loss_name = getattr(lf, "name", f"loss_{i}")
                if losses.dim() == 0:
                    key = f"test/{loss_name}:aggregate"
                    loss_dict_sum[key] = loss_dict_sum.get(key, 0.0) + losses.item()
                else:
                    if losses.numel() > 1:
                        for idx, var_name in enumerate(out_variables):
                            if idx < losses.numel():
                                key = f"test/{loss_name}:{var_name}"
                                loss_dict_sum[key] = loss_dict_sum.get(key, 0.0) + losses[idx].item()
                        key = f"test/{loss_name}:aggregate"
                        loss_dict_sum[key] = loss_dict_sum.get(key, 0.0) + losses[-1].item()
                    else:
                        key = f"test/{loss_name}:aggregate"
                        val = losses.item() if losses.numel() == 1 else losses
                        loss_dict_sum[key] = loss_dict_sum.get(key, 0.0) + val

            # ---- Compute SSIM in denormalized (physical) space ----
            yhat_denorm = denorm_transform(yhat)
            y_denorm = denorm_transform(y)
            batch_ssim = compute_ssim_batch(yhat_denorm, y_denorm)
            if not np.isnan(batch_ssim):
                ssim_sum += batch_ssim
                ssim_count += 1

            num_batches += 1

            if world_rank == 0 and batch_idx % 50 == 0:
                print(f"  Processed batch {batch_idx}...", flush=True)

    # Average over batches (local)
    if num_batches > 0:
        for key in loss_dict_sum:
            loss_dict_sum[key] /= num_batches

    local_ssim = ssim_sum / max(ssim_count, 1) if ssim_count > 0 else 0.0

    # All-reduce across GPUs for accurate global averages
    for key in list(loss_dict_sum.keys()):
        tensor_val = torch.tensor(loss_dict_sum[key], device=device)
        dist.all_reduce(tensor_val, op=dist.ReduceOp.AVG)
        loss_dict_sum[key] = tensor_val.item()

    # All-reduce SSIM
    ssim_tensor = torch.tensor(local_ssim, device=device)
    dist.all_reduce(ssim_tensor, op=dist.ReduceOp.AVG)
    loss_dict_sum["test/ssim:aggregate"] = ssim_tensor.item()

    # Also add per-variable SSIM (same as aggregate for single output var)
    if len(out_vars) == 1:
        loss_dict_sum[f"test/ssim:{out_vars[0]}"] = ssim_tensor.item()

    if world_rank == 0:
        elapsed = time.time() - t0
        print(f"  Test evaluation complete in {elapsed:.1f}s ({num_batches} batches per GPU)", flush=True)

    return loss_dict_sum


def main():
    parser = ArgumentParser(description="Evaluate HRRR ORBIT-2 model on test set")
    parser.add_argument("config", type=str, help="Path to HRRR config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pt file")
    parser.add_argument("--output", type=str, default="test_results.json", help="Output JSON file for results (default: test_results.json)")
    parser.add_argument("--master-port", type=str, default="29500", help="Master port for distributed")
    args = parser.parse_args()

    # Distributed setup
    os.environ["MASTER_ADDR"] = str(os.environ.get("HOSTNAME", "localhost"))
    os.environ["MASTER_PORT"] = args.master_port
    os.environ["WORLD_SIZE"] = os.environ.get("SLURM_NTASKS", "1")
    os.environ["RANK"] = os.environ.get("SLURM_PROCID", "0")

    world_size = int(os.environ["WORLD_SIZE"])
    world_rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("SLURM_LOCALID", "0"))

    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    dist.init_process_group("nccl", timeout=timedelta(seconds=7200), rank=world_rank, world_size=world_size)

    if world_rank == 0:
        print(f"=" * 80)
        print(f"HRRR ORBIT-2 Test Evaluation")
        print(f"=" * 80)
        print(f"World size: {world_size} GPUs")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"=" * 80, flush=True)

    # Load config
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    seed_everything(42)

    # Create data module and model
    data_module, in_vars, out_vars = create_data_module(config, world_rank, world_size)
    model = create_model(config, device, world_rank, in_vars, out_vars)

    if world_rank == 0:
        print(f"Input variables ({len(in_vars)}): {in_vars}", flush=True)
        print(f"Output variables ({len(out_vars)}): {out_vars}", flush=True)

    # FSDP wrap (must match training setup exactly)
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Block},
    )

    # Use float32 for evaluation to avoid precision issues
    model = FSDP(
        model,
        device_id=local_rank,
        mixed_precision=None,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=True,
    )

    # Apply activation checkpointing (must match training)
    reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.REENTRANT,
    )
    check_fn = lambda submodule: isinstance(submodule, Block)
    apply_activation_checkpointing(model, checkpoint_wrapper_fn=reentrant_wrapper, check_fn=check_fn)

    # Load checkpoint
    if world_rank == 0:
        print(f"\nLoading checkpoint: {args.checkpoint}", flush=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    loaded_epoch = checkpoint.get("epoch", "unknown")
    if world_rank == 0:
        print(f"Loaded checkpoint from epoch {loaded_epoch}", flush=True)
    del checkpoint

    # Create loss functions for evaluation (need test climatology)
    lat, lon = data_module.get_lat_lon()

    if world_rank == 0:
        print("Loading test climatology...", flush=True)
    test_clim = get_climatology(data_module, split="test")
    test_metainfo = MetricsMetaInfo(in_vars, out_vars, lat, lon, test_clim)

    # Create evaluation metrics: lat_rmse, lat_acc, lat_mse
    test_losses = [
        cl.load_loss(device, model, "lat_rmse", False, test_metainfo),
        cl.load_loss(device, model, "lat_acc", False, test_metainfo),
        cl.load_loss(device, model, "lat_mse", False, test_metainfo),
    ]

    # Transforms: denormalize for RMSE and ACC, None for lat_mse (normalized space)
    denorm = cl.load_transform("denormalize", data_module)
    test_transforms = [denorm, denorm, None]

    # Get test dataloader
    test_dl = data_module.test_dataloader()

    # Run evaluation
    results = evaluate_test_set(
        model, test_dl, device, world_rank, world_size,
        test_losses, test_transforms, denorm,
        in_vars, out_vars,
    )

    # Print and save results (rank 0 only)
    if world_rank == 0:
        print(f"\n{'=' * 80}")
        print(f"TEST RESULTS (Epoch {loaded_epoch})")
        print(f"{'=' * 80}")

        # Organize results nicely
        summary = {
            "checkpoint": args.checkpoint,
            "epoch": loaded_epoch,
            "metrics": {},
        }

        # Extract key metrics
        for key in sorted(results.keys()):
            value = results[key]
            print(f"  {key}: {value:.6f}")
            summary["metrics"][key] = value

        print(f"{'=' * 80}")

        # Highlight the main metrics
        print(f"\n--- Summary ---")
        acc_key = "test/lat_acc:aggregate"
        rmse_key = "test/lat_rmse:aggregate"
        mse_key = "test/lat_mse:aggregate"
        ssim_key = "test/ssim:aggregate"

        if acc_key in results:
            print(f"  ACC  (lat-weighted): {results[acc_key]:.6f}")
        if rmse_key in results:
            print(f"  RMSE (lat-weighted): {results[rmse_key]:.6f}")
        if mse_key in results:
            print(f"  lat_MSE            : {results[mse_key]:.6f}")
        if ssim_key in results:
            print(f"  SSIM               : {results[ssim_key]:.6f}")

        print(f"\n")

        # Save to JSON
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Results saved to {args.output}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()