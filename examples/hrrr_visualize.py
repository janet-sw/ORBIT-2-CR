"""Visualization script for HRRR composite reflectivity prediction + super-resolution.

Loads a trained ORBIT-2 checkpoint and generates side-by-side visualizations
of input, prediction, and ground truth for test samples.

Usage (single GPU is fine for visualization):
    srun -n1 python hrrr_visualize.py ../configs/hrrr_forecasting.yaml \
        --checkpoint /path/to/checkpoint_epoch_0029.pt \
        --index 0 --num_samples 5

Or in a SLURM batch script with multiple GPUs (FSDP will be used):
    srun python hrrr_visualize.py ../configs/hrrr_forecasting.yaml \
        --checkpoint /path/to/checkpoint_epoch_0029.pt \
        --index 0 --num_samples 5
"""

import os
import sys
import yaml
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType, FullStateDictConfig
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
from climate_learn.models.hub.components.vit_blocks import Block
from climate_learn.utils.fused_attn import FusedAttn
from utils import seed_everything

# Try to import matplotlib (may not be available on compute nodes)
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not available, will save numpy arrays only", flush=True)


def create_model(config, device, world_rank, in_vars, out_vars):
    """Create the Res_Slim_ViT model (same logic as era5_forecasting.py)."""
    from climate_learn.models.hub import Res_Slim_ViT

    preset = config["model"]["preset"]
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
    """Create data module for downscaling (same logic as era5_forecasting.py)."""
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
    batch_size = 1  # Visualize one sample at a time
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


def load_checkpoint_weights(model, checkpoint_path, world_rank):
    """Load model weights from checkpoint into FSDP-wrapped model."""
    if world_rank == 0:
        print(f"Loading checkpoint: {checkpoint_path}", flush=True)

    # For FSDP models, we need to load using the FSDP state dict API
    # Load on CPU first
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint.get("epoch", "unknown")

    if world_rank == 0:
        print(f"Loaded checkpoint from epoch {epoch}", flush=True)

    del checkpoint
    return epoch


def denormalize_output(data_module, tensor, out_vars):
    """Denormalize output tensor back to physical units."""
    denorm = cl.load_transform("denormalize", data_module)
    return denorm(tensor)


def save_visualization(input_lr, prediction, ground_truth, out_vars, save_dir, index, epoch):
    """Save visualization of a single sample."""
    os.makedirs(save_dir, exist_ok=True)

    var_name = out_vars[0]  # For HRRR, this is composite_reflectivity

    # Save numpy arrays (always works, even without matplotlib)
    np.save(os.path.join(save_dir, f"sample_{index}_prediction.npy"), prediction)
    np.save(os.path.join(save_dir, f"sample_{index}_ground_truth.npy"), ground_truth)
    np.save(os.path.join(save_dir, f"sample_{index}_input_lr.npy"), input_lr)

    if not HAS_MATPLOTLIB:
        print(f"  Saved numpy arrays for sample {index}", flush=True)
        return

    # Determine color range from ground truth
    vmin = ground_truth.min()
    vmax = ground_truth.max()

    fig, axes = plt.subplots(1, 3, figsize=(24, 6))

    # Input (low-res) — show the first channel (e.g., 2m_temperature) as a reference
    # Since composite_reflectivity is NOT in input, show mean of all input channels
    im0 = axes[0].imshow(input_lr, cmap="viridis", aspect="auto")
    axes[0].set_title(f"Input (LR mean, 0.1°)", fontsize=14)
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    # Prediction (super-resolved)
    im1 = axes[1].imshow(prediction, cmap="turbo", vmin=vmin, vmax=vmax, aspect="auto")
    axes[1].set_title(f"Prediction ({var_name}, 0.05°)", fontsize=14)
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Ground truth (high-res)
    im2 = axes[2].imshow(ground_truth, cmap="turbo", vmin=vmin, vmax=vmax, aspect="auto")
    axes[2].set_title(f"Ground Truth ({var_name}, 0.05°)", fontsize=14)
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    # Compute quick metrics
    rmse_val = np.sqrt(np.mean((prediction - ground_truth) ** 2))
    corr = np.corrcoef(prediction.flatten(), ground_truth.flatten())[0, 1]

    fig.suptitle(
        f"HRRR {var_name} — Sample {index} (Epoch {epoch}) | RMSE={rmse_val:.4f}, Corr={corr:.4f}",
        fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    fig_path = os.path.join(save_dir, f"sample_{index}_visualization.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Also save difference map
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    diff = prediction - ground_truth
    abs_max = max(abs(diff.min()), abs(diff.max()))
    im_diff = ax2.imshow(diff, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max, aspect="auto")
    ax2.set_title(f"Error (Pred - Truth) | Mean Bias={diff.mean():.4f}", fontsize=14)
    plt.colorbar(im_diff, ax=ax2, fraction=0.046)
    plt.tight_layout()
    fig2_path = os.path.join(save_dir, f"sample_{index}_error_map.png")
    plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved visualization to {fig_path}", flush=True)


def main():
    parser = ArgumentParser(description="Visualize HRRR ORBIT-2 model outputs")
    parser.add_argument("config", type=str, help="Path to HRRR config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pt file")
    parser.add_argument("--index", type=int, default=0, help="Starting test sample index (default: 0)")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of test samples to visualize (default: 5)")
    parser.add_argument("--save_dir", type=str, default="visualizations", help="Output directory (default: visualizations)")
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
        print(f"Initialized {world_size} GPUs for visualization", flush=True)

    # Load config
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    seed_everything(42)

    # Create data module and model
    data_module, in_vars, out_vars = create_data_module(config, world_rank, world_size)
    model = create_model(config, device, world_rank, in_vars, out_vars)

    # FSDP wrap (must match training setup)
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Block},
    )

    # Use float32 for visualization (bfloat16 → numpy conversion issues)
    model = FSDP(
        model,
        device_id=local_rank,
        mixed_precision=None,  # float32 for visualization
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=True,
    )

    # Load checkpoint
    epoch = load_checkpoint_weights(model, args.checkpoint, world_rank)

    # Get denormalization transform
    denorm = cl.load_transform("denormalize", data_module)

    # Set to eval mode
    model.eval()

    # Get test dataloader
    test_dl = data_module.test_dataloader()

    if world_rank == 0:
        print(f"\nVisualizing {args.num_samples} test samples starting from index {args.index}...", flush=True)

    sample_count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dl):
            if batch_idx < args.index:
                continue
            if sample_count >= args.num_samples:
                break

            x, y, in_variables, out_variables = batch
            x = x.to(device)
            y = y.to(device)

            if x.dim() == 5:
                x = x.flatten(1, 2)
            if y.dim() == 5:
                y = y.flatten(1, 2)

            # Forward pass
            yhat = model.forward(x, in_variables, out_variables)

            # Denormalize to physical units
            yhat_denorm = denorm(yhat)
            y_denorm = denorm(y)

            # Only rank 0 saves visualizations
            if world_rank == 0:
                # Get numpy arrays
                pred_np = yhat_denorm[0, 0].cpu().float().numpy()  # [H, W] for first output var
                truth_np = y_denorm[0, 0].cpu().float().numpy()
                input_np = x[0].mean(dim=0).cpu().float().numpy()  # Mean across input channels

                print(f"\nSample {batch_idx}:", flush=True)
                print(f"  Prediction: shape={pred_np.shape}, range=[{pred_np.min():.2f}, {pred_np.max():.2f}]", flush=True)
                print(f"  Ground truth: shape={truth_np.shape}, range=[{truth_np.min():.2f}, {truth_np.max():.2f}]", flush=True)

                save_visualization(input_np, pred_np, truth_np, out_vars, args.save_dir, batch_idx, epoch)

            sample_count += 1

    if world_rank == 0:
        print(f"\nVisualization complete! Results saved to {args.save_dir}/", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()