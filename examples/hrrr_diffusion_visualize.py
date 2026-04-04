"""Visualization script for HRRR diffusion-refined composite reflectivity.

Loads both the frozen v6 backbone and the trained diffusion model, then
produces 4-panel comparisons:
    Input (LR) | v6 Prediction (blurry) | Diffusion Sample (sharp) | Ground Truth

Usage (1 node / 8 GPUs is enough):
    srun -n8 python hrrr_diffusion_visualize.py \
        ../configs/hrrr_diffusion_refine.yaml \
        --diff_checkpoint /path/to/diffusion_checkpoint.pt \
        --num_samples 5 --num_steps 30

Or single-GPU:
    srun -n1 python hrrr_diffusion_visualize.py \
        ../configs/hrrr_diffusion_refine.yaml \
        --diff_checkpoint /path/to/diffusion_checkpoint.pt \
        --num_samples 3 --num_steps 20

With warm-start (SDEdit) sampling (recommended — preserves v6 structure):
    srun -n1 python hrrr_diffusion_visualize.py \
        ../configs/hrrr_diffusion_refine.yaml \
        --diff_checkpoint /path/to/diffusion_checkpoint.pt \
        --warm_start --sigma_start 2.0 --num_steps 20
"""

import os
import sys
import yaml
import time
import functools
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from argparse import ArgumentParser
from datetime import timedelta

import climate_learn as cl
from climate_learn.data.processing.era5_constants import (
    PRESSURE_LEVEL_VARS,
    DEFAULT_PRESSURE_LEVELS,
)
from climate_learn.models.hub.res_slimvit import Res_Slim_ViT
from climate_learn.models.hub.edm_unet import ConditionalUNet, ResBlock
from climate_learn.models.hub.edm import EDMPrecond
from climate_learn.utils.fused_attn import FusedAttn

from utils import seed_everything

# Matplotlib — may not be available on all compute nodes
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not available, will save numpy arrays only", flush=True)


# ============================================================================
# Helpers (shared with training script)
# ============================================================================

def expand_variables(raw_vars):
    expanded = []
    for var in raw_vars:
        if var in PRESSURE_LEVEL_VARS:
            for level in DEFAULT_PRESSURE_LEVELS:
                expanded.append(f"{var}_{level}")
        else:
            expanded.append(var)
    return expanded


def load_v6_backbone(config, device, in_vars, out_vars, rank):
    """Load frozen v6 model (no FSDP, eval, no grad)."""
    mc = config["backbone"]
    all_vars = list(dict.fromkeys(in_vars + out_vars))

    constant_vars = ["land_sea_mask", "orography", "lattitude", "landcover"]
    num_constant_vars = sum(1 for v in constant_vars if v in in_vars)

    model = Res_Slim_ViT(
        default_vars=all_vars,
        img_size=tuple(mc["img_size"]),
        in_channels=len(in_vars),
        out_channels=len(out_vars),
        superres_mag=mc["superres_factor"],
        history=mc.get("history", 1),
        patch_size=mc["patch_size"],
        cnn_ratio=mc.get("cnn_ratio", 4),
        learn_pos_emb=mc.get("learn_pos_emb", True),
        embed_dim=mc["embed_dim"],
        depth=mc["depth"],
        decoder_depth=mc["decoder_depth"],
        num_heads=mc["num_heads"],
        mlp_ratio=mc.get("mlp_ratio", 4.0),
        drop_path=0.0,
        drop_rate=0.0,
        num_constant_vars=num_constant_vars,
        FusedAttn_option=FusedAttn.DEFAULT,
        input_refine_cnn=mc.get("input_refine_cnn", False),
        output_refine_cnn=mc.get("output_refine_cnn", False),
    )

    ckpt = torch.load(mc["checkpoint"], map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    del ckpt

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if rank == 0:
        n = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"v6 backbone loaded: {n:.2f}M params (frozen)", flush=True)
    return model


def load_diffusion_model(config, ckpt_path, device, rank):
    """Load trained EDM diffusion model (no FSDP for inference)."""
    dc = config["diffusion"]

    unet = ConditionalUNet(
        in_channels=dc.get("in_channels", 2),
        out_channels=dc.get("out_channels", 1),
        base_channels=dc.get("base_channels", 64),
        channel_mults=tuple(dc.get("channel_mults", [1, 2, 4, 4])),
        num_res_blocks=dc.get("num_res_blocks", 2),
        time_emb_dim=dc.get("time_emb_dim", 256),
        num_groups=dc.get("num_groups", 8),
    )
    edm = EDMPrecond(unet, sigma_data=dc.get("sigma_data", 1.0))

    if rank == 0:
        print(f"Loading diffusion checkpoint: {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Prefer EMA weights if available (smoother, better sample quality)
    if "ema_state_dict" in ckpt:
        edm.load_state_dict(ckpt["ema_state_dict"])
        if rank == 0:
            print("  Using EMA weights for visualization", flush=True)
    else:
        edm.load_state_dict(ckpt["model_state_dict"])
        if rank == 0:
            print("  No EMA in checkpoint, using training weights", flush=True)
    epoch = ckpt.get("epoch", "unknown")
    del ckpt

    edm = edm.to(device).eval()
    for p in edm.parameters():
        p.requires_grad_(False)

    if rank == 0:
        n = sum(p.numel() for p in edm.parameters()) / 1e6
        print(f"Diffusion model loaded: {n:.2f}M params (epoch {epoch})", flush=True)
    return edm, epoch


# ============================================================================
# Visualization
# ============================================================================

def save_visualization(input_lr, v6_pred, diff_pred, ground_truth,
                       out_vars, save_dir, index, epoch, rmse_v6, rmse_diff):
    """Save 4-panel comparison + error maps."""
    os.makedirs(save_dir, exist_ok=True)
    var_name = out_vars[0]

    # Always save numpy arrays
    np.save(os.path.join(save_dir, f"sample_{index}_input_lr.npy"), input_lr)
    np.save(os.path.join(save_dir, f"sample_{index}_v6_pred.npy"), v6_pred)
    np.save(os.path.join(save_dir, f"sample_{index}_diff_pred.npy"), diff_pred)
    np.save(os.path.join(save_dir, f"sample_{index}_ground_truth.npy"), ground_truth)

    if not HAS_MATPLOTLIB:
        print(f"  Saved numpy arrays for sample {index}", flush=True)
        return

    vmin = ground_truth.min()
    vmax = ground_truth.max()

    # ---- 4-panel comparison ----
    fig, axes = plt.subplots(1, 4, figsize=(32, 6))

    im0 = axes[0].imshow(input_lr, cmap="viridis", aspect="auto")
    axes[0].set_title("Input (LR mean, 0.1°)", fontsize=13)
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(v6_pred, cmap="turbo", vmin=vmin, vmax=vmax, aspect="auto")
    axes[1].set_title(f"v6 Prediction (RMSE={rmse_v6:.3f})", fontsize=13)
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(diff_pred, cmap="turbo", vmin=vmin, vmax=vmax, aspect="auto")
    axes[2].set_title(f"Diffusion Sample (RMSE={rmse_diff:.3f})", fontsize=13)
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    im3 = axes[3].imshow(ground_truth, cmap="turbo", vmin=vmin, vmax=vmax, aspect="auto")
    axes[3].set_title("Ground Truth (0.05°)", fontsize=13)
    plt.colorbar(im3, ax=axes[3], fraction=0.046)

    corr_v6 = np.corrcoef(v6_pred.flatten(), ground_truth.flatten())[0, 1]
    corr_diff = np.corrcoef(diff_pred.flatten(), ground_truth.flatten())[0, 1]

    fig.suptitle(
        f"HRRR {var_name} — Sample {index} (Diff Epoch {epoch})\n"
        f"v6: RMSE={rmse_v6:.4f}, Corr={corr_v6:.4f}  |  "
        f"Diffusion: RMSE={rmse_diff:.4f}, Corr={corr_diff:.4f}",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    fig_path = os.path.join(save_dir, f"sample_{index}_comparison.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    # ---- Error maps (v6 vs diffusion) ----
    fig2, axes2 = plt.subplots(1, 2, figsize=(20, 6))

    err_v6 = v6_pred - ground_truth
    err_diff = diff_pred - ground_truth
    abs_max = max(abs(err_v6.min()), abs(err_v6.max()), abs(err_diff.min()), abs(err_diff.max()))

    im_e0 = axes2[0].imshow(err_v6, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max, aspect="auto")
    axes2[0].set_title(f"v6 Error (bias={err_v6.mean():.4f})", fontsize=13)
    plt.colorbar(im_e0, ax=axes2[0], fraction=0.046)

    im_e1 = axes2[1].imshow(err_diff, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max, aspect="auto")
    axes2[1].set_title(f"Diffusion Error (bias={err_diff.mean():.4f})", fontsize=13)
    plt.colorbar(im_e1, ax=axes2[1], fraction=0.046)

    fig2.suptitle(f"Error Maps — Sample {index}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig2_path = os.path.join(save_dir, f"sample_{index}_error_maps.png")
    plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {fig_path}", flush=True)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = ArgumentParser(description="Visualize HRRR diffusion-refined predictions")
    parser.add_argument("config", type=str, help="Path to diffusion config YAML")
    parser.add_argument("--diff_checkpoint", type=str, required=True,
                        help="Path to trained diffusion checkpoint .pt")
    parser.add_argument("--index", type=int, default=0,
                        help="Starting test sample index (default: 0)")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to visualize (default: 5)")
    parser.add_argument("--num_steps", type=int, default=30,
                        help="Heun sampling steps (default: 30, more = sharper but slower)")
    parser.add_argument("--warm_start", action="store_true",
                        help="Use SDEdit warm-start sampling (start from v6 pred + noise)")
    parser.add_argument("--sigma_start", type=float, default=2.0,
                        help="Starting sigma for warm-start sampling (default: 2.0)")
    parser.add_argument("--save_dir", type=str, default="visualizations_diffusion",
                        help="Output directory (default: visualizations_diffusion)")
    parser.add_argument("--master-port", type=str, default="29500")
    args = parser.parse_args()

    # ---- Distributed setup ----
    os.environ.setdefault("MASTER_ADDR", os.environ.get("HOSTNAME", "localhost"))
    os.environ.setdefault("MASTER_PORT", args.master_port)
    os.environ.setdefault("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1"))
    os.environ.setdefault("RANK", os.environ.get("SLURM_PROCID", "0"))

    world_size = int(os.environ["WORLD_SIZE"])
    world_rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("SLURM_LOCALID", "0"))

    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    dist.init_process_group("nccl", timeout=timedelta(seconds=3600),
                            rank=world_rank, world_size=world_size)

    if world_rank == 0:
        print(f"Initialized {world_size} GPU(s) for visualization", flush=True)

    # ---- Config ----
    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed_everything(42)

    # ---- Data (batch_size=1 for visualization) ----
    dc = config["data"]
    in_vars = expand_variables(dc["variables"])
    out_vars = expand_variables(dc["out_variables"])

    data_module = cl.data.IterDataModule(
        "downscaling",
        dc["low_res_dir"],
        dc["high_res_dir"],
        in_vars,
        out_vars,
        data_par_size=world_size,
        batch_size=1,
        num_workers=2,
        buffer_size=100,
    )
    data_module.setup()
    test_dl = data_module.test_dataloader()

    # ---- Load models (no FSDP needed for inference with full checkpoint) ----
    v6_model = load_v6_backbone(config, device, in_vars, out_vars, world_rank)
    edm_model, diff_epoch = load_diffusion_model(
        config, args.diff_checkpoint, device, world_rank
    )

    # ---- Denormalization ----
    denorm = cl.load_transform("denormalize", data_module)

    # ---- Run visualization ----
    if world_rank == 0:
        if args.warm_start:
            print(f"\nWarm-start (SDEdit) sampling: σ_start={args.sigma_start}, {args.num_steps} Heun steps", flush=True)
        else:
            print(f"\nFull sampling from noise: σ_max=80, {args.num_steps} Heun steps", flush=True)
        print(f"Visualizing {args.num_samples} test samples starting at index {args.index}\n", flush=True)

    sample_count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dl):
            if batch_idx < args.index:
                continue
            if sample_count >= args.num_samples:
                break

            x, y, in_variables, out_variables = batch
            x, y = x.to(device), y.to(device)
            if x.dim() == 5:
                x = x.flatten(1, 2)
            if y.dim() == 5:
                y = y.flatten(1, 2)

            t0 = time.time()

            # 1. v6 conditioning
            y_cond = v6_model(x, in_variables, out_variables)

            # 2. Diffusion sampling (Heun ODE solver)
            y_sample = edm_model.sample(
                y_cond=y_cond,
                shape=y.shape,
                num_steps=args.num_steps,
                warm_start=args.warm_start,
                sigma_start=args.sigma_start,
            )

            elapsed = time.time() - t0

            # Denormalize
            y_cond_dn = denorm(y_cond)
            y_sample_dn = denorm(y_sample)
            y_dn = denorm(y)

            # Only rank 0 saves
            if world_rank == 0:
                v6_np = y_cond_dn[0, 0].cpu().float().numpy()
                diff_np = y_sample_dn[0, 0].cpu().float().numpy()
                truth_np = y_dn[0, 0].cpu().float().numpy()
                input_np = x[0].mean(dim=0).cpu().float().numpy()

                rmse_v6 = float(np.sqrt(np.mean((v6_np - truth_np) ** 2)))
                rmse_diff = float(np.sqrt(np.mean((diff_np - truth_np) ** 2)))

                print(f"Sample {batch_idx}: v6 RMSE={rmse_v6:.4f}, Diff RMSE={rmse_diff:.4f} "
                      f"({elapsed:.1f}s)", flush=True)

                save_visualization(
                    input_np, v6_np, diff_np, truth_np,
                    out_vars, args.save_dir, batch_idx, diff_epoch,
                    rmse_v6, rmse_diff,
                )

            sample_count += 1

    if world_rank == 0:
        print(f"\nDone! Results saved to {args.save_dir}/", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()