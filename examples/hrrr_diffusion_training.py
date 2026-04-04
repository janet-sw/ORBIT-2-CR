"""
Diffusion refinement training for HRRR composite reflectivity.

Two-stage approach:
  Stage 1 (already done): MSE-trained Res_Slim_ViT produces blurry mean prediction.
  Stage 2 (this script):  EDM-based conditional diffusion sharpens the prediction.

The frozen v6 backbone generates conditioning on the fly.  A lightweight
ConditionalUNet (~30M params) learns to denoise and is wrapped with FSDP.

Usage (SLURM):
    srun python hrrr_diffusion_training.py configs/hrrr_diffusion_refine.yaml
"""

# Standard library
import os
import sys
import time
import yaml
import functools
from datetime import timedelta

# PyTorch
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
    FullOptimStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

# Project
import climate_learn as cl
from climate_learn.data.processing.era5_constants import (
    PRESSURE_LEVEL_VARS,
    DEFAULT_PRESSURE_LEVELS,
    CONSTANTS,
)
from climate_learn.models.hub.res_slimvit import Res_Slim_ViT
from climate_learn.models.hub.edm_unet import ConditionalUNet, ResBlock
from climate_learn.models.hub.edm import EDMPrecond
from climate_learn.utils.fused_attn import FusedAttn

# Local
from utils import seed_everything


# ============================================================================
# Helpers
# ============================================================================

def log_gpu_memory(device, msg=""):
    alloc = torch.cuda.memory_allocated(device) / 1024**3
    peak = torch.cuda.max_memory_allocated(device) / 1024**3
    resv = torch.cuda.memory_reserved(device) / 1024**3
    print(f"{msg} Allocated: {alloc:.2f}GB, Peak: {peak:.2f}GB, Reserved: {resv:.2f}GB", flush=True)


# ============================================================================
# EMA (Exponential Moving Average) — CPU-based, FSDP-compatible
# ============================================================================

class EMATracker:
    """
    Maintains an exponential moving average of model weights on CPU (rank 0).

    After each training epoch, call ``update()`` with the FSDP full state dict.
    For validation, call ``state_dict()`` to get the EMA weights.

    Memory: ~25M params × 4 bytes = ~100 MB on CPU.  No GPU overhead.
    """

    def __init__(self, decay=0.9999):
        self.decay = decay
        self.shadow = {}      # {name: Tensor} on CPU, float32
        self.initialized = False

    @torch.no_grad()
    def update(self, model_state_dict):
        """Update EMA with a new full state dict (CPU tensors)."""
        if not self.initialized:
            # First call: copy weights as-is
            for name, param in model_state_dict.items():
                self.shadow[name] = param.clone().float()
            self.initialized = True
        else:
            d = self.decay
            for name, param in model_state_dict.items():
                if name in self.shadow:
                    self.shadow[name].mul_(d).add_(param.float(), alpha=1.0 - d)

    def state_dict(self):
        """Return a copy of the EMA weights."""
        return {k: v.clone() for k, v in self.shadow.items()}


def parse_config(path, rank):
    if rank == 0:
        print(f"\nLoading config from: {path}", flush=True)
    with open(path) as f:
        config = yaml.safe_load(f)
    return config


def expand_variables(raw_vars):
    """Expand pressure-level shorthands into individual variable names."""
    expanded = []
    for var in raw_vars:
        if var in PRESSURE_LEVEL_VARS:
            for level in DEFAULT_PRESSURE_LEVELS:
                expanded.append(f"{var}_{level}")
        else:
            expanded.append(var)
    return expanded


# ============================================================================
# Load frozen v6 backbone
# ============================================================================

def load_v6_backbone(config, device, in_vars, out_vars, rank):
    """
    Load the MSE-trained Res_Slim_ViT (v6) in eval mode on every GPU.

    The checkpoint was saved with FSDP FULL_STATE_DICT (rank-0 only),
    so it can be loaded into a regular non-FSDP model.
    """
    mc = config["backbone"]
    all_vars = list(dict.fromkeys(in_vars + out_vars))  # preserve order, deduplicate

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
        drop_path=0.0,       # no stochastic depth at inference
        drop_rate=0.0,       # no dropout at inference
        num_constant_vars=num_constant_vars,
        FusedAttn_option=FusedAttn.DEFAULT,
        input_refine_cnn=mc.get("input_refine_cnn", False),
        output_refine_cnn=mc.get("output_refine_cnn", False),
    )

    # Load checkpoint
    ckpt_path = mc["checkpoint"]
    if rank == 0:
        print(f"Loading v6 backbone from: {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"v6 backbone loaded: {n_params:.2f}M params (frozen)", flush=True)

    return model


# ============================================================================
# Create diffusion model
# ============================================================================

def create_diffusion_model(config, device, rank):
    """Create ConditionalUNet wrapped with EDMPrecond."""
    dc = config["diffusion"]

    unet = ConditionalUNet(
        in_channels=dc.get("in_channels", 2),        # noisy + v6_pred
        out_channels=dc.get("out_channels", 1),
        base_channels=dc.get("base_channels", 64),
        channel_mults=tuple(dc.get("channel_mults", [1, 2, 4, 4])),
        num_res_blocks=dc.get("num_res_blocks", 2),
        time_emb_dim=dc.get("time_emb_dim", 256),
        num_groups=dc.get("num_groups", 8),
    )

    edm = EDMPrecond(unet, sigma_data=dc.get("sigma_data", 1.0))

    if rank == 0:
        n_params = sum(p.numel() for p in edm.parameters()) / 1e6
        print(f"Diffusion model created: {n_params:.2f}M params", flush=True)

    return edm


# ============================================================================
# Data module
# ============================================================================

def create_data_module(config, world_size, rank):
    """Create the same downscaling data module used for v6 training."""
    dc = config["data"]
    tc = config["trainer"]

    in_vars = expand_variables(dc["variables"])
    out_vars = expand_variables(dc["out_variables"])

    if rank == 0:
        print(f"Input variables ({len(in_vars)}):  {in_vars}", flush=True)
        print(f"Output variables ({len(out_vars)}): {out_vars}", flush=True)

    data_module = cl.data.IterDataModule(
        "downscaling",
        dc["low_res_dir"],
        dc["high_res_dir"],
        in_vars,
        out_vars,
        data_par_size=world_size,
        batch_size=tc["batch_size"],
        num_workers=tc.get("num_workers", 2),
        buffer_size=tc.get("buffer_size", 100),
    )
    data_module.setup()

    return data_module, in_vars, out_vars


# ============================================================================
# Training step
# ============================================================================

def training_step(v6_model, edm_model, batch, device, in_vars, out_vars, edm_config):
    """
    One training step:
      1. Run frozen v6 to get blurry conditioning
      2. Compute EDM diffusion loss
    """
    x, y, in_variables, out_variables = batch
    x, y = x.to(device), y.to(device)

    # Flatten history dim if present
    if x.dim() == 5:
        x = x.flatten(1, 2)
    if y.dim() == 5:
        y = y.flatten(1, 2)

    # 1. Generate conditioning from frozen backbone (no grad)
    with torch.no_grad():
        y_cond = v6_model(x, in_variables, out_variables)
        # Clamp precipitation-like variables to >= 0
        y_cond = y_cond.clamp(min=-3.0)   # reasonable lower bound in normalized space

    # 2. EDM loss — call edm_model() (NOT .training_loss()) so that
    #    FSDP's forward hook triggers the parameter all-gather.
    loss = edm_model(
        y,                                          # x = y_clean in train mode
        y_cond,
        P_mean=edm_config.get("P_mean", -1.2),
        P_std=edm_config.get("P_std", 1.2),
    )

    return loss


# ============================================================================
# Validation / sampling
# ============================================================================

def heun_sample(edm_model, y_cond, shape, num_steps, sigma_max=80.0, sigma_min=0.002, rho=7.0):
    """
    Heun's 2nd-order ODE sampler that calls edm_model(..., mode="denoise")
    at each step, so every denoising call goes through FSDP's forward hook.
    """
    device = y_cond.device
    dtype = y_cond.dtype

    step_indices = torch.arange(num_steps, device=device, dtype=torch.float64)
    t_steps = (
        sigma_max ** (1.0 / rho)
        + step_indices / (num_steps - 1) * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
    ) ** rho
    t_steps = torch.cat([t_steps, torch.zeros(1, device=device, dtype=torch.float64)])
    t_steps = t_steps.to(dtype)

    x = torch.randn(*shape, device=device, dtype=dtype) * t_steps[0]

    for i in range(num_steps):
        t_cur = t_steps[i]
        t_next = t_steps[i + 1]
        sigma = t_cur.reshape(1, 1, 1, 1).expand(shape[0], -1, -1, -1)

        # Each call goes through FSDP forward → all-gather → denoise
        denoised = edm_model(x, y_cond, sigma=sigma, mode="denoise")
        d_cur = (x - denoised) / t_cur
        x_next = x + (t_next - t_cur) * d_cur

        if t_next > 0:
            sigma_next = t_next.reshape(1, 1, 1, 1).expand(shape[0], -1, -1, -1)
            denoised_next = edm_model(x_next, y_cond, sigma=sigma_next, mode="denoise")
            d_next = (x_next - denoised_next) / t_next
            x_next = x + (t_next - t_cur) * (d_cur + d_next) / 2.0

        x = x_next

    return x


@torch.no_grad()
def validate_epoch(v6_model, edm_model, val_dataloader, device, in_vars, out_vars,
                   num_steps, rank):
    """
    Run deterministic Heun sampling on a few validation batches and compute
    RMSE between the diffusion sample and the ground truth.

    Every denoising call goes through ``edm_model(..., mode="denoise")``,
    which triggers FSDP's forward hook and handles bfloat16 properly.
    No ``summon_full_params`` needed.
    """
    edm_model.eval()
    total_mse = 0.0
    total_mse_v6 = 0.0
    num_samples = 0

    for batch_idx, batch in enumerate(val_dataloader):
        if batch_idx >= 4:       # only evaluate a few batches (sampling is slow)
            break

        x, y, in_variables, out_variables = batch
        x, y = x.to(device), y.to(device)
        if x.dim() == 5:
            x = x.flatten(1, 2)
        if y.dim() == 5:
            y = y.flatten(1, 2)

        # v6 conditioning
        y_cond = v6_model(x, in_variables, out_variables)

        # Heun sampling — each denoise step calls edm_model() through FSDP
        y_sample = heun_sample(
            edm_model, y_cond, shape=y.shape, num_steps=num_steps,
        )

        # MSE
        total_mse += ((y_sample - y) ** 2).mean().item() * y.shape[0]
        total_mse_v6 += ((y_cond - y) ** 2).mean().item() * y.shape[0]
        num_samples += y.shape[0]

    # Reduce across ranks
    stats = torch.tensor([total_mse, total_mse_v6, num_samples], device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    mse_diff, mse_v6, n = stats[0].item(), stats[1].item(), stats[2].item()

    edm_model.train()

    if rank == 0 and n > 0:
        rmse_diff = (mse_diff / n) ** 0.5
        rmse_v6 = (mse_v6 / n) ** 0.5
        print(f"  [Val] Diffusion RMSE: {rmse_diff:.4f}  |  v6-only RMSE: {rmse_v6:.4f}", flush=True)

    return (mse_diff / max(n, 1)) ** 0.5


# ============================================================================
# Checkpoint save / load
# ============================================================================

def save_checkpoint(edm_model, optimizer, scheduler, epoch, save_path, rank, ema=None):
    """Save full state dict (rank-0 only, FSDP-aware), including EMA."""
    os.makedirs(save_path, exist_ok=True)
    file_name = os.path.join(save_path, f"checkpoint_epoch_{epoch:04d}.pt")

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    optim_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)

    with FSDP.state_dict_type(edm_model, StateDictType.FULL_STATE_DICT, save_policy, optim_policy):
        model_state = edm_model.state_dict()
        optim_state = FSDP.optim_state_dict(edm_model, optimizer)

    if rank == 0:
        save_dict = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optim_state,
            "scheduler_state_dict": scheduler.state_dict(),
        }
        if ema is not None and ema.initialized:
            save_dict["ema_state_dict"] = ema.state_dict()
        torch.save(save_dict, file_name)
        print(f"  Checkpoint saved: {file_name}", flush=True)

    dist.barrier()


def load_checkpoint(ckpt_path, edm_model, optimizer, scheduler, rank, ema=None):
    """Load checkpoint and return the next epoch to train."""
    if rank == 0:
        print(f"Resuming from: {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    edm_model.load_state_dict(ckpt["model_state_dict"])

    optim_state = FSDP.optim_state_dict_to_load(
        edm_model, optimizer, ckpt["optimizer_state_dict"]
    )
    optimizer.load_state_dict(optim_state)
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    # Restore EMA if available
    if ema is not None and "ema_state_dict" in ckpt:
        ema.shadow = {k: v.float() for k, v in ckpt["ema_state_dict"].items()}
        ema.initialized = True
        if rank == 0:
            print("  EMA weights restored from checkpoint", flush=True)
    elif ema is not None:
        # Old checkpoint without EMA — initialize from model weights
        ema.update(ckpt["model_state_dict"])
        if rank == 0:
            print("  EMA initialized from model weights (no EMA in checkpoint)", flush=True)

    return ckpt["epoch"] + 1


# ============================================================================
# Main
# ============================================================================

def main(device):
    world_size = int(os.environ["SLURM_NTASKS"])
    world_rank = dist.get_rank()
    local_rank = int(os.environ["SLURM_LOCALID"])

    # ---- Config ----
    config_path = sys.argv[1]
    config = parse_config(config_path, world_rank)
    tc = config["trainer"]
    dc = config["diffusion"]

    max_epochs = tc["max_epochs"]
    data_type = tc.get("data_type", "bfloat16")
    cp_save_path = tc.get("checkpoint_save_path", "checkpoints/diffusion")
    ckpt_path = tc.get("checkpoint", None)
    val_interval = tc.get("val_interval", 10)
    save_interval = tc.get("save_interval", 5)

    seed_everything(42)

    # ---- Data ----
    data_module, in_vars, out_vars = create_data_module(config, world_size, world_rank)
    train_dl = data_module.train_dataloader()
    val_dl = data_module.val_dataloader()

    if world_rank == 0:
        log_gpu_memory(device, "After data module")

    # ---- Frozen v6 backbone ----
    v6_model = load_v6_backbone(config, device, in_vars, out_vars, world_rank)

    if world_rank == 0:
        log_gpu_memory(device, "After v6 load")

    # ---- Diffusion model ----
    edm_model = create_diffusion_model(config, device, world_rank)

    # ---- FSDP wrapping ----
    if data_type == "bfloat16":
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    else:
        mp_policy = None

    # Wrap ResBlock for FSDP (similar to how Block is wrapped for the ViT)
    auto_wrap = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={ResBlock},
    )

    edm_model = FSDP(
        edm_model,
        device_id=local_rank,
        mixed_precision=mp_policy,
        auto_wrap_policy=auto_wrap,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=True,
    )

    if world_rank == 0:
        log_gpu_memory(device, "After FSDP wrapping")

    # ---- Optimizer + Scheduler ----
    num_gpus = world_size
    base_lr = float(dc.get("base_lr", 2e-4))
    # NOTE: Unlike supervised training, diffusion models (EDM) are
    # sensitive to LR and do NOT benefit from sqrt(N) batch scaling.
    # Use the base_lr directly (Karras et al. 2022 uses a flat 2e-4).
    lr = base_lr

    optimizer = torch.optim.AdamW(
        edm_model.parameters(),
        lr=lr,
        weight_decay=float(dc.get("weight_decay", 1e-4)),
        betas=(float(dc.get("beta_1", 0.9)), float(dc.get("beta_2", 0.999))),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epochs,
        eta_min=float(dc.get("eta_min", 1e-6)),
    )

    # Gradient scaler for bfloat16
    scaler = ShardedGradScaler(init_scale=8192, growth_interval=100) if data_type == "bfloat16" else None

    # ---- EMA ----
    ema_decay = float(dc.get("ema_decay", 0.9999))
    ema = EMATracker(decay=ema_decay) if world_rank == 0 else None
    if world_rank == 0:
        print(f"EMA enabled with decay={ema_decay}", flush=True)

    # ---- Resume ----
    start_epoch = 0
    if ckpt_path:
        start_epoch = load_checkpoint(ckpt_path, edm_model, optimizer, scheduler, world_rank, ema=ema)

    # ---- Training loop ----
    if world_rank == 0:
        print(f"\n{'='*72}", flush=True)
        print(f"Starting diffusion training: epochs {start_epoch}..{max_epochs-1}", flush=True)
        print(f"LR: {lr:.2e}  |  batch_size: {tc['batch_size']}  |  GPUs: {num_gpus}", flush=True)
        print(f"{'='*72}\n", flush=True)

    for epoch in range(start_epoch, max_epochs):
        t0 = time.time()
        edm_model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_dl):
            optimizer.zero_grad(set_to_none=True)

            loss = training_step(
                v6_model, edm_model, batch, device, in_vars, out_vars, dc,
            )

            # Backward
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(edm_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                # Floor the scaler to avoid underflow on ROCm
                if hasattr(scaler, "_scale") and scaler._scale < 128:
                    scaler._scale = torch.tensor(128.0, device=device)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(edm_model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx == 0 and world_rank == 0:
                print(
                    f"Epoch {epoch} | Batch 0 | Loss: {loss.item():.6f} "
                    f"| LR: {optimizer.param_groups[0]['lr']:.2e}",
                    flush=True,
                )

        scheduler.step()

        # Average loss across ranks
        avg_loss = torch.tensor(total_loss / max(num_batches, 1), device=device)
        dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
        elapsed = time.time() - t0

        if world_rank == 0:
            print(
                f"Epoch {epoch} completed | Avg Loss: {avg_loss.item():.6f} "
                f"| Time: {elapsed:.1f}s | LR: {optimizer.param_groups[0]['lr']:.2e}",
                flush=True,
            )

        # ---- Update EMA ----
        # ALL ranks must enter state_dict_type (it triggers all-gather),
        # but only rank 0 gets the full state and updates the EMA.
        ema_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(edm_model, StateDictType.FULL_STATE_DICT, ema_save_policy):
            full_state = edm_model.state_dict()   # rank 0 gets full, others get {}
        if ema is not None:
            ema.update(full_state)
        del full_state

        # Validation
        if (epoch + 1) % val_interval == 0:
            validate_epoch(
                v6_model, edm_model, val_dl, device, in_vars, out_vars,
                num_steps=dc.get("val_num_steps", 20), rank=world_rank,
            )

        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            save_checkpoint(edm_model, optimizer, scheduler, epoch, cp_save_path, world_rank, ema=ema)

    # Final save
    save_checkpoint(edm_model, optimizer, scheduler, max_epochs - 1, cp_save_path, world_rank, ema=ema)

    if world_rank == 0:
        print("\nTraining complete.", flush=True)


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    # SLURM env
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    os.environ.setdefault("MASTER_ADDR", os.environ.get("SLURM_LAUNCH_NODE_IPADDR", "localhost"))
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", os.environ.get("SLURM_PROCID", "0"))
    os.environ.setdefault("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1"))

    dist.init_process_group("nccl", timeout=timedelta(seconds=7200))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    main(device)

    dist.destroy_process_group()