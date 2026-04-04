"""
Conditional UNet denoiser for EDM-based diffusion refinement.

Takes concatenated (noisy_target, v6_prediction) as input and predicts the
denoised target.  Designed to be lightweight (~30M params) so it trains
within the same compute budget as the deterministic backbone.

Architecture
------------
- 4-level encoder–decoder with skip connections
- ResBlocks with GroupNorm + SiLU + scale-shift time conditioning
- No self-attention (keeps memory low at 512×1024 resolution)
- Sinusoidal time embedding → MLP for sigma conditioning

Reference: Karras et al. 2022 "Elucidating the Design Space of
Diffusion-Based Generative Models" (EDM)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Building blocks
# ============================================================================

class SinusoidalPosEmb(nn.Module):
    """Map scalar sigma → D-dimensional sinusoidal embedding."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """x: [B] → [B, dim]"""
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=x.device, dtype=torch.float32)
            * -(math.log(10000.0) / (half - 1))
        )
        args = x.float()[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1).to(x.dtype)


class TimeEmbedding(nn.Module):
    """Sinusoidal embedding → 2-layer MLP."""

    def __init__(self, base_dim, emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            SinusoidalPosEmb(base_dim),
            nn.Linear(base_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, t):
        """t: [B] → [B, emb_dim]"""
        return self.mlp(t)


class ResBlock(nn.Module):
    """
    Residual block with time conditioning via adaptive scale-shift.

    norm1 → SiLU → conv1 → (+ time scale/shift) → norm2 → SiLU → conv2 → + skip
    """

    def __init__(self, in_ch, out_ch, time_emb_dim, num_groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch * 2),  # scale + shift
        )
        self.norm2 = nn.GroupNorm(num_groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        # 1x1 conv for channel change; identity when in_ch == out_ch
        self.skip_proj = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

        # Zero-init last conv so the block starts as identity
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x, t_emb):
        """
        x: [B, C, H, W]
        t_emb: [B, time_emb_dim]
        """
        h = self.conv1(F.silu(self.norm1(x)))

        # Scale-shift from time embedding
        t = self.time_proj(t_emb)[:, :, None, None]
        scale, shift = t.chunk(2, dim=1)
        h = self.norm2(h) * (1.0 + scale) + shift

        h = self.conv2(F.silu(h))
        return h + self.skip_proj(x)


class Downsample(nn.Module):
    """2× spatial downsampling via strided convolution."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """2× spatial upsampling via nearest-neighbor + convolution."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ============================================================================
# Conditional UNet
# ============================================================================

class ConditionalUNet(nn.Module):
    """
    Lightweight UNet for diffusion refinement of weather predictions.

    Parameters
    ----------
    in_channels : int
        Number of input channels.  Default 2 = (noisy_target, v6_prediction).
    out_channels : int
        Number of output channels.  Default 1 = denoised composite_reflectivity.
    base_channels : int
        Channel width at the first encoder level.
    channel_mults : tuple of int
        Multipliers for each encoder/decoder level.
    num_res_blocks : int
        Number of ResBlocks per encoder level.
    time_emb_dim : int
        Dimensionality of the time embedding vector.
    num_groups : int
        Group count for GroupNorm (must divide all channel widths).
    """

    def __init__(
        self,
        in_channels=2,
        out_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4, 4),
        num_res_blocks=2,
        time_emb_dim=256,
        num_groups=8,
    ):
        super().__init__()
        self.channel_mults = channel_mults
        self.num_res_blocks = num_res_blocks

        # --- Time embedding ---
        self.time_emb = TimeEmbedding(base_channels, time_emb_dim)

        # --- Input projection ---
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # --- Encoder ---
        self.enc_blocks = nn.ModuleList()  # flat list of ResBlocks
        self.downsamples = nn.ModuleList()

        skip_channels = [base_channels]     # track channels for decoder skips
        ch = base_channels
        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.enc_blocks.append(ResBlock(ch, out_ch, time_emb_dim, num_groups))
                ch = out_ch
                skip_channels.append(ch)
            if level < len(channel_mults) - 1:
                self.downsamples.append(Downsample(ch))
                skip_channels.append(ch)

        # --- Bottleneck ---
        self.mid1 = ResBlock(ch, ch, time_emb_dim, num_groups)
        self.mid2 = ResBlock(ch, ch, time_emb_dim, num_groups)

        # --- Decoder ---
        self.dec_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for level in reversed(range(len(channel_mults))):
            out_ch = base_channels * channel_mults[level]
            # +1 blocks per level to absorb the extra skip from downsample
            for _ in range(num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                self.dec_blocks.append(
                    ResBlock(ch + skip_ch, out_ch, time_emb_dim, num_groups)
                )
                ch = out_ch
            if level > 0:
                self.upsamples.append(Upsample(ch))

        # --- Output projection ---
        self.out_norm = nn.GroupNorm(num_groups, ch)
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(self, x, c_noise):
        """
        Parameters
        ----------
        x : Tensor [B, in_channels, H, W]
            Concatenated (scaled_noisy_target, conditioning).
        c_noise : Tensor [B]
            Noise-level embedding input, typically ln(σ)/4.

        Returns
        -------
        Tensor [B, out_channels, H, W]
        """
        t = self.time_emb(c_noise)
        h = self.input_conv(x)

        # ---- Encoder ----
        skips = [h]
        block_idx = 0
        down_idx = 0
        for level in range(len(self.channel_mults)):
            for _ in range(self.num_res_blocks):
                h = self.enc_blocks[block_idx](h, t)
                block_idx += 1
                skips.append(h)
            if level < len(self.channel_mults) - 1:
                h = self.downsamples[down_idx](h)
                down_idx += 1
                skips.append(h)

        # ---- Bottleneck ----
        h = self.mid1(h, t)
        h = self.mid2(h, t)

        # ---- Decoder ----
        block_idx = 0
        up_idx = 0
        for level in reversed(range(len(self.channel_mults))):
            for _ in range(self.num_res_blocks + 1):
                h = torch.cat([h, skips.pop()], dim=1)
                h = self.dec_blocks[block_idx](h, t)
                block_idx += 1
            if level > 0:
                h = self.upsamples[up_idx](h)
                up_idx += 1

        return self.out_conv(F.silu(self.out_norm(h)))
