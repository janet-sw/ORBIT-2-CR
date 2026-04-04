"""
EDM (Elucidated Diffusion Model) preconditioning, loss, and sampling.

Implements the framework from Karras et al. 2022:
  "Elucidating the Design Space of Diffusion-Based Generative Models"

The key idea: wrap a raw denoiser network F_θ with preconditioning so that
its input/output magnitudes are well-behaved across all noise levels σ.

IMPORTANT (FSDP compatibility):
  ALL parameter access must go through ``forward()``, which is the only
  entry point that triggers FSDP's all-gather.  ``forward()`` supports
  two modes:

      mode="train"   → compute EDM training loss  (default)
      mode="denoise" → run preconditioned denoiser D(y_noisy; σ, y_cond)

  The Heun sampling loop lives in ``validate_epoch`` in the training
  script, calling ``edm_model(..., mode="denoise")`` at each step.

Usage
-----
    unet = ConditionalUNet(in_channels=2, out_channels=1, ...)
    edm  = EDMPrecond(unet, sigma_data=1.0)

    # Training step  (goes through FSDP forward)
    loss = edm_model(y_clean, y_cond, P_mean=-1.2, P_std=1.2)

    # Denoising step (also goes through FSDP forward)
    denoised = edm_model(y_noisy, y_cond, sigma=sigma, mode="denoise")

    # Non-FSDP sampling (e.g. standalone inference without FSDP)
    y_sample = edm.sample(y_cond, shape=(B, 1, 512, 1024), num_steps=30)
"""

import torch
import torch.nn as nn


class EDMPrecond(nn.Module):
    """
    EDM preconditioning wrapper around a conditional denoiser network.

    The raw network F_θ receives:
        - input:  c_in(σ) · y_noisy  concatenated with  y_cond
        - sigma:  c_noise(σ) = ln(σ) / 4

    The final denoised output is:
        D(y_noisy; σ, y_cond) = c_skip(σ) · y_noisy + c_out(σ) · F_θ(...)

    Parameters
    ----------
    model : nn.Module
        Raw denoiser network (e.g. ConditionalUNet).
        forward(x, c_noise) → Tensor, where x has (noisy + cond) channels.
    sigma_data : float
        RMS of the clean data distribution.  Set to 1.0 if data is
        standardized (mean=0, std=1).
    """

    def __init__(self, model, sigma_data=1.0):
        super().__init__()
        self.model = model
        self.sigma_data = sigma_data

    # ------------------------------------------------------------------
    # Preconditioning coefficients (Table 1 in Karras et al. 2022)
    # ------------------------------------------------------------------

    def c_skip(self, sigma):
        return self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma):
        return sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()

    def c_in(self, sigma):
        return 1.0 / (sigma ** 2 + self.sigma_data ** 2).sqrt()

    @staticmethod
    def c_noise(sigma):
        return sigma.log() / 4.0

    def loss_weight(self, sigma):
        """Per-sample loss weight λ(σ)  (Eq. 8 in Karras et al.)."""
        return (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

    # ------------------------------------------------------------------
    # Preconditioned denoiser (internal — called from within forward)
    # ------------------------------------------------------------------

    def _denoise(self, y_noisy, sigma, y_cond):
        """
        D(y_noisy; σ, y_cond) = c_skip · y_noisy + c_out · F_θ(c_in · y_noisy ‖ y_cond, c_noise)
        """
        c_skip = self.c_skip(sigma)
        c_out = self.c_out(sigma)
        c_in = self.c_in(sigma)
        c_noise = self.c_noise(sigma.reshape(sigma.shape[0]))   # [B]

        x_in = torch.cat([c_in * y_noisy, y_cond], dim=1)
        F_out = self.model(x_in, c_noise)

        return c_skip * y_noisy + c_out * F_out

    # ------------------------------------------------------------------
    # Forward  (FSDP-compatible, dual-mode entry point)
    # ------------------------------------------------------------------

    def forward(self, x, y_cond, sigma=None, P_mean=-1.2, P_std=1.2, mode="train"):
        """
        Unified forward that triggers FSDP's all-gather in every code path.

        Parameters
        ----------
        mode="train" (default):
            x     = y_clean  [B, C, H, W]   — clean target
            y_cond            [B, C, H, W]   — v6 conditioning
            Returns: scalar loss

        mode="denoise":
            x     = y_noisy  [B, C, H, W]   — noisy sample
            y_cond            [B, C, H, W]   — v6 conditioning
            sigma             [B, 1, 1, 1]   — noise level  (required)
            Returns: denoised prediction  [B, C, H, W]
        """
        if mode == "denoise":
            return self._denoise(x, sigma, y_cond)

        # ---- mode == "train" ----
        y_clean = x
        B = y_clean.shape[0]
        device = y_clean.device

        # 1. Sample sigma ~ LogNormal(P_mean, P_std²)
        ln_sigma = torch.randn(B, device=device, dtype=torch.float32) * P_std + P_mean
        sigma = ln_sigma.exp()[:, None, None, None].to(y_clean.dtype)

        # 2. Corrupt
        noise = torch.randn_like(y_clean)
        y_noisy = y_clean + sigma * noise

        # 3. Denoise
        y_denoised = self._denoise(y_noisy, sigma, y_cond)

        # 4. Weighted MSE
        weight = self.loss_weight(sigma)
        loss = (weight * (y_denoised - y_clean) ** 2).mean()

        return loss

    # ------------------------------------------------------------------
    # Standalone sampler (for non-FSDP inference, e.g. visualization)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        y_cond,
        shape,
        num_steps=30,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        warm_start=False,
        sigma_start=None,
    ):
        """
        Heun's 2nd-order ODE sampler  (σ_max → σ_min).

        When ``warm_start=True``, uses SDEdit-style initialization:
        start from ``y_cond + σ_start · ε`` instead of pure noise at
        σ_max, and only denoise over the range [σ_start, σ_min].
        This preserves the large-scale structure from the v6 backbone
        while letting the diffusion model sharpen fine details.

        NOTE: This calls ``_denoise`` directly, so it only works when
        parameters are fully materialized (non-FSDP, or after loading a
        full checkpoint).  For FSDP validation, use the Heun loop in the
        training script which calls ``edm_model(..., mode="denoise")``.
        """
        device = y_cond.device
        dtype = y_cond.dtype

        # Determine effective starting sigma
        if warm_start:
            effective_sigma_max = sigma_start if sigma_start is not None else 2.0
        else:
            effective_sigma_max = sigma_max

        step_indices = torch.arange(num_steps, device=device, dtype=torch.float64)
        t_steps = (
            effective_sigma_max ** (1.0 / rho)
            + step_indices / (num_steps - 1) * (sigma_min ** (1.0 / rho) - effective_sigma_max ** (1.0 / rho))
        ) ** rho
        t_steps = torch.cat([t_steps, torch.zeros(1, device=device, dtype=torch.float64)])
        t_steps = t_steps.to(dtype)

        # Initialize: warm-start from v6 prediction, or pure noise
        if warm_start:
            x = y_cond + t_steps[0] * torch.randn(*shape, device=device, dtype=dtype)
        else:
            x = torch.randn(*shape, device=device, dtype=dtype) * t_steps[0]

        for i in range(num_steps):
            t_cur = t_steps[i]
            t_next = t_steps[i + 1]
            sigma = t_cur.reshape(1, 1, 1, 1).expand(shape[0], -1, -1, -1)

            denoised = self._denoise(x, sigma, y_cond)
            d_cur = (x - denoised) / t_cur
            x_next = x + (t_next - t_cur) * d_cur

            if t_next > 0:
                sigma_next = t_next.reshape(1, 1, 1, 1).expand(shape[0], -1, -1, -1)
                denoised_next = self._denoise(x_next, sigma_next, y_cond)
                d_next = (x_next - denoised_next) / t_next
                x_next = x + (t_next - t_cur) * (d_cur + d_next) / 2.0

            x = x_next

        return x