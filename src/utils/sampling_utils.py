from typing import Optional

import torch
import torch.nn.functional as F


# ============================================
# Noise Schedulers (how to compute z from x0 and noise)
# ============================================

def add_noise(x0, noise, t, config, cond_seq_mask=None):
    """Flow-matching interpolation z = t*x0 + (1-t)*noise*scale, preserving cond tokens."""
    t_expanded = t.reshape(-1, 1, 1)
    z = t_expanded * x0 + (1 - t_expanded) * noise * config.denoiser_noise_scale
    if cond_seq_mask is not None:
        z = cond_seq_mask * x0 + (1 - cond_seq_mask) * z
    return z


# ============================================
# Time Schedulers (how to sample t)
# ============================================

def sample_timesteps(
    batch_size: int,
    P_mean: float = -0.8,
    P_std: float = 0.8,
    time_schedule: str = 'logit_normal',
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
):
    """Sample timesteps using various time schedules.

    Args:
        batch_size: Number of samples
        P_mean: Mean for logit-normal distribution
        P_std: Std for logit-normal distribution
        time_schedule: 'logit_normal' or 'uniform'

    Returns:
        Sampled timesteps in [0, 1]
    """
    if time_schedule == 'logit_normal':
        z = torch.randn((batch_size,), dtype=dtype, device=device) * P_std + P_mean
        return torch.sigmoid(z)
    if time_schedule == 'uniform':
        return torch.rand((batch_size,), dtype=dtype, device=device)
    raise ValueError(f"Unknown time_schedule: {time_schedule}")


def get_sampling_steps(
    n_steps: int, time_schedule: str = "logit_normal",
    P_mean: float = -0.8, P_std: float = 0.8,
    device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a length-(n_steps+1) tensor of t values in [0, 1] for a sampling run.

    - "uniform": evenly-spaced linspace from 0 to 1 (deterministic).
    - "logit_normal": sorted logit-normal samples with 0 / 1 endpoints (random).
    """
    if time_schedule == "uniform":
        return torch.linspace(0.0, 1.0, n_steps + 1, dtype=dtype, device=device)
    if time_schedule == "logit_normal":
        steps = sample_timesteps(
            batch_size=n_steps - 1,
            P_mean=P_mean, P_std=P_std, time_schedule=time_schedule,
            device=device, dtype=dtype,
        )
        steps = torch.sort(steps).values
        endpoints_lo = torch.zeros((1,), dtype=dtype, device=steps.device)
        endpoints_hi = torch.ones((1,), dtype=dtype, device=steps.device)
        return torch.cat([endpoints_lo, steps, endpoints_hi], dim=0)
    raise ValueError(f"Unknown time_schedule: {time_schedule}")


# ============================================
# CFG Scale Sampling (how to sample cfg scale)
# ============================================

def sample_cfg_scale(batch_size, cfg_min=0.0, cfg_max=3.0,
                     dtype=torch.float32, device=None):
    """Sample CFG scale from log-uniform distribution in [cfg_min, cfg_max]."""
    u = torch.rand((batch_size,), dtype=dtype, device=device)
    a = float(1.0 + cfg_min)
    b = float(1.0 + cfg_max)
    log_ratio = torch.tensor(b / a, dtype=dtype, device=u.device).log()
    return a * torch.exp(u * log_ratio) - 1.0


# ============================================
# Conditioning helpers (preserve clean tokens during sampling)
# ============================================

def restore_cond(z_updated, cond_seq, cond_seq_mask):
    """Restore clean conditioning tokens in z after a denoising step."""
    mask = cond_seq_mask
    target_ndim = max(z_updated.dim(), cond_seq.dim())
    while mask.dim() < target_ndim:
        mask = mask.unsqueeze(-1)
    return torch.where(mask > 0, cond_seq, z_updated)


def restore_vx(v, x, cond_seq, cond_seq_mask):
    """Restore cond positions: x -> clean cond_seq, v -> 0 (cond tokens don't move)."""
    if cond_seq is not None:
        x = restore_cond(x, cond_seq, cond_seq_mask)
        v = restore_cond(v, torch.zeros_like(cond_seq), cond_seq_mask)
    return v, x


# ============================================
# Flow-matching forward passes (with optional self-cond / CFG)
# ============================================

def net_out_to_v_x(net_out, z, t, t_eps=5e-2):
    """Convert x_pred network output to v and x.

    When the model returns a tuple (denoised_output, decoder_logits),
    decoder logits are discarded here (used separately in training).
    """
    if isinstance(net_out, tuple):
        net_out = net_out[0]
    t_reshaped = t.reshape(-1, 1, 1)
    x = net_out
    denom = torch.clamp(1.0 - t_reshaped, min=t_eps)
    v = (x - z) / denom
    return v, x


def _forward_sample_self_cond(
    model, z, t_batch, x_pred_prev, config,
    self_cond_cfg_scale, cond_seq, cond_seq_mask,
    sc_noise_scale=0.0,
):
    """Forward pass with self-conditioning.

    Args:
        sc_noise_scale: if > 0, add isotropic Gaussian noise to x_pred_prev
            before concatenating as self-conditioning input.  Noise decays
            as (1-t)² so perturbation vanishes at t→1.  Inspired by LeWM's
            SIGReg — the noise prevents the self-conditioning from collapsing
            into a single mode.
    """
    t_eps = config.t_eps
    self_cond_prob = config.self_cond_prob

    def _restore(v, x):
        return restore_vx(v, x, cond_seq=cond_seq, cond_seq_mask=cond_seq_mask)

    def _perturb_x_pred(xp, t_b):
        """Add isotropic noise that decays as t→1."""
        scn = getattr(config, '_sc_noise_scale', 0.0)
        if scn <= 0 or xp is None:
            return xp
        # Decay factor: (1-t)² — strong near t≈0, gone at t=1
        decay = (1.0 - t_b.reshape(-1, 1, 1)) ** 2
        noise = scn * decay * torch.randn_like(xp)
        return restore_cond(xp + noise, cond_seq, cond_seq_mask)

    if config.num_self_cond_cfg_tokens > 0:
        if x_pred_prev is None:
            x_pred_prev = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)
        x_pred_prev = _perturb_x_pred(x_pred_prev, t_batch)
        z_input_cond = torch.cat([z, x_pred_prev], dim=-1)
        self_cond_scale_batch = torch.full((z.shape[0],), float(self_cond_cfg_scale),
                                           dtype=z.dtype, device=z.device)
        net_out_cond = model(z_input_cond, t_batch, deterministic=True,
                             self_cond_cfg_scale=self_cond_scale_batch)
        v_cond, x_cond = net_out_to_v_x(net_out_cond, z, t_batch, t_eps)
        return _restore(v_cond, x_cond)

    # No self-conditioning
    if self_cond_prob == 0:
        net_out = model(z, t_batch, deterministic=True)
        v, x = net_out_to_v_x(net_out, z, t_batch, t_eps)
        return _restore(v, x)

    # Combined unconditional and conditional forward pass
    v_uncond = x_uncond = None
    if self_cond_cfg_scale != 1 or x_pred_prev is None:
        z_uncond = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)
        z_input_uncond = torch.cat([z, z_uncond], dim=-1)
        net_out_uncond = model(z_input_uncond, t_batch, deterministic=True)
        v_uncond, x_uncond = net_out_to_v_x(net_out_uncond, z, t_batch, t_eps)
        v_uncond, x_uncond = _restore(v_uncond, x_uncond)
        if self_cond_cfg_scale == 0.0 or x_pred_prev is None:
            return v_uncond, x_uncond

    x_pred_prev = _perturb_x_pred(x_pred_prev, t_batch)
    z_input_cond = torch.cat([z, x_pred_prev], dim=-1)
    net_out_cond = model(z_input_cond, t_batch, deterministic=True)
    v_cond, x_cond = net_out_to_v_x(net_out_cond, z, t_batch, t_eps)
    v_cond, x_cond = _restore(v_cond, x_cond)
    if self_cond_cfg_scale == 1:
        return v_cond, x_cond

    v_out = v_uncond + self_cond_cfg_scale * (v_cond - v_uncond)
    x_out = x_uncond + self_cond_cfg_scale * (x_cond - x_uncond)
    return _restore(v_out, x_out)


def _forward_sample(
    model, z, t_batch, x_pred_prev, config,
    cfg_scale, self_cond_cfg_scale, cond_seq, cond_seq_mask,
    sc_noise_scale=0.0,
):
    """Forward pass with optional self-conditioning and CFG."""
    v_cond, x_cond = _forward_sample_self_cond(
        model, z, t_batch, x_pred_prev, config,
        self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )
    if cfg_scale == 1.0:
        return v_cond, x_cond

    # Unconditional forward: zero out cond prefix, no self-cond state, no restore
    z_uncond = restore_cond(z, torch.zeros_like(z), cond_seq_mask)
    x_pred_prev_uncond = (
        None if x_pred_prev is None
        else restore_cond(x_pred_prev, torch.zeros_like(x_pred_prev), cond_seq_mask)
    )
    v_uncond, x_uncond = _forward_sample_self_cond(
        model, z_uncond, t_batch, x_pred_prev_uncond, config,
        self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=torch.zeros_like(cond_seq), cond_seq_mask=cond_seq_mask,
    )

    v_out = v_uncond + cfg_scale * (v_cond - v_uncond)
    x_out = x_uncond + cfg_scale * (x_cond - x_uncond)
    return restore_vx(v_out, x_out, cond_seq, cond_seq_mask)


def _ode_step(
    model, z, t, t_next, x_pred_prev,
    config, cfg_scale, self_cond_cfg_scale,
    cond_seq, cond_seq_mask,
):
    """Single ODE (Euler) step for sampling."""
    t_batch = torch.full((z.shape[0],), float(t), dtype=z.dtype, device=z.device)
    v_pred, x_pred = _forward_sample(
        model=model, z=z, t_batch=t_batch, x_pred_prev=x_pred_prev,
        config=config, cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )
    return z + (t_next - t) * v_pred, x_pred


def _heun_step(
    model, z, t, t_next, x_pred_prev,
    config, cfg_scale, self_cond_cfg_scale,
    cond_seq, cond_seq_mask,
):
    """Single Heun (2nd-order Runge-Kutta / improved Euler) step for sampling.

    Heun's method for dz/dt = v(z, t):
      1. predictor: v1 = v(z, t),        z_pred = z + h * v1
      2. corrector: v2 = v(z_pred, t+h), z_next = z + h/2 * (v1 + v2)

    Per-step error is O(h^3) vs Euler's O(h^2), so fewer steps can achieve
    the same global accuracy. Each step costs 2 NFEs (network evaluations).
    """
    h = float(t_next - t)

    # Stage 1: evaluate at current position (same as one Euler step).
    t_batch = torch.full((z.shape[0],), float(t), dtype=z.dtype, device=z.device)
    v1, x_pred_stage1 = _forward_sample(
        model=model, z=z, t_batch=t_batch, x_pred_prev=x_pred_prev,
        config=config, cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )
    z_pred = restore_cond(z + h * v1, cond_seq, cond_seq_mask)
    x_pred_stage1 = restore_cond(x_pred_stage1, cond_seq, cond_seq_mask)

    # Stage 2: evaluate at the predicted next position.
    t_next_batch = torch.full((z.shape[0],), float(t_next), dtype=z.dtype, device=z.device)
    v2, x_pred = _forward_sample(
        model=model, z=z_pred, t_batch=t_next_batch, x_pred_prev=x_pred_stage1,
        config=config, cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )

    # Trapezoidal corrector: average the two velocity estimates.
    return restore_cond(z + h / 2.0 * (v1 + v2), cond_seq, cond_seq_mask), x_pred


def _heun_step_with_err(
    model, z, t, t_next, x_pred_prev,
    config, cfg_scale, self_cond_cfg_scale,
    cond_seq, cond_seq_mask,
    sc_noise_scale=0.0,
):
    """Heun step + free error estimate from Euler-vs-Heun discrepancy."""
    h = float(t_next - t)

    t_batch = torch.full((z.shape[0],), float(t), dtype=z.dtype, device=z.device)
    v1, x_pred_stage1 = _forward_sample(
        model=model, z=z, t_batch=t_batch, x_pred_prev=x_pred_prev,
        config=config, cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )
    z_pred = restore_cond(z + h * v1, cond_seq, cond_seq_mask)
    x_pred_stage1 = restore_cond(x_pred_stage1, cond_seq, cond_seq_mask)

    t_next_batch = torch.full((z.shape[0],), float(t_next), dtype=z.dtype, device=z.device)
    v2, x_pred = _forward_sample(
        model=model, z=z_pred, t_batch=t_next_batch, x_pred_prev=x_pred_stage1,
        config=config, cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )
    z_next = restore_cond(z + h / 2.0 * (v1 + v2), cond_seq, cond_seq_mask)

    # Local error: |z_heun - z_euler| = |h/2 * (v2 - v1)|
    diff = (v2 - v1).reshape(z.shape[0], -1)
    err_per_sample = diff.norm(dim=1) / (z.reshape(z.shape[0], -1).norm(dim=1) + 1e-8)
    err = (h / 2.0) * err_per_sample.max().item()

    return z_next, x_pred, err


def _heun_adaptive_step(
    model, z, t, t_next, x_pred_prev,
    config, cfg_scale, self_cond_cfg_scale,
    cond_seq, cond_seq_mask, tol, max_depth=4,
    sc_noise_scale=0.0,
):
    """Adaptive Heun step — subdivides when local error > tol * h.

    Error estimate is free (Euler vs Heun difference from the same 2 NFEs).
    Subdivision doubles NFE for that interval, but only triggers near t→1
    where the dynamics are stiff.  max_depth caps recursion depth.
    """
    h = float(t_next - t)
    if h < 1e-6 or max_depth <= 0:
        z_next, x_pred, _ = _heun_step_with_err(
            model, z, t, t_next, x_pred_prev,
            config, cfg_scale, self_cond_cfg_scale, cond_seq, cond_seq_mask,
        )
        return z_next, x_pred

    z_next, x_pred, err = _heun_step_with_err(
        model, z, t, t_next, x_pred_prev,
        config, cfg_scale, self_cond_cfg_scale, cond_seq, cond_seq_mask,
    )

    if err <= tol * h:
        return z_next, x_pred

    # Subdivide
    t_mid = (float(t) + float(t_next)) / 2.0
    z_mid, x_mid = _heun_adaptive_step(
        model, z, t, t_mid, x_pred_prev,
        config, cfg_scale, self_cond_cfg_scale, cond_seq, cond_seq_mask,
        tol, max_depth - 1,
    )
    return _heun_adaptive_step(
        model, z_mid, t_mid, t_next, x_mid,
        config, cfg_scale, self_cond_cfg_scale, cond_seq, cond_seq_mask,
        tol, max_depth - 1,
    )


def _exp_step(
    model, z, t, t_next, x_pred_prev,
    config, cfg_scale, self_cond_cfg_scale,
    cond_seq, cond_seq_mask,
):
    """Exponential integrator step — analytically solves the linear part.

    The Flow Matching ODE  dz/dt = (x0 - z)/(1-t)  is semi-linear.
    Treating x0_pred as locally constant gives the exact solution:
        z(t_next) = x0 + (z - x0) * (1-t_next) / (1-t)

    At t_next=1 this gives z=x0 exactly, handling the singularity naturally.
    Cost: 1 NFE per step (same as Euler).
    """
    t_batch = torch.full((z.shape[0],), float(t), dtype=z.dtype, device=z.device)
    _, x0 = _forward_sample(
        model=model, z=z, t_batch=t_batch, x_pred_prev=x_pred_prev,
        config=config, cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )
    # Analytical solution of dz/dt = (x0 - z)/(1-t) with constant x0.
    ratio = (1.0 - float(t_next)) / (1.0 - float(t))
    z_next = x0 + (z - x0) * ratio
    return restore_cond(z_next, cond_seq, cond_seq_mask), x0


def _exp_heun_step(
    model, z, t, t_next, x_pred_prev,
    config, cfg_scale, self_cond_cfg_scale,
    cond_seq, cond_seq_mask,
):
    """Exponential Heun (2nd-order) — combines exponential integration with
    a midpoint corrector.

      1. x0_1 = model(z, t)                                          (1 NFE)
      2. z_mid  = x0_1 + (z - x0_1) * (1-t_mid)/(1-t)    (exp half-step)
      3. x0_2 = model(z_mid, t_mid)                                  (2 NFE)
      4. z_next = x0_2 + (z - x0_2) * (1-t_next)/(1-t)   (exp full-step)

    Cost: 2 NFEs per step. Analogous to DPM-Solver-2 for diffusion models.
    """
    t_mid = (float(t) + float(t_next)) / 2.0

    # Stage 1: exponential half-step with x0 at current t.
    t_batch = torch.full((z.shape[0],), float(t), dtype=z.dtype, device=z.device)
    _, x0_1 = _forward_sample(
        model=model, z=z, t_batch=t_batch, x_pred_prev=x_pred_prev,
        config=config, cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )
    ratio_mid = (1.0 - t_mid) / (1.0 - float(t))
    z_mid = x0_1 + (z - x0_1) * ratio_mid
    z_mid = restore_cond(z_mid, cond_seq, cond_seq_mask)

    # Stage 2: exponential full-step with x0 at midpoint.
    t_mid_batch = torch.full((z.shape[0],), t_mid, dtype=z.dtype, device=z.device)
    _, x0 = _forward_sample(
        model=model, z=z_mid, t_batch=t_mid_batch, x_pred_prev=x0_1,
        config=config, cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )
    ratio = (1.0 - float(t_next)) / (1.0 - float(t))
    z_next = x0 + (z - x0) * ratio
    return restore_cond(z_next, cond_seq, cond_seq_mask), x0


def _pc_step(
    model, z, t, t_next, x_pred_prev,
    config, cfg_scale, self_cond_cfg_scale,
    cond_seq, cond_seq_mask, gamma, generator,
):
    """Predictor-Corrector step (2 NFEs).

    1. Predictor: exponential integrator  t → t_next  (deterministic, 1 NFE)
    2. Corrector:  add noise at t_next, then denoise back via exp (1 NFE)

    The corrector acts as a Langevin MCMC step that pushes the sample toward
    the local data manifold, compensating for integration error.  gamma controls
    the noise level; values around 0.3–0.8 work well (much smaller than pure SDE).
    """
    h = float(t_next - t)

    # --- Predictor (exp, 1 NFE) ---
    t_batch = torch.full((z.shape[0],), float(t), dtype=z.dtype, device=z.device)
    _, x0_pred = _forward_sample(
        model=model, z=z, t_batch=t_batch, x_pred_prev=x_pred_prev,
        config=config, cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )
    ratio = (1.0 - float(t_next)) / (1.0 - float(t))
    z_pred = x0_pred + (z - x0_pred) * ratio
    z_pred = restore_cond(z_pred, cond_seq, cond_seq_mask)

    # --- Corrector (noise injection + exp denoise, 1 NFE) ---
    alpha = max(0.0, min(1.0, 1.0 - gamma * h))
    t_back = alpha * float(t_next)

    if z.is_cuda:
        eps = torch.randn(z.shape, dtype=z.dtype, device=z.device) * config.denoiser_noise_scale
    else:
        eps = torch.randn(z.shape, generator=generator, dtype=z.dtype) * config.denoiser_noise_scale

    z_noise = restore_cond(alpha * z_pred + (1.0 - alpha) * eps, cond_seq, cond_seq_mask)
    t_back_batch = torch.full((z.shape[0],), t_back, dtype=z.dtype, device=z.device)
    _, x0 = _forward_sample(
        model=model, z=z_noise, t_batch=t_back_batch, x_pred_prev=x0_pred,
        config=config, cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )
    corr_ratio = (1.0 - float(t_next)) / (1.0 - float(t_back))
    z_corr = x0 + (z_noise - x0) * corr_ratio
    return restore_cond(z_corr, cond_seq, cond_seq_mask), x0


def _sde_step(
    model, z, t, t_next, x_pred_prev,
    config, cfg_scale, self_cond_cfg_scale,
    cond_seq, cond_seq_mask, gamma, generator,
    eps=None,
):
    """Per-step SDE-style sampler with hybrid (t-and-step) noise scaling.

    t_back = t * (1 - gamma * h), where h = t_next - t. alpha = 1 - gamma*h is the
    signal-preservation fraction, constant in t. gamma=0 degenerates to a plain ODE step.
    Uniform-N-step equivalence with old multiplicative gamma_old: gamma_hybrid = gamma_old * N.

    Args:
        eps: optional pre-generated noise tensor (shape = z.shape, already scaled).
             When provided, rdandn is skipped — used for reproducible controlled experiments.
    """
    h = float(t_next - t)
    alpha = max(0.0, min(1.0, 1.0 - gamma * h))
    t_back = alpha * float(t)
    if eps is not None:
        pass  # use pre-generated noise (already scaled)
    elif z.is_cuda:
        eps = torch.randn(z.shape, dtype=z.dtype, device=z.device) * config.denoiser_noise_scale
    else:
        eps = torch.randn(z.shape, generator=generator, dtype=z.dtype) * config.denoiser_noise_scale
    z_back = restore_cond(alpha * z + (1.0 - alpha) * eps, cond_seq, cond_seq_mask)
    t_batch = torch.full((z.shape[0],), t_back, dtype=z.dtype, device=z.device)
    v_pred, x_pred = _forward_sample(
        model=model, z=z_back, t_batch=t_batch, x_pred_prev=x_pred_prev,
        config=config, cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )
    return z_back + (t_next - t_back) * v_pred, x_pred


def _sde_ml_step(
    model, z, t, t_next, x_pred_prev,
    config, cfg_scale, self_cond_cfg_scale,
    cond_seq, cond_seq_mask, gamma, generator,
    num_langevin=2,
):
    """Multi-Langevin SDE step — k rounds of noise-inject + denoise per step.

    Each round uses independent random noise, so the stochastic correction
    accumulates more strongly than simply increasing gamma.  The final
    integrator uses exp (analytical) instead of Euler.
    Cost: num_langevin NFEs per step.
    """
    h = float(t_next - t)
    alpha = max(0.0, min(1.0, 1.0 - gamma * h))
    t_back = alpha * float(t)
    noise_scale = config.denoiser_noise_scale

    z_current = z
    x_current = x_pred_prev

    for _ in range(num_langevin):
        if z.is_cuda:
            eps = torch.randn(z.shape, dtype=z.dtype, device=z.device) * noise_scale
        else:
            eps = torch.randn(z.shape, generator=generator, dtype=z.dtype) * noise_scale
        z_noise = restore_cond(
            alpha * z_current + (1.0 - alpha) * eps,
            cond_seq, cond_seq_mask,
        )
        t_back_batch = torch.full((z.shape[0],), t_back, dtype=z.dtype, device=z.device)
        _, x_current = _forward_sample(
            model=model, z=z_noise, t_batch=t_back_batch, x_pred_prev=x_current,
            config=config, cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
            cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
        )
        ratio = (1.0 - float(t_next)) / (1.0 - float(t_back))
        z_current = x_current + (z_noise - x_current) * ratio
        z_current = restore_cond(z_current, cond_seq, cond_seq_mask)

    return z_current, x_current
