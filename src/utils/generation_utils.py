from typing import Optional

import torch
import torch.nn as nn

from configs.config import Config, SamplingConfig
from utils.sampling_utils import restore_cond, _ode_step, _heun_step, _heun_adaptive_step, _exp_step, _exp_heun_step, _pc_step, _sde_ml_step, _sde_step


# ============================================
# Generation utilities
# ============================================

def mask_after_eos(predicted_ids: torch.Tensor, eos_token_id: int, pad_token_id: int) -> torch.Tensor:
    """Mask everything at/after first EOS token per sequence."""
    eos_mask = (predicted_ids == eos_token_id)
    keep_mask = (eos_mask.to(torch.int32).cumsum(dim=1) == 0)
    return torch.where(keep_mask, predicted_ids, torch.full_like(predicted_ids, pad_token_id))


def shift_left(x: torch.Tensor, shift_per_sample: torch.Tensor, pad_value=0, axis: int = 1) -> torch.Tensor:
    """Shift each sample left along the sequence axis; pad emptied positions."""
    if x.dim() < 2:
        raise ValueError("x must have at least batch and sequence dimensions")
    if axis < 0:
        axis = x.dim() + axis
    if axis == 0:
        raise ValueError("axis=0 is the batch axis and cannot be shifted")
    shift_per_sample = shift_per_sample.to(torch.long)
    if axis != 1:
        x = x.movedim(axis, 1)
    seq_len = x.shape[1]
    base_idx = torch.arange(seq_len, device=x.device)[None, :]
    gather_idx = shift_per_sample[:, None].to(x.device) + base_idx
    valid = gather_idx < seq_len
    gather_idx = gather_idx.clamp(0, seq_len - 1)
    if x.dim() == 2:
        shifted = torch.gather(x, 1, gather_idx)
        shifted = torch.where(valid, shifted, torch.full_like(shifted, pad_value))
    else:
        expand_shape = [-1, -1] + list(x.shape[2:])
        idx = gather_idx.view(*gather_idx.shape, *([1] * (x.dim() - 2))).expand(*expand_shape)
        valid_b = valid.view(*valid.shape, *([1] * (x.dim() - 2))).expand(*expand_shape)
        shifted = torch.gather(x, 1, idx)
        shifted = torch.where(valid_b, shifted, torch.full_like(shifted, pad_value))
    if axis != 1:
        shifted = shifted.movedim(1, axis)
    return shifted


# ============================================
# Single-batch sampling (PyTorch)
# ============================================

@torch.no_grad()
def _generate_samples_single_batch(
    model: nn.Module,
    generator: torch.Generator,
    z: torch.Tensor,
    t_steps: torch.Tensor,
    cond_seq: Optional[torch.Tensor],
    cond_seq_mask: Optional[torch.Tensor],
    config: Config,
    sampling_config: SamplingConfig,
    cfg_scale: float,
    self_cond_cfg_scale: float,
) -> torch.Tensor:
    """Generate samples for a single batch (PyTorch Euler / SDE rollout)."""
    method = sampling_config.sampling_method
    batch_size, max_length, d_model = z.shape
    if cond_seq is None:
        cond_seq = torch.zeros((batch_size, max_length, d_model), dtype=z.dtype, device=z.device)
        cond_seq_mask = torch.zeros((batch_size, max_length), dtype=z.dtype, device=z.device)

    config._sc_noise_scale = float(getattr(sampling_config, 'sc_noise_scale', 0.0))
    step_kwargs = dict(
        model=model, config=config,
        cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )

    z = restore_cond(z, cond_seq, cond_seq_mask)
    x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)

    n = t_steps.shape[0]
    sde_gamma = getattr(sampling_config, "sde_gamma", 0.0)

    use_bf16 = bool(getattr(config, "use_bf16", True)) and z.is_cuda
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
        for i in range(n - 2):
            t = t_steps[i].item()
            t_next = t_steps[i + 1].item()
            if method == "sde":
                z, x_pred = _sde_step(
                    z=z, t=t, t_next=t_next, x_pred_prev=x_pred,
                    gamma=sde_gamma, generator=generator, **step_kwargs,
                )
            elif method == "sde_anneal":
                gamma_start = sde_gamma
                gamma_end = float(getattr(sampling_config, 'sde_gamma_end', 0.0))
                anneal_power = float(getattr(sampling_config, 'sde_anneal_power', 1.0))
                # gamma(t) = gamma_end + (gamma_start - gamma_end) * (1-t)^p
                #   p=1: linear decay (fast early, slow near end)
                #   p=2: quadratic (slow early, fast near end — keeps diversity longer)
                #   t≈0 → gamma=gamma_start (high noise, diversity)
                #   t≈1 → gamma=gamma_end   (low noise, clean convergence)
                gamma_t = gamma_end + (gamma_start - gamma_end) * (1.0 - float(t)) ** anneal_power
                z, x_pred = _sde_step(
                    z=z, t=t, t_next=t_next, x_pred_prev=x_pred,
                    gamma=gamma_t, generator=generator, **step_kwargs,
                )
            elif method == "sde_ml":
                num_langevin = int(getattr(sampling_config, 'num_langevin', 2))
                z, x_pred = _sde_ml_step(
                    z=z, t=t, t_next=t_next, x_pred_prev=x_pred,
                    gamma=sde_gamma, generator=generator,
                    num_langevin=num_langevin, **step_kwargs,
                )
            elif method == "ode":
                z, x_pred = _ode_step(z=z, t=t, t_next=t_next, x_pred_prev=x_pred, **step_kwargs)
            elif method == "heun":
                z, x_pred = _heun_step(z=z, t=t, t_next=t_next, x_pred_prev=x_pred, **step_kwargs)
            elif method == "heun_adaptive":
                tol = float(getattr(sampling_config, 'heun_tol', 0.05))
                z, x_pred = _heun_adaptive_step(
                    z=z, t=t, t_next=t_next, x_pred_prev=x_pred, tol=tol, **step_kwargs,
                )
            elif method == "exp":
                z, x_pred = _exp_step(z=z, t=t, t_next=t_next, x_pred_prev=x_pred, **step_kwargs)
            elif method == "exp_heun":
                z, x_pred = _exp_heun_step(z=z, t=t, t_next=t_next, x_pred_prev=x_pred, **step_kwargs)
            elif method == "pc":
                z, x_pred = _pc_step(
                    z=z, t=t, t_next=t_next, x_pred_prev=x_pred,
                    gamma=sde_gamma, generator=generator, **step_kwargs,
                )
            elif method == "sde_exp":
                threshold = float(getattr(sampling_config, 'sde_exp_threshold', 0.8))
                if t < threshold:
                    z, x_pred = _sde_step(
                        z=z, t=t, t_next=t_next, x_pred_prev=x_pred,
                        gamma=sde_gamma, generator=generator, **step_kwargs,
                    )
                else:
                    z, x_pred = _exp_step(z=z, t=t, t_next=t_next, x_pred_prev=x_pred, **step_kwargs)
            else:
                raise ValueError(f"Invalid sampling method: {method}")

            # Post-step z-space noise injection (decays as t→1).
            z_noise_scale = float(getattr(sampling_config, 'z_noise_scale', 0.0))
            if z_noise_scale > 0:
                decay = (1.0 - float(t_next)) ** 2
                z = z + z_noise_scale * decay * torch.randn_like(z)
                z = restore_cond(z, cond_seq, cond_seq_mask)

        # Final step (t → 1): use exp for methods that benefit from it, ODE otherwise.
        t = t_steps[-2].item()
        t_next = t_steps[-1].item()
        if method in ("sde_exp", "pc", "sde_ml", "heun_adaptive", "sde_anneal"):
            z, x_pred = _exp_step(z=z, t=t, t_next=t_next, x_pred_prev=x_pred, **step_kwargs)
        else:
            z, x_pred = _ode_step(z=z, t=t, t_next=t_next, x_pred_prev=x_pred, **step_kwargs)
    return z


@torch.no_grad()
def _dlm_decode_batch(z: torch.Tensor, model: nn.Module, t_final_val,
                      config, self_cond_cfg_scale: float,
                      decode_temperature: float = 0.0,
                      decode_top_k: int = 0,
                      repetition_penalty: float = 1.0,
                      latent_noise_scale: float = 0.0) -> torch.Tensor:
    """Decode z -> tokens with the DLM decoder head.

    Args:
        decode_temperature: if > 0, use temperature-scaled multinomial
            sampling instead of argmax.  >1.0 increases diversity.
        decode_top_k: if > 0, restrict sampling to top-k tokens.
        repetition_penalty: if > 1.0, penalize tokens that appear multiple
            times in the argmax reference decode.
        latent_noise_scale: if > 0, add Gaussian noise to z before decoding.
            Inspired by LeWM's SIGReg (isotropic Gaussian prior on latents).
            Forces the decoder to handle a neighbourhood, boosting diversity.
            Typical values: 0.05–0.2 (relative to denoiser_noise_scale=2.0).
    """
    import math

    # Perturb latent with isotropic Gaussian noise (0 extra NFE).
    if latent_noise_scale > 0:
        z = z + latent_noise_scale * torch.randn_like(z)

    batch_size = z.shape[0]
    if isinstance(t_final_val, torch.Tensor) and t_final_val.dim() == 0:
        t_final = torch.full((batch_size,), t_final_val.item(), dtype=z.dtype, device=z.device)
    else:
        t_final = torch.full((batch_size,), float(t_final_val), dtype=z.dtype, device=z.device)
    sc_batch = (
        torch.full((batch_size,), float(self_cond_cfg_scale), dtype=z.dtype, device=z.device)
        if config.num_self_cond_cfg_tokens > 0 else None
    )
    z_input = torch.cat([z, torch.zeros_like(z)], dim=-1) if config.self_cond_prob > 0 else z
    use_bf16 = bool(getattr(config, "use_bf16", True)) and z.is_cuda
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
        _, decoder_logits = model(
            z_input, t_final, deterministic=True,
            self_cond_cfg_scale=sc_batch,
            decoder_step_active=True,
        )
    if decode_temperature > 0:
        logits = decoder_logits / decode_temperature

        if repetition_penalty != 1.0:
            log_penalty = math.log(repetition_penalty)
            # Reference tokens from argmax preview (one-shot, no extra NFE)
            ref_ids = logits.argmax(dim=-1)  # (B, L)
            B, L, V = logits.shape
            penalty = torch.zeros(B, V, dtype=logits.dtype, device=logits.device)
            for b in range(B):
                seen = {}
                for p in range(L):
                    tok = ref_ids[b, p].item()
                    if tok in seen:
                        seen[tok] += 1
                        logits[b, p, tok] -= log_penalty * seen[tok]
                    else:
                        seen[tok] = 1

        if decode_top_k > 0:
            top_k = min(decode_top_k, logits.size(-1))
            top_vals, _ = logits.topk(top_k, dim=-1)
            threshold = top_vals[..., -1:]
            probs = torch.softmax(
                torch.where(logits >= threshold, logits, torch.full_like(logits, float('-inf'))),
                dim=-1,
            )
        else:
            probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(batch_size, -1)
    return decoder_logits.argmax(dim=-1)


def _build_run_name(sampling_method, num_sampling_steps, cfg_scale, self_cond_cfg_scale,
                    time_schedule, sde_gamma, suffix, num_langevin=None, heun_tol=None,
                    decode_temperature=None, repetition_penalty=None,
                    latent_noise_scale=None, sc_noise_scale=None, z_noise_scale=None,
                    sde_gamma_end=None, sde_anneal_power=None):
    ts_str = f"-ts_{time_schedule}"
    sccfg_str = f"-sccfg{self_cond_cfg_scale}" if self_cond_cfg_scale != 1.0 else ""
    if sampling_method in ("sde", "pc", "sde_exp", "sde_ml", "sde_anneal"):
        sde_str = f"-gamma{sde_gamma}"
        if sampling_method == "sde_anneal" and sde_gamma_end is not None:
            sde_str += f"to{sde_gamma_end}"
            if sde_anneal_power is not None and sde_anneal_power != 1.0:
                sde_str += f"p{sde_anneal_power}"
    else:
        sde_str = ""
    ml_str = f"-m{num_langevin}" if num_langevin is not None and sampling_method == "sde_ml" else ""
    tol_str = f"-tol{heun_tol}" if heun_tol is not None and sampling_method == "heun_adaptive" else ""
    temp_str = f"-T{decode_temperature}" if decode_temperature is not None and decode_temperature > 0 else ""
    rp_str = f"-rp{repetition_penalty}" if repetition_penalty is not None and repetition_penalty != 1.0 else ""
    lns_str = f"-lns{latent_noise_scale}" if latent_noise_scale is not None and latent_noise_scale > 0 else ""
    scn_str = f"-scn{sc_noise_scale}" if sc_noise_scale is not None and sc_noise_scale > 0 else ""
    zn_str = f"-zn{z_noise_scale}" if z_noise_scale is not None and z_noise_scale > 0 else ""
    return f"{sampling_method}-steps{num_sampling_steps}-cfg{cfg_scale}{sccfg_str}{ts_str}{sde_str}{ml_str}{tol_str}{temp_str}{rp_str}{lns_str}{scn_str}{zn_str}-{suffix}"
