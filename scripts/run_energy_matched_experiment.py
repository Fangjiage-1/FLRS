#!/usr/bin/env python
"""Energy-matched rollback-allocation experiment for FLRS on ELF-B.

Compares four rollback schedules matched to the same direct-injection budget B2,
plus two fixed-gamma baselines.  All configurations share identical:
  - model checkpoint
  - time grid (per seed)
  - initial latent noise z0
  - per-step rollback noise epsilon_k

Usage:
    # Step 1 — dry-run (B2 verification only, no generation):
    python scripts/run_energy_matched_experiment.py \
        --config src/configs/training_configs/train_owt_ELF-B.yml \
        --checkpoint embedded-language-flows/ELF-B-owt-torch \
        --dry_run

    # Step 2 — full experiment:
    python scripts/run_energy_matched_experiment.py \
        --config src/configs/training_configs/train_owt_ELF-B.yml \
        --checkpoint embedded-language-flows/ELF-B-owt-torch \
        --nfe 64 --seeds 41,42,43,44,45 --num_samples 200 \
        --output_dir outputs/energy_matched_experiment
"""

import argparse
import copy
import json
import os
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from transformers import AutoTokenizer

from configs.config import Config, SamplingConfig, load_config_from_yaml
from modules.t5_encoder import get_encoder
from modules.model import ELF_models
from utils.train_utils import TrainState, get_optimizer, unwrap_model
from utils.checkpoint_utils import load_checkpoint
from utils.data_utils import get_pad_token_id
from utils.sampling_utils import (
    get_sampling_steps, restore_cond, _forward_sample,
    _sde_step, _exp_step, _ode_step,
)
from utils.generation_utils import _dlm_decode_batch, mask_after_eos
from utils.schedule_utils import (
    compute_gamma, _compute_per_step_stats,
    bisection_search, verify_B2_match,
    compute_empirical_energy, print_schedule_report,
)


# ============================================
# Repeated-span coverage metric
# ============================================

def repeated_span_coverage(texts: List[str], n: int = 5, min_repeats: int = 3) -> float:
    """Fraction of valid windows instantiating qualifying repeated n-grams.

    Args:
        texts: list of generated text strings
        n: n-gram size (default 5)
        min_repeats: minimum occurrences for an n-gram to be considered "repeated"

    Returns:
        float in [0, 1] — overall coverage ratio
    """
    total_covered = 0
    total_windows = 0
    for text in texts:
        tokens = text.strip().split()
        if len(tokens) < n:
            continue
        num_windows = len(tokens) - n + 1
        total_windows += num_windows
        ngram_counts = Counter()
        for i in range(num_windows):
            ngram_counts[tuple(tokens[i:i + n])] += 1
        repeated_ngrams = {ng for ng, cnt in ngram_counts.items() if cnt >= min_repeats}
        for i in range(num_windows):
            if tuple(tokens[i:i + n]) in repeated_ngrams:
                total_covered += 1
    return total_covered / total_windows if total_windows > 0 else 0.0


def repetition_incidence(texts: List[str], n: int = 5, min_repeats: int = 3) -> float:
    """Fraction of samples containing a qualifying repeated n-gram."""
    if not texts:
        return 0.0
    affected = 0
    for text in texts:
        tokens = text.strip().split()
        if len(tokens) < n:
            continue
        ngram_counts = Counter()
        for i in range(len(tokens) - n + 1):
            ngram_counts[tuple(tokens[i:i + n])] += 1
        if any(cnt >= min_repeats for cnt in ngram_counts.values()):
            affected += 1
    return affected / len(texts)


def unigram_entropy_from_texts(texts: List[str]) -> float:
    """Average unigram entropy across texts."""
    entropies = []
    for text in texts:
        tokens = text.strip().split()
        if not tokens:
            continue
        counter = Counter(tokens)
        total = len(tokens)
        probs = np.array([c / total for c in counter.values()])
        ent = float(-np.sum(probs * np.log(probs + 1e-10)))
        entropies.append(ent)
    return float(np.mean(entropies)) if entropies else 0.0


def avg_generation_length(texts: List[str]) -> float:
    lengths = [len(t.strip().split()) for t in texts]
    return float(np.mean(lengths)) if lengths else 0.0


# ============================================
# Model loading
# ============================================

def load_elf_model(config_path: str, checkpoint_path: str, device: torch.device):
    """Load ELF model, tokenizer, encoder from config and checkpoint."""
    print(f"Loading config from {config_path}")
    config = load_config_from_yaml(config_path)
    print(f"  Model: {config.model}")
    print(f"  sigma (denoiser_noise_scale): {config.denoiser_noise_scale}")

    print(f"Loading tokenizer: {config.tokenizer_name or config.encoder_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)

    print(f"Loading encoder: {config.encoder_model_name}")
    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    print(f"Building {config.model}")
    vocab_size = tokenizer.vocab_size
    model = ELF_models[config.model](
        text_encoder_dim=encoder_config.d_model,
        max_length=config.max_length,
        attn_drop=config.attn_dropout,
        proj_drop=config.proj_dropout,
        num_time_tokens=config.num_time_tokens,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        vocab_size=vocab_size,
        num_model_mode_tokens=config.num_model_mode_tokens,
        bottleneck_dim=config.bottleneck_dim,
    ).to(device)

    optimizer = get_optimizer(model, config, lr=1e-4)
    g = torch.Generator(device="cpu").manual_seed(config.seed)
    state = TrainState(
        model=model, optimizer=optimizer, lr_scheduler=None,
        ema_params1=TrainState.init_ema(model), step=0, epoch=0,
        dropout_generator=g,
    )

    print(f"Loading checkpoint from {checkpoint_path}")
    state, step = load_checkpoint(checkpoint_path, state)
    state.model = state.model.to(device).eval()
    print(f"  Checkpoint step: {step}")

    return state, config, tokenizer, pad_token_id, encoder_config, encoder


# ============================================
# Shared-noise sampling loop
# ============================================

@torch.no_grad()
def generate_with_pregenerated_noise(
    model: nn.Module,
    z: torch.Tensor,
    t_steps: torch.Tensor,
    pregenerated_eps: List[torch.Tensor],
    config: Config,
    cfg_scale: float,
    self_cond_cfg_scale: float,
    schedule_kind: str,
    schedule_amplitude: float,
    final_endpoint_type: str,
) -> Tuple[torch.Tensor, float, List[Dict]]:
    """Run ELF rollback sampling with pre-generated noise.

    Returns:
        (final_latent, energy_inc, per_step_info)

        energy_inc = sum_k mean(noise_delta_k^2), the batch-averaged
        injected-noise energy for this batch (scalar float, detached).
    """
    batch_size, max_length, d_model = z.shape
    cond_seq = torch.zeros((batch_size, max_length, d_model), dtype=z.dtype, device=z.device)
    cond_seq_mask = torch.zeros((batch_size, max_length), dtype=z.dtype, device=z.device)

    config._sc_noise_scale = 0.0

    z = restore_cond(z, cond_seq, cond_seq_mask)
    x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)

    n = t_steps.shape[0]
    use_bf16 = bool(getattr(config, "use_bf16", True)) and z.is_cuda

    per_step_info: List[Dict] = []
    energy_inc = 0.0

    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
        for i in range(n - 2):
            t = float(t_steps[i].item())
            t_next = float(t_steps[i + 1].item())
            h = t_next - t

            gamma_t = compute_gamma(t, schedule_kind, schedule_amplitude)
            raw_alpha = 1.0 - gamma_t * h
            alpha = max(0.0, min(1.0, raw_alpha))
            noise_coeff = 1.0 - alpha

            eps = pregenerated_eps[i]
            # Compute energy increment immediately, then discard the delta tensor
            delta_sq_mean = (noise_coeff ** 2) * eps.pow(2).mean()
            energy_inc += float(delta_sq_mean.item())

            per_step_info.append({
                "step": i, "t": t, "t_next": t_next, "h": h,
                "gamma": gamma_t, "alpha": alpha,
                "noise_coeff": noise_coeff, "clipped": raw_alpha < 0.0,
            })

            z, x_pred = _sde_step(
                model=model, z=z, t=t, t_next=t_next, x_pred_prev=x_pred,
                config=config, cfg_scale=cfg_scale,
                self_cond_cfg_scale=self_cond_cfg_scale,
                cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
                gamma=gamma_t, generator=None, eps=eps,
            )

        # Final step (t[-2] -> t[-1] = 1)
        t_final = float(t_steps[-2].item())
        t_next_final = float(t_steps[-1].item())
        if final_endpoint_type == "exp":
            z, x_pred = _exp_step(
                model=model, z=z, t=t_final, t_next=t_next_final,
                x_pred_prev=x_pred, config=config, cfg_scale=cfg_scale,
                self_cond_cfg_scale=self_cond_cfg_scale,
                cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
            )
        else:
            z, x_pred = _ode_step(
                model=model, z=z, t=t_final, t_next=t_next_final,
                x_pred_prev=x_pred, config=config, cfg_scale=cfg_scale,
                self_cond_cfg_scale=self_cond_cfg_scale,
                cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
            )

    return z, energy_inc, per_step_info


# ============================================
# Single-config run
# ============================================

@torch.no_grad()
def run_single_config(
    model: nn.Module,
    tokenizer,
    config: Config,
    t_steps: torch.Tensor,
    pregenerated_noise_batches: List[List[torch.Tensor]],
    pregenerated_z0_batches: List[torch.Tensor],
    schedule_kind: str,
    schedule_amplitude: float,
    final_endpoint_type: str,
    cfg_scale: float,
    self_cond_cfg_scale: float,
    device: torch.device,
    param_dtype: torch.dtype,
    eos_token_id: int,
    pad_token_id: int,
) -> Dict:
    """Run generation for a single schedule config across all batches.

    Returns dict with generated texts and metadata.
    """
    n_batches = len(pregenerated_z0_batches)
    all_texts: List[str] = []
    total_energy = 0.0
    total_sample_count = 0
    first_per_step_info: List[Dict] = []
    total_gen_time = 0.0
    total_decode_time = 0.0

    for batch_idx in range(n_batches):
        z = pregenerated_z0_batches[batch_idx].to(device=device, dtype=param_dtype)
        eps_list = [
            e.to(device=device, dtype=param_dtype)
            for e in pregenerated_noise_batches[batch_idx]
        ]

        gen_start = time.time()
        latent, energy_inc, per_step_info = generate_with_pregenerated_noise(
            model=model, z=z, t_steps=t_steps,
            pregenerated_eps=eps_list,
            config=config, cfg_scale=cfg_scale,
            self_cond_cfg_scale=self_cond_cfg_scale,
            schedule_kind=schedule_kind,
            schedule_amplitude=schedule_amplitude,
            final_endpoint_type=final_endpoint_type,
        )
        total_gen_time += time.time() - gen_start

        total_energy += energy_inc * z.shape[0]  # de-mean: multiply by batch size
        total_sample_count += z.shape[0]

        if batch_idx == 0:
            first_per_step_info = per_step_info

        # Free GPU noise tensors immediately
        del eps_list, z
        if device.type == "cuda":
            torch.cuda.empty_cache()

        dec_start = time.time()
        t_final_val = t_steps[-1].item()
        predicted_ids = _dlm_decode_batch(
            z=latent, model=model, t_final_val=t_final_val,
            config=config, self_cond_cfg_scale=self_cond_cfg_scale,
            decode_temperature=0.0, decode_top_k=0,
            repetition_penalty=1.0, latent_noise_scale=0.0,
        )
        total_decode_time += time.time() - dec_start

        predicted_ids = mask_after_eos(predicted_ids, eos_token_id=eos_token_id, pad_token_id=pad_token_id)

        for i in range(predicted_ids.shape[0]):
            text = tokenizer.decode(predicted_ids[i].detach().cpu().numpy(), skip_special_tokens=True)
            all_texts.append(text)

        del latent, predicted_ids

    sigma = config.denoiser_noise_scale
    stats = _compute_per_step_stats(
        t_steps, schedule_kind, schedule_amplitude, sigma,
        final_exp=(final_endpoint_type == "exp"),
    )
    B2 = stats["B2"]
    E_theory = stats["E_theory"]
    E_empirical = total_energy / max(total_sample_count, 1)

    rel_energy_err = abs(E_empirical - E_theory) / E_theory if E_theory > 0 else 0.0

    num_clipped = sum(1 for s in first_per_step_info if s["clipped"])
    max_nc = max((s["noise_coeff"] for s in first_per_step_info), default=0.0)
    sum_nc = sum(s["noise_coeff"] for s in first_per_step_info)

    return {
        "texts": all_texts,
        "num_texts": len(all_texts),
        "B2": B2,
        "E_theory": E_theory,
        "E_empirical": E_empirical,
        "relative_energy_error": rel_energy_err,
        "max_noise_coefficient": max_nc,
        "sum_noise_coefficient": sum_nc,
        "num_clipped_steps": num_clipped,
        "per_step_info": first_per_step_info,
        "gen_time": total_gen_time,
        "decode_time": total_decode_time,
    }


# ============================================
# PPL evaluation (offline — after ELF model freed)
# ============================================

def compute_ppl_metrics(texts: List[str], ppl_model_name: str = "gpt2-large",
                        max_length: int = 1024, batch_size: int = 16) -> Dict:
    """Compute generative PPL + unigram entropy using GPT-2 Large."""
    from utils.metrics_utils import Metrics

    nonempty = [s for s in texts if isinstance(s, str) and s.strip()]
    skipped = len(texts) - len(nonempty)
    if skipped > 0:
        print(f"  PPL eval: skipped {skipped} empty samples")
    if not nonempty:
        return {"ppl": float("nan"), "mean_entropy": float("nan")}

    metrics = Metrics(
        gen_ppl_eval_model_name_or_path=ppl_model_name,
        eval_ppl_batch_size=batch_size,
        eval_context_size=max_length,
    )
    results = metrics.record_generative_perplexity(
        text_samples=nonempty, max_length=max_length, retokenize=True,
    )
    return {"ppl": results["ppl"], "mean_entropy": results["mean_entropy"]}


# ============================================
# Main experiment
# ============================================

CONFIGS = [
    # (label, schedule_kind, amplitude_initial, final_endpoint_type, is_energy_matched)
    ("flrs_linear",        "early_linear",    2.0,  "exp", True),
    ("matched_fixed",      "fixed",           None,  "exp", True),
    ("late_linear",        "late_linear",     None,  "exp", True),
    ("flrs_quadratic",     "early_quadratic", None,  "exp", True),
    ("fixed_reference",    "fixed",           1.0,   "ode", False),
    ("fixed_exp_endpoint", "fixed",           1.0,   "exp", False),
]


def parse_args():
    p = argparse.ArgumentParser(description="FLRS Energy-Matched Rollback Allocation")
    p.add_argument("--config", type=str, required=True,
                   help="Path to training config YAML")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path or HF repo id for ELF checkpoint")
    p.add_argument("--nfe", type=int, default=64)
    p.add_argument("--seeds", type=str, default="41,42,43,44,45")
    p.add_argument("--num_samples", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--cfg_scale", type=float, default=1.0)
    p.add_argument("--self_cond_cfg_scale", type=float, default=3.0)
    p.add_argument("--output_dir", type=str, default="outputs/energy_matched_experiment")
    p.add_argument("--dry_run", action="store_true",
                   help="Only compute B_target and bisection amplitudes, no generation")
    p.add_argument("--device", type=str, default="cuda",
                   help="Device to use (cuda or cpu)")
    p.add_argument("--ppl_model", type=str, default="gpt2-large")
    p.add_argument("--ppl_batch_size", type=int, default=16)
    p.add_argument("--skip_ppl", action="store_true",
                   help="Skip PPL evaluation (faster; run eval_ppl.py separately)")
    p.add_argument("--save_texts", action="store_true", default=True,
                   help="Save generated texts as JSONL files")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device) if isinstance(args.device, str) else args.device
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA not available, falling back to CPU")

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    n_steps = args.nfe
    sigma = None  # will be set after config load

    print("=" * 70)
    print("FLRS Energy-Matched Rollback Allocation")
    print(f"  NFE: {n_steps}")
    print(f"  Seeds: {seeds}")
    print(f"  Samples per run: {args.num_samples}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Output: {args.output_dir}")
    print("=" * 70)

    # ---- Load model once ----
    state, config, tokenizer, pad_token_id, encoder_config, encoder = load_elf_model(
        args.config, args.checkpoint, device,
    )
    sigma = config.denoiser_noise_scale
    model = _build_eval_model_inline(state)
    param_dtype = next(model.parameters()).dtype
    d_model = model.text_encoder_dim
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1

    # Free original state (model copy + optimizer states) and encoder from GPU
    del state, encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("Freed original state and encoder from GPU memory.")

    # ---- Generate a canonical time grid for B2 computation ----
    # Use a fixed seed for the canonical time grid so amplitudes are deterministic.
    canonical_gen = torch.Generator(device="cpu").manual_seed(9999)
    # Temporarily override global seed for deterministic grid
    torch.manual_seed(9999)
    canonical_t_steps = get_sampling_steps(
        n_steps=n_steps, time_schedule="logit_normal",
        P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
        device=torch.device("cpu"), dtype=torch.float64,
    )
    print(f"\nCanonical time grid: {len(canonical_t_steps)} points "
          f"(min={canonical_t_steps[0]:.6f}, max={canonical_t_steps[-1]:.6f})")

    # ---- Compute B_target from early_linear A=2.0 ----
    print("\n" + "=" * 70)
    print("Computing B_target from FLRS-Linear (A=2.0)")
    print("=" * 70)
    c1_stats = print_schedule_report(
        canonical_t_steps, "early_linear", 2.0, sigma, final_exp=True,
    )
    B_target = c1_stats["B2"]
    print(f"\nB_target = {B_target:.12f}")

    # ---- Bisection search for matched amplitudes ----
    print("\n" + "=" * 70)
    print("Bisection search for energy-matched amplitudes")
    print("=" * 70)

    A_fixed, B_fixed, it_fixed = bisection_search(
        canonical_t_steps, "fixed", B_target, sigma, final_exp=True,
    )
    print(f"  fixed:           A={A_fixed:.8f}, B2={B_fixed:.12f}, iters={it_fixed}")

    A_late, B_late, it_late = bisection_search(
        canonical_t_steps, "late_linear", B_target, sigma, final_exp=True,
    )
    print(f"  late_linear:     A={A_late:.8f}, B2={B_late:.12f}, iters={it_late}")

    A_quad, B_quad, it_quad = bisection_search(
        canonical_t_steps, "early_quadratic", B_target, sigma, final_exp=True,
    )
    print(f"  early_quadratic: A={A_quad:.8f}, B2={B_quad:.12f}, iters={it_quad}")

    # ---- Verify B2 match ----
    print("\n" + "=" * 70)
    print("Verifying B2 match (tol=1e-6)")
    print("=" * 70)
    all_ok = verify_B2_match(
        canonical_t_steps, sigma, {
            "FLRS-Linear":    ("early_linear", 2.0),
            "Matched Fixed":  ("fixed", A_fixed),
            "Late-Linear":    ("late_linear", A_late),
            "FLRS-Quadratic": ("early_quadratic", A_quad),
        }, final_exp=True, tol=1e-6,
    )
    if not all_ok:
        print("ERROR: B2 match verification failed!")
        sys.exit(1)
    print("All energy-matched schedules pass B2 verification.")

    if args.dry_run:
        print("\nDry run complete. Amplitudes:")
        print(f"  A_fixed = {A_fixed:.8f}")
        print(f"  A_late  = {A_late:.8f}")
        print(f"  A_quad  = {A_quad:.8f}")
        print(f"  B_target = {B_target:.12f}")
        return

    # ---- Run experiment per seed ----
    all_results: List[Dict] = []

    for seed in seeds:
        print(f"\n{'#' * 70}")
        print(f"# Seed {seed}")
        print(f"{'#' * 70}")

        # Set seed and create generator
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        seed_gen = torch.Generator(device="cpu").manual_seed(seed)

        # ---- Generate per-seed time grid ----
        per_seed_t_steps = get_sampling_steps(
            n_steps=n_steps, time_schedule="logit_normal",
            P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
            device=torch.device("cpu"), dtype=torch.float64,
        )
        n_intervals = len(per_seed_t_steps) - 1
        n_noise_steps = n_intervals - 1  # last is exp/ode

        # ---- Compute per-seed B_target and matched amplitudes ----
        c1_stats_seed = _compute_per_step_stats(
            per_seed_t_steps, "early_linear", 2.0, sigma, final_exp=True,
        )
        B_target_seed = c1_stats_seed["B2"]
        print(f"  B_target (seed {seed}) = {B_target_seed:.8f}")

        A_fixed_seed, B_fixed_seed, _ = bisection_search(
            per_seed_t_steps, "fixed", B_target_seed, sigma, final_exp=True,
        )
        A_late_seed, B_late_seed, _ = bisection_search(
            per_seed_t_steps, "late_linear", B_target_seed, sigma, final_exp=True,
        )
        A_quad_seed, B_quad_seed, _ = bisection_search(
            per_seed_t_steps, "early_quadratic", B_target_seed, sigma, final_exp=True,
        )
        print(f"  A_fixed={A_fixed_seed:.6f}  A_late={A_late_seed:.6f}  A_quad={A_quad_seed:.6f}")

        # Verify per-seed B2 match
        seed_ok = verify_B2_match(
            per_seed_t_steps, sigma, {
                "FLRS-Linear": ("early_linear", 2.0),
                "Matched Fixed": ("fixed", A_fixed_seed),
                "Late-Linear": ("late_linear", A_late_seed),
                "FLRS-Quadratic": ("early_quadratic", A_quad_seed),
            }, final_exp=True, tol=1e-6,
        )
        if not seed_ok:
            print(f"  WARNING: B2 match failed for seed {seed}!")

        amplitude_map = {
            "flrs_linear": 2.0,
            "matched_fixed": A_fixed_seed,
            "late_linear": A_late_seed,
            "flrs_quadratic": A_quad_seed,
            "fixed_reference": 1.0,
            "fixed_exp_endpoint": 1.0,
        }

        # ---- Batch-by-batch: pre-generate noise once, run all configs, then free ----
        num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
        cfg_labels = [c[0] for c in CONFIGS]

        # Per-config accumulators
        acc_texts: Dict[str, List[str]] = {lbl: [] for lbl in cfg_labels}
        acc_energy: Dict[str, float] = {lbl: 0.0 for lbl in cfg_labels}
        acc_samples: Dict[str, int] = {lbl: 0 for lbl in cfg_labels}
        acc_gen_time: Dict[str, float] = {lbl: 0.0 for lbl in cfg_labels}
        acc_decode_time: Dict[str, float] = {lbl: 0.0 for lbl in cfg_labels}
        first_info: Dict[str, Optional[List]] = {lbl: None for lbl in cfg_labels}

        print(f"  Generating {args.num_samples} samples in {num_batches} batch(es)...")

        for batch_idx in range(num_batches):
            current_bs = min(args.batch_size, args.num_samples - batch_idx * args.batch_size)

            # ---- Pre-generate z0 + eps for THIS batch only ----
            z0_cpu = torch.randn(
                (current_bs, config.max_length, d_model),
                generator=seed_gen, dtype=param_dtype,
            ) * sigma
            eps_cpu_list = []
            for _ in range(n_noise_steps):
                eps_cpu_list.append(
                    torch.randn(
                        (current_bs, config.max_length, d_model),
                        generator=seed_gen, dtype=param_dtype,
                    ) * sigma
                )

            # ---- Run all 6 configs on this batch, reusing the same noise ----
            for cfg_label, schedule_kind, amp_initial, final_ep, is_matched in CONFIGS:
                amp = amplitude_map[cfg_label]

                # Clone noise to GPU (each config gets identical noise via clone)
                z = z0_cpu.clone().to(device=device, dtype=param_dtype)
                eps_list = [e.clone().to(device=device, dtype=param_dtype) for e in eps_cpu_list]

                gen_start = time.time()
                latent, energy_inc, per_step_info = generate_with_pregenerated_noise(
                    model=model, z=z, t_steps=per_seed_t_steps,
                    pregenerated_eps=eps_list,
                    config=config, cfg_scale=args.cfg_scale,
                    self_cond_cfg_scale=args.self_cond_cfg_scale,
                    schedule_kind=schedule_kind,
                    schedule_amplitude=amp,
                    final_endpoint_type=final_ep,
                )
                acc_gen_time[cfg_label] += time.time() - gen_start
                acc_energy[cfg_label] += energy_inc * current_bs
                acc_samples[cfg_label] += current_bs
                if first_info[cfg_label] is None:
                    first_info[cfg_label] = per_step_info

                del eps_list, z
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                # Decode
                dec_start = time.time()
                t_final_val = per_seed_t_steps[-1].item()
                predicted_ids = _dlm_decode_batch(
                    z=latent, model=model, t_final_val=t_final_val,
                    config=config, self_cond_cfg_scale=args.self_cond_cfg_scale,
                    decode_temperature=0.0, decode_top_k=0,
                    repetition_penalty=1.0, latent_noise_scale=0.0,
                )
                acc_decode_time[cfg_label] += time.time() - dec_start

                predicted_ids = mask_after_eos(predicted_ids, eos_token_id=eos_token_id, pad_token_id=pad_token_id)
                for i in range(predicted_ids.shape[0]):
                    text = tokenizer.decode(predicted_ids[i].detach().cpu().numpy(), skip_special_tokens=True)
                    acc_texts[cfg_label].append(text)

                del latent, predicted_ids

            # Free CPU noise for this batch
            del z0_cpu, eps_cpu_list

            if (batch_idx + 1) % max(1, num_batches // 5) == 0:
                print(f"    Batch {batch_idx + 1}/{num_batches} done")

        # ---- Build result dicts per config ----
        for cfg_label, schedule_kind, amp_initial, final_ep, is_matched in CONFIGS:
            amp = amplitude_map[cfg_label]
            texts = acc_texts[cfg_label]
            sigma_local = config.denoiser_noise_scale

            stats = _compute_per_step_stats(
                per_seed_t_steps, schedule_kind, amp, sigma_local,
                final_exp=(final_ep == "exp"),
            )
            B2 = stats["B2"]
            E_theory = stats["E_theory"]
            E_empirical = acc_energy[cfg_label] / max(acc_samples[cfg_label], 1)
            rel_energy_err = abs(E_empirical - E_theory) / E_theory if E_theory > 0 else 0.0

            pi = first_info[cfg_label] or []
            num_clipped = sum(1 for s in pi if s.get("clipped", False))
            max_nc = max((s.get("noise_coeff", 0.0) for s in pi), default=0.0)
            sum_nc = sum(s.get("noise_coeff", 0.0) for s in pi)

            rsc = repeated_span_coverage(texts)
            ri = repetition_incidence(texts)
            ue = unigram_entropy_from_texts(texts)
            avg_len = avg_generation_length(texts)
            wall_time = acc_gen_time[cfg_label] + acc_decode_time[cfg_label]

            print(f"  {cfg_label}: {len(texts)} texts, B2={B2:.8f}, "
                  f"gen={acc_gen_time[cfg_label]:.1f}s, dec={acc_decode_time[cfg_label]:.1f}s, "
                  f"rel_err={rel_energy_err:.2e}, clipped={num_clipped}")

            # Save texts
            if args.save_texts:
                out_dir = os.path.join(args.output_dir, f"seed_{seed}", cfg_label)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, "generated.jsonl")
                with open(out_path, "w", encoding="utf-8") as f:
                    for tid, text in enumerate(texts):
                        f.write(json.dumps({"id": tid, "generated": text}, ensure_ascii=False) + "\n")

            all_results.append({
                "texts": texts,
                "num_texts": len(texts),
                "B2": B2,
                "E_theory": E_theory,
                "E_empirical": E_empirical,
                "relative_energy_error": rel_energy_err,
                "max_noise_coefficient": max_nc,
                "sum_noise_coefficient": sum_nc,
                "num_clipped_steps": num_clipped,
                "per_step_info": pi,
                "gen_time": acc_gen_time[cfg_label],
                "decode_time": acc_decode_time[cfg_label],
                "seed": seed,
                "schedule_kind": schedule_kind,
                "schedule_amplitude": amp,
                "final_endpoint_type": final_ep,
                "config_label": cfg_label,
                "is_energy_matched": is_matched,
                "rsc": rsc,
                "ri": ri,
                "unigram_entropy": ue,
                "avg_length": avg_len,
                "wall_clock_time": wall_time,
            })

        # Clear per-seed accumulators
        del acc_texts, acc_energy, acc_samples, acc_gen_time, acc_decode_time, first_info

    # ---- Free ELF model before PPL eval ----
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- PPL evaluation (offline) ----
    if not args.skip_ppl:
        print("\n" + "=" * 70)
        print("PPL Evaluation (GPT-2 Large)")
        print("=" * 70)
        for r in all_results:
            label = f"seed_{r['seed']}/{r['config_label']}"
            print(f"  Computing PPL for {label} ({r['num_texts']} texts)...")
            ppl_results = compute_ppl_metrics(
                r["texts"], ppl_model_name=args.ppl_model,
                max_length=config.eval_ppl_max_length,
                batch_size=args.ppl_batch_size,
            )
            r["ppl"] = ppl_results["ppl"]
            r["mean_entropy"] = ppl_results["mean_entropy"]
            print(f"    PPL={r['ppl']:.4f}  Entropy={r['mean_entropy']:.4f}")

    # ---- Save per-run JSON ----
    def _to_native(v):
        """Convert numpy / torch scalars to Python native types for JSON."""
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, torch.Tensor):
            return float(v.item()) if v.numel() == 1 else v.tolist()
        return v

    os.makedirs(args.output_dir, exist_ok=True)
    results_for_json = []
    for r in all_results:
        entry = {
            k: _to_native(v) for k, v in r.items()
            if k not in ("texts", "per_step_info")
        }
        results_for_json.append(entry)

    results_path = os.path.join(args.output_dir, "all_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_for_json, f, indent=2, ensure_ascii=False)
    print(f"\nPer-run results saved to {results_path}")

    # ---- Summary CSV ----
    _save_summary_csv(all_results, args.output_dir)
    _print_summary_table(all_results, seeds, B_target)

    print("\n" + "=" * 70)
    print("Experiment complete!")
    print("=" * 70)


def _build_eval_model_inline(state):
    """Build eval-mode model copy with EMA params."""
    model = unwrap_model(state.model)
    eval_model = copy.deepcopy(model)
    if state.ema_params1:
        eval_model.load_state_dict(state.ema_params1)
    eval_model.eval()
    return eval_model


def _save_summary_csv(all_results: List[Dict], output_dir: str):
    """Save summary CSV with mean ± std across seeds."""
    import csv

    # Group by config label
    configs_order = [c[0] for c in CONFIGS]
    config_labels = list(dict.fromkeys(r["config_label"] for r in all_results))

    csv_path = os.path.join(output_dir, "summary.csv")
    fieldnames = [
        "config", "amplitude", "B2", "n_seeds",
        "ppl_mean", "ppl_std",
        "entropy_mean", "entropy_std",
        "rsc_mean", "rsc_std",
        "ri_mean", "ri_std",
        "unigram_entropy_mean", "unigram_entropy_std",
        "avg_length_mean", "avg_length_std",
        "E_theory_mean", "E_empirical_mean",
        "num_clipped_mean",
        "wall_clock_time_mean", "wall_clock_time_std",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for label in configs_order:
            group = [r for r in all_results if r["config_label"] == label]
            if not group:
                continue
            n = len(group)

            def stat(key):
                vals = [r.get(key, float("nan")) for r in group]
                valid = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
                if not valid:
                    return float("nan"), float("nan")
                mean = float(np.mean(valid))
                std = float(np.std(valid)) if len(valid) > 1 else 0.0
                return mean, std

            ppl_m, ppl_s = stat("ppl")
            ent_m, ent_s = stat("mean_entropy")
            rsc_m, rsc_s = stat("rsc")
            ri_m, ri_s = stat("ri")
            ue_m, ue_s = stat("unigram_entropy")
            al_m, al_s = stat("avg_length")
            et_m, _ = stat("E_theory")
            ee_m, _ = stat("E_empirical")
            nc_m, _ = stat("num_clipped_steps")
            wt_m, wt_s = stat("wall_clock_time")
            amp_val = group[0]["schedule_amplitude"]
            b2_val = group[0]["B2"]

            writer.writerow({
                "config": label, "amplitude": f"{amp_val:.6f}",
                "B2": f"{b2_val:.8f}", "n_seeds": n,
                "ppl_mean": f"{ppl_m:.4f}", "ppl_std": f"{ppl_s:.4f}",
                "entropy_mean": f"{ent_m:.4f}", "entropy_std": f"{ent_s:.4f}",
                "rsc_mean": f"{rsc_m:.6f}", "rsc_std": f"{rsc_s:.6f}",
                "ri_mean": f"{ri_m:.6f}", "ri_std": f"{ri_s:.6f}",
                "unigram_entropy_mean": f"{ue_m:.6f}", "unigram_entropy_std": f"{ue_s:.6f}",
                "avg_length_mean": f"{al_m:.1f}", "avg_length_std": f"{al_s:.1f}",
                "E_theory_mean": f"{et_m:.6f}", "E_empirical_mean": f"{ee_m:.6f}",
                "num_clipped_mean": f"{nc_m:.1f}",
                "wall_clock_time_mean": f"{wt_m:.1f}", "wall_clock_time_std": f"{wt_s:.1f}",
            })

    print(f"Summary CSV saved to {csv_path}")


def _print_summary_table(all_results, seeds, B_target):
    """Print the final summary table."""
    configs_order = [c[0] for c in CONFIGS]

    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    header = f"{'Schedule':<28s} {'Amp':>8s} {'B2':>10s} {'PPL':>12s} {'Entropy':>12s} {'RSC':>10s} {'RI':>10s} {'Clip':>6s}"
    print(header)
    print("-" * 90)

    for label in configs_order:
        group = [r for r in all_results if r["config_label"] == label]
        if not group:
            continue

        def stat(key):
            vals = [r.get(key, float("nan")) for r in group]
            valid = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
            if not valid:
                return float("nan"), float("nan")
            m = float(np.mean(valid))
            s = float(np.std(valid)) if len(valid) > 1 else 0.0
            return m, s

        ppl_m, ppl_s = stat("ppl")
        ent_m, ent_s = stat("mean_entropy")
        rsc_m, rsc_s = stat("rsc")
        ri_m, ri_s = stat("ri")
        nc_m, _ = stat("num_clipped_steps")
        amp = group[0]["schedule_amplitude"]
        b2 = stat("B2")[0]

        def fmt_val(m, s, prec=2):
            if np.isnan(m):
                return f"{'N/A':>12s}"
            return f"{m:>{prec+3}.{prec}f}±{s:<{prec+3}.{prec}f}"

        nc_str = f"{nc_m:>4.1f}" if not np.isnan(nc_m) else f"{'N/A':>4s}"
        print(f"{label:<28s} {amp:>8.4f} {b2:>10.6f} "
              f"{fmt_val(ppl_m, ppl_s, 2)} "
              f"{fmt_val(ent_m, ent_s, 2)} "
              f"{fmt_val(rsc_m, rsc_s, 4)} "
              f"{fmt_val(ri_m, ri_s, 4)} "
              f"{nc_str}")

    print("-" * 90)
    print(f"Canonical B_target = {B_target:.8f}")

    # Paired deltas — only if PPL is available
    c1_group = [r for r in all_results if r["config_label"] == "flrs_linear"]
    c2_group = [r for r in all_results if r["config_label"] == "matched_fixed"]
    has_ppl = all("ppl" in r and not np.isnan(r["ppl"]) for r in c1_group + c2_group)
    if has_ppl and c1_group and c2_group and len(c1_group) == len(c2_group):
        print("\n" + "-" * 70)
        print("Paired deltas: FLRS-Linear vs Matched Fixed")
        deltas_ppl = []
        deltas_rsc = []
        early_better = 0
        for r1, r2 in zip(c1_group, c2_group):
            dp = r1["ppl"] - r2["ppl"]
            dr = r1["rsc"] - r2["rsc"]
            deltas_ppl.append(dp)
            deltas_rsc.append(dr)
            if dp < 0:
                early_better += 1
        dp_mean = float(np.mean(deltas_ppl))
        dp_std = float(np.std(deltas_ppl))
        dr_mean = float(np.mean(deltas_rsc))
        dr_std = float(np.std(deltas_rsc))
        print("  Delta_PPL = FLRS-Linear - Matched Fixed")
        print(f"    mean = {dp_mean:.4f}  std = {dp_std:.4f}")
        print(f"    FLRS-Linear wins (lower PPL) in {early_better}/{len(deltas_ppl)} seeds")
        print("  Delta_RSC = FLRS-Linear - Matched Fixed")
        print(f"    mean = {dr_mean:.6f}  std = {dr_std:.6f}")
    else:
        print("\n(PPL not available — run without --skip_ppl for paired deltas)")


if __name__ == "__main__":
    main()
