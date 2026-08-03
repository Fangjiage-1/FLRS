"""Schedule utilities for ELF SDE energy-matched experiments.

Provides gamma schedules, B2 (noise injection budget) computation, and
bisection search for amplitude matching.
"""

from typing import Dict, List, Tuple, Optional
import torch


# ============================================
# Gamma schedule functions
# ============================================

def compute_gamma(t: float, schedule_kind: str, amplitude: float) -> float:
    """Compute gamma at continuous time t in [0, 1].

    Args:
        t: continuous time value in [0, 1]
        schedule_kind: one of 'fixed', 'early_linear', 'late_linear', 'early_quadratic'
        amplitude: schedule amplitude A

    Returns:
        gamma value at time t (non-negative)
    """
    if schedule_kind == "fixed":
        return float(amplitude)
    elif schedule_kind == "early_linear":
        return float(amplitude) * (1.0 - float(t))
    elif schedule_kind == "late_linear":
        return float(amplitude) * float(t)
    elif schedule_kind == "early_quadratic":
        return float(amplitude) * (1.0 - float(t)) ** 2
    else:
        raise ValueError(f"Unknown schedule_kind: {schedule_kind}")


def _compute_per_step_stats(
    t_steps: torch.Tensor,
    schedule_kind: str,
    amplitude: float,
    sigma: float,
    final_exp: bool = True,
) -> Dict:
    """Compute per-step alpha_k, noise coefficients, and B2 on the given time grid.

    The last interval (k = n_intervals-1) ALWAYS has noise_coeff = 0 because
    it is handled by a deterministic integrator (either _exp_step or _ode_step).
    The final_exp flag only records which integrator is used; it does not affect B2.

    Args:
        t_steps: 1D tensor of t values (length n_steps+1), must be sorted ascending
        schedule_kind: gamma schedule type
        amplitude: schedule amplitude
        sigma: noise scale (config.denoiser_noise_scale)
        final_exp: records the endpoint type for metadata; does NOT affect B2

    Returns:
        dict with keys: B2, E_theory, per_step alpha, gamma, h, noise_coeff,
                        num_clipped, max_noise_coeff, sum_noise_coeff
    """
    t_arr = t_steps.detach().cpu().numpy()
    n_intervals = len(t_arr) - 1

    per_step = {
        "t_k": [],
        "h_k": [],
        "gamma_k": [],
        "alpha_k": [],
        "noise_coeff": [],  # 1 - alpha_k
        "is_last": [],
    }

    B2 = 0.0
    num_clipped = 0
    max_noise_coeff = 0.0
    sum_noise_coeff = 0.0

    final_idx = n_intervals - 1

    for k in range(n_intervals):
        t_k = float(t_arr[k])
        t_next = float(t_arr[k + 1])
        h_k = t_next - t_k
        is_last = (k == final_idx)

        # The last interval is always handled by a deterministic integrator
        # (_exp_step or _ode_step), so its noise contribution is always 0.
        if is_last:
            gamma_k = compute_gamma(t_k, schedule_kind, amplitude)  # record for info
            alpha_k = 1.0
            noise_coeff = 0.0
        else:
            gamma_k = compute_gamma(t_k, schedule_kind, amplitude)
            raw_alpha = 1.0 - gamma_k * h_k
            if raw_alpha < 0.0:
                num_clipped += 1
            alpha_k = max(0.0, min(1.0, raw_alpha))
            noise_coeff = 1.0 - alpha_k

        per_step["t_k"].append(t_k)
        per_step["h_k"].append(h_k)
        per_step["gamma_k"].append(gamma_k)
        per_step["alpha_k"].append(alpha_k)
        per_step["noise_coeff"].append(noise_coeff)
        per_step["is_last"].append(is_last)

        B2 += noise_coeff ** 2
        max_noise_coeff = max(max_noise_coeff, noise_coeff)
        sum_noise_coeff += noise_coeff

    E_theory = (sigma ** 2) * B2

    return {
        "B2": B2,
        "E_theory": E_theory,
        "per_step": per_step,
        "num_clipped": num_clipped,
        "max_noise_coeff": max_noise_coeff,
        "sum_noise_coeff": sum_noise_coeff,
    }


def bisection_search(
    t_steps: torch.Tensor,
    schedule_kind: str,
    B_target: float,
    sigma: float,
    final_exp: bool = True,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> Tuple[float, float, int]:
    """Bisection search for amplitude A such that B2(schedule, A) == B_target.

    Args:
        t_steps: 1D tensor of t values
        schedule_kind: gamma schedule type
        B_target: target B2 value
        sigma: noise scale (only used for plausibility checks, B2 is sigma-invariant)
        final_exp: whether last step is deterministic exp
        tol: relative tolerance on B2 error
        max_iter: maximum bisection iterations

    Returns:
        (A_found, B2_found, num_iter)
    """
    # Find upper bound: B2 is monotonic increasing in A for all schedules.
    A_lo = 0.0
    A_hi = 1.0
    # Expand upper bound until B2 exceeds target
    for _ in range(50):
        stats_hi = _compute_per_step_stats(t_steps, schedule_kind, A_hi, sigma, final_exp)
        if stats_hi["B2"] >= B_target:
            break
        A_hi *= 2.0
    else:
        raise RuntimeError(
            f"Could not find upper bound for {schedule_kind}: "
            f"B_target={B_target}, B2(A={A_hi})={stats_hi['B2']}"
        )

    # Ensure lower bound is below target
    stats_lo = _compute_per_step_stats(t_steps, schedule_kind, A_lo, sigma, final_exp)
    if stats_lo["B2"] > B_target:
        # This shouldn't happen since A=0 gives B2=0, but handle edge cases
        A_lo = 0.0  # force

    for it in range(max_iter):
        A_mid = (A_lo + A_hi) / 2.0
        stats_mid = _compute_per_step_stats(t_steps, schedule_kind, A_mid, sigma, final_exp)
        B_mid = stats_mid["B2"]

        if B_target == 0.0:
            return 0.0, 0.0, 0

        rel_err = abs(B_mid - B_target) / B_target
        if rel_err < tol:
            return A_mid, B_mid, it + 1

        if B_mid < B_target:
            A_lo = A_mid
        else:
            A_hi = A_mid

    # Return best estimate even if tolerance not met
    A_final = (A_lo + A_hi) / 2.0
    stats_final = _compute_per_step_stats(t_steps, schedule_kind, A_final, sigma, final_exp)
    return A_final, stats_final["B2"], max_iter


def verify_B2_match(
    t_steps: torch.Tensor,
    sigma: float,
    schedules: Dict[str, Tuple[str, float]],
    final_exp: bool = True,
    tol: float = 1e-6,
) -> bool:
    """Verify that all schedules in the dict have the same B2.

    Args:
        t_steps: time grid tensor
        sigma: noise scale
        schedules: dict of {label: (schedule_kind, amplitude)}
        final_exp: final step is deterministic exp
        tol: relative tolerance

    Returns:
        True if all B2 values match within tolerance
    """
    b2_values = {}
    all_ok = True
    for label, (kind, amp) in schedules.items():
        stats = _compute_per_step_stats(t_steps, kind, amp, sigma, final_exp)
        b2_values[label] = stats["B2"]

    if not b2_values:
        return True

    ref_label = list(b2_values.keys())[0]
    ref_b2 = b2_values[ref_label]

    for label, b2 in b2_values.items():
        rel_err = abs(b2 - ref_b2) / ref_b2 if ref_b2 > 0 else 0.0
        status = "OK" if rel_err < tol else "FAIL"
        if rel_err >= tol:
            all_ok = False
        print(f"  {label:30s}  B2={b2:.12f}  rel_err={rel_err:.2e}  [{status}]")

    return all_ok


def compute_empirical_energy(
    noise_deltas: List[torch.Tensor],
) -> float:
    """Compute empirical injected noise energy from pre-computed noise deltas.

    E_empirical = sum_k mean(noise_delta_k^2)

    Args:
        noise_deltas: list of per-step noise tensors (1 - alpha_k) * sigma * epsilon_k

    Returns:
        scalar float
    """
    total = 0.0
    for delta in noise_deltas:
        total += float(delta.pow(2).mean().item())
    return total


def print_schedule_report(
    t_steps: torch.Tensor,
    schedule_kind: str,
    amplitude: float,
    sigma: float,
    final_exp: bool = True,
):
    """Print a detailed report for a schedule on a given time grid."""
    stats = _compute_per_step_stats(t_steps, schedule_kind, amplitude, sigma, final_exp)
    print(f"Schedule: {schedule_kind}, A={amplitude:.6f}")
    print(f"  B2                  = {stats['B2']:.12f}")
    print(f"  E_theory            = {stats['E_theory']:.6f}")
    print(f"  num_clipped_steps   = {stats['num_clipped']}")
    print(f"  max_noise_coeff     = {stats['max_noise_coeff']:.6f}")
    print(f"  sum_noise_coeff     = {stats['sum_noise_coeff']:.6f}")
    print(f"  n_intervals         = {len(stats['per_step']['t_k'])}")
    return stats
