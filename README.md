# FLRS: Front-Loaded Rollback Sampling for ELF

This repository contains the PyTorch implementation and evaluation code for
**Front-Loaded Rollback Sampling (FLRS)**. FLRS is training-free: it keeps the
ELF checkpoint, time grid, decoder, and number of network evaluations fixed,
and changes only the rollback-strength schedule used by the released sampler.

FLRS schedules the full native rollback operator. The coefficient therefore
jointly affects time rollback, deterministic contraction, and Gaussian
injection; it should not be interpreted as an isolated post-hoc noise scale.

## Installation

```bash
conda create -n flrs python=3.10 -y
conda activate flrs
pip install -r requirements.txt
```

The code builds on the official
[ELF implementation](https://github.com/lillian039/ELF) and uses its released
checkpoints. For example, `embedded-language-flows/ELF-B-owt-torch` is loaded
automatically when supplied as `--checkpoint_path`.

## Quick start

Run FLRS on the ELF-B OpenWebText checkpoint:

```bash
NGPU=1 bash scripts/launch.sh eval \
    src/configs/training_configs/train_owt_ELF-B.yml \
    --checkpoint_path embedded-language-flows/ELF-B-owt-torch \
    --seeds 42 \
    --config_override \
      "sampling_configs_path=src/configs/sampling_configs/flrs_sampling_configs.yml" \
    --config_override use_bf16=true \
    --config_override use_compile=true
```

The default paper configuration is represented as:

```yaml
- sampling_method: flrs
  num_sampling_steps: [64]
  cfgs: [1]
  self_cond_cfg_scales: [3]
  rollback_gamma_start: 2.0
  rollback_gamma_end: 0.0
  rollback_power: 1.0
  time_schedule: logit_normal
```

The schedule is

```text
gamma(t) = gamma_end + (gamma_start - gamma_end) * (1 - t)^p.
```

`rollback_power: 1` gives FLRS-Linear and `rollback_power: 2` gives
FLRS-Quadratic.

## Reproducing the main comparisons

- Cross-model unconditional evaluation:
  `scripts/eval_elf_b.sh`, `scripts/eval_elf_m.sh`, and
  `scripts/eval_elf_l.sh`. These use five ELF-B seeds, three ELF-M/L seeds,
  and reset the RNG before each sampler.
- WMT14 German-to-English:
  `src/configs/sampling_configs/wmt_sampling_configs.yml`
- XSum:
  `src/configs/sampling_configs/xsum_sampling_configs.yml`
- Energy-matched allocation:

```bash
python scripts/run_energy_matched_experiment.py \
    --config src/configs/training_configs/train_owt_ELF-B.yml \
    --checkpoint embedded-language-flows/ELF-B-owt-torch \
    --nfe 64 \
    --seeds 41,42,43,44,45 \
    --num_samples 1000 \
    --output_dir outputs/energy_matched
```

Use `--dry_run` first to verify the matched direct-injection budgets without
loading the model.

For multi-stage diagnostic controls, paper labels denote prescribed NFE
budgets. The terminal one-call interval gives 31/63 actual calls for
two-stage 16/32-step controls and 46 for the three-round 16-step control.

Compute the paper's repetition diagnostics from generated JSONL:

```bash
python scripts/compute_repetition_metrics.py \
    --input outputs/<run>/all_generated_<epoch>_<step>.jsonl
```

The script reports repetition incidence (RI) and repeated-window coverage
(RSC) using the shared event "a 5-gram occurs at least three times."

Summarize one model's paired runs with:

```bash
python scripts/summarize_cross_model.py \
    --input outputs/elf_b-cross_model \
    --model ELF-B
```

For human evaluation, fill the anonymized item-level schema documented in
`human_eval/README.md`, then recompute vote percentages and Fleiss' kappa:

```bash
python scripts/evaluate_human_preferences.py \
    --input human_eval/annotations.csv \
    --output human_eval/summary.json
```

## Code map

- `src/utils/generation_utils.py`: sampler dispatch and FLRS schedule
- `src/utils/sampling_utils.py`: native ELF rollback-and-advance update
- `src/utils/schedule_utils.py`: energy-matched schedule utilities
- `src/configs/sampling_configs/`: paper and diagnostic configurations
- `scripts/run_energy_matched_experiment.py`: paired, shared-noise experiment
- `scripts/compute_repetition_metrics.py`: RSC and RI implementation
- `scripts/evaluate_human_preferences.py`: human votes and Fleiss' kappa
- `scripts/summarize_cross_model.py`: cross-seed mean and SD

## Naming and compatibility

The public method identifier is `flrs`, with parameters
`rollback_gamma_start`, `rollback_gamma_end`, and `rollback_power`. Code from
early experiments may use `sde_anneal`, `sde_gamma`, `sde_gamma_end`, and
`sde_anneal_power`; these names remain accepted only as legacy input aliases.
New output directories and configurations always use the FLRS terminology.

## Scope

The included implementation is specific to ELF's rollback operator. Applying
the broader temporal-allocation principle to a model family without this
operator requires a newly defined transition and separate compute-matched
validation.

## License

This code is released under the [MIT License](LICENSE).

The repository is derived from the MIT-licensed ELF PyTorch implementation.
The original ELF portions retain their upstream copyright notice; the FLRS
implementation and accompanying experiment code are modifications released
under the same license.

