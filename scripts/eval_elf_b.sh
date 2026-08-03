#!/usr/bin/env bash
# ELF-B cross-model-table evaluation: Fixed Rollback versus FLRS-Linear.
set -euo pipefail

NGPU=${NGPU:-4}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_SAMPLES=${NUM_SAMPLES:-1000}
SEEDS=${SEEDS:-"41,42,43,44,45"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/elf_b-cross_model"}
CHECKPOINT=${CHECKPOINT:-"embedded-language-flows/ELF-B-owt-torch"}

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

NGPU="$NGPU" bash scripts/launch.sh eval \
    src/configs/training_configs/train_owt_ELF-B.yml \
    --checkpoint_path "$CHECKPOINT" \
    --seeds "$SEEDS" \
    --paired_sampling \
    --config_override "global_batch_size=$(( NGPU * BATCH_SIZE ))" \
    --config_override "num_samples=$NUM_SAMPLES" \
    --config_override "output_dir=$OUTPUT_DIR" \
    --config_override "sampling_configs_path=src/configs/sampling_configs/cross_model_sampling_configs.yml" \
    --config_override "use_bf16=true" \
    --config_override "use_compile=true" \
    --config_override "use_wandb=false" \
    --config_override "online_eval=true" \
    --config_override "eval_ppl_batch_size=8" \
    --config_override "hf_repo_id=null"

echo "Done. Results saved in: $OUTPUT_DIR"
