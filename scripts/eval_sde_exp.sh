#!/usr/bin/env bash
# ===========================================================================
# ELF evaluation — SDE + exp hybrid
#   t < threshold:  SDE with noise injection (error correction)
#   t >= threshold: exponential integrator (handles t→1 singularity)
# ===========================================================================
set -euo pipefail

NGPU=${NGPU:-4}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_SAMPLES=${NUM_SAMPLES:-1000}
SEEDS=${SEEDS:-"42,123,456"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/elf_b-owt_sde_exp_eval"}
CHECKPOINT=${CHECKPOINT:-"embedded-language-flows/ELF-B-owt-torch"}

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

TRAIN_CONFIG="src/configs/training_configs/train_owt_ELF-B.yml"
SAMPLING_CONFIG="src/configs/sampling_configs/sde_exp_sampling_configs.yml"

echo "============================================================"
echo " ELF SDE+exp Hybrid Evaluation"
echo "============================================================"
echo " GPUs:               $NGPU"
echo " Per-GPU batch size: $BATCH_SIZE"
echo " Global batch size:  $(( NGPU * BATCH_SIZE ))"
echo " Num samples:        $NUM_SAMPLES"
echo " Seeds:              $SEEDS"
echo " Checkpoint:         $CHECKPOINT"
echo " Output dir:         $OUTPUT_DIR"
echo " Sampling config:    $SAMPLING_CONFIG"
echo "============================================================"
echo ""

NGPU="$NGPU" bash scripts/launch.sh eval "$TRAIN_CONFIG" \
    --checkpoint_path "$CHECKPOINT" \
    --seeds "$SEEDS" \
    --config_override "global_batch_size=$(( NGPU * BATCH_SIZE ))" \
    --config_override "num_samples=$NUM_SAMPLES" \
    --config_override "output_dir=$OUTPUT_DIR" \
    --config_override "sampling_configs_path=$SAMPLING_CONFIG" \
    --config_override "use_bf16=true" \
    --config_override "use_compile=true" \
    --config_override "use_wandb=false" \
    --config_override "online_eval=true" \
    --config_override "eval_ppl_batch_size=8" \
    --config_override "hf_repo_id=null"

echo ""
echo "Done. Results saved in: $OUTPUT_DIR"
