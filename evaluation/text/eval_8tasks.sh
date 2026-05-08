#!/bin/bash
# 8-task text degradation eval (lm-evaluation-harness).
#
# Runs in two stages:
#   Step 1: 7 full-set tasks (boolq, eq_bench, gpqa_diamond_zeroshot,
#           gsm8k_cot, ifeval, mmlu, truthfulqa_mc2)
#   Step 2: mmlu_pro with --limit 0.0416 --seed 42 (≈500 stratified items)
#
# Both steps write to the same --output directory; parse_results.py reads
# them recursively.
#
# Usage:
#   bash eval_8tasks.sh --model <path_or_hf_id> [options]
#
# Options:
#   --model        <path>   HF model ID or local directory (required)
#   --tokenizer    <path>   Separate tokenizer (default: same as --model)
#   --gpu          <N>      CUDA_VISIBLE_DEVICES index (default: 0)
#   --output       <dir>    Result directory (default: eval_results/text_8tasks_<slug>)
#   --tasks        <list>   Comma-separated task list (default: all 8)
#   --no-chat-template      Disable apply_chat_template
#   --mmlu-pro-limit <f>    Override mmlu_pro --limit (default: 0.0416)
#   --mmlu-pro-seed  <n>    Override mmlu_pro --seed  (default: 42)
#
# Environment:
#   LM_EVAL_BIN              Path to lm_eval binary (default: `lm_eval` on PATH)
#   HUGGING_FACE_HUB_TOKEN   Optional HF token for gated models

set -uo pipefail

MODEL=""
TOKENIZER=""
GPU=0
OUTPUT=""
TASKS="boolq,eq_bench,gpqa_diamond_zeroshot,gsm8k_cot,ifeval,mmlu,truthfulqa_mc2,mmlu_pro"
CHAT_TEMPLATE_FLAG="--apply_chat_template"
MMLU_PRO_LIMIT="0.0416"
MMLU_PRO_SEED="42"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)            MODEL="$2";          shift 2 ;;
        --tokenizer)        TOKENIZER="$2";      shift 2 ;;
        --gpu)              GPU="$2";            shift 2 ;;
        --output)           OUTPUT="$2";         shift 2 ;;
        --tasks)            TASKS="$2";          shift 2 ;;
        --mmlu-pro-limit)   MMLU_PRO_LIMIT="$2"; shift 2 ;;
        --mmlu-pro-seed)    MMLU_PRO_SEED="$2";  shift 2 ;;
        --no-chat-template) CHAT_TEMPLATE_FLAG=""; shift ;;
        -h|--help)
            sed -n '2,28p' "$0"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "Error: --model is required" >&2
    exit 1
fi

LM_EVAL_BIN="${LM_EVAL_BIN:-lm_eval}"

if [[ -z "$OUTPUT" ]]; then
    SLUG=$(echo "$MODEL" | tr '/.' '__')
    OUTPUT="eval_results/text_8tasks_${SLUG}"
fi

if [[ -n "$TOKENIZER" ]]; then
    MODEL_ARGS="pretrained=${MODEL},tokenizer=${TOKENIZER},dtype=bfloat16,trust_remote_code=True"
else
    MODEL_ARGS="pretrained=${MODEL},dtype=bfloat16,trust_remote_code=True"
fi

mkdir -p "$OUTPUT"

TASKS_CLEAN=$(echo "$TASKS" | tr -d ' ')
STEP1_TASKS=$(printf '%s\n' "$TASKS_CLEAN" | tr ',' '\n' | awk '$0 != "mmlu_pro" && $0 != ""' | paste -sd, -)
RUN_MMLU_PRO=0
if printf '%s\n' "$TASKS_CLEAN" | tr ',' '\n' | grep -qx 'mmlu_pro'; then
    RUN_MMLU_PRO=1
fi

echo "========================================"
echo "Model:     $MODEL"
echo "Tokenizer: ${TOKENIZER:-"(same as model)"}"
echo "GPU:       $GPU"
echo "Output:    $OUTPUT"
echo "Tasks:     $TASKS_CLEAN"
echo "lm_eval:   $LM_EVAL_BIN"
echo "Chat template: ${CHAT_TEMPLATE_FLAG:+enabled}${CHAT_TEMPLATE_FLAG:-disabled}"
echo "========================================"

RC=0

# ── Step 1: 7 tasks (full sets) ───────────────────────────────────────────────
if [[ -n "$STEP1_TASKS" ]]; then
    echo "[$(date +%H:%M)] Step 1/2: ${STEP1_TASKS}"
    CUDA_VISIBLE_DEVICES=${GPU} \
    "$LM_EVAL_BIN" \
        --model hf \
        --model_args "${MODEL_ARGS}" \
        --tasks "${STEP1_TASKS}" \
        --batch_size auto \
        ${CHAT_TEMPLATE_FLAG} \
        --output_path "${OUTPUT}" \
        || { echo "[WARN] Step 1 failed (exit $?)"; RC=1; }
else
    echo "[$(date +%H:%M)] Step 1/2: skipped (no non-mmlu_pro tasks requested)"
fi

# ── Step 2: mmlu_pro stratified subsample (~500) ──────────────────────────────
if [[ "$RUN_MMLU_PRO" -eq 1 ]]; then
    echo "[$(date +%H:%M)] Step 2/2: mmlu_pro (limit=${MMLU_PRO_LIMIT}, seed=${MMLU_PRO_SEED})"
    CUDA_VISIBLE_DEVICES=${GPU} \
    "$LM_EVAL_BIN" \
        --model hf \
        --model_args "${MODEL_ARGS}" \
        --tasks "mmlu_pro" \
        --batch_size auto \
        ${CHAT_TEMPLATE_FLAG} \
        --limit "${MMLU_PRO_LIMIT}" \
        --seed "${MMLU_PRO_SEED}" \
        --output_path "${OUTPUT}" \
        || { echo "[WARN] Step 2 (mmlu_pro) failed (exit $?)"; RC=1; }
else
    echo "[$(date +%H:%M)] Step 2/2: skipped (mmlu_pro not requested)"
fi

echo "[$(date +%H:%M)] Done → ${OUTPUT}  (exit code: $RC)"
exit $RC
