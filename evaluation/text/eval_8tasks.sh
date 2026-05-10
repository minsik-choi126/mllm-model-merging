#!/bin/bash
# 8-task text-eval, with **per-task protocols** chosen to reproduce the
# corresponding Qwen2.5-Instruct blog/tech-report numbers (within the
# ±1-5 pt drift inherent to lm-evaluation-harness vs. Qwen's eval stacks).
#
# Per-task protocol decisions (see evaluation/text/README.md and the eval-protocol
# audit for sources):
#
#   gsm8k_cot                  : 0-shot CoT + chat template + max_gen_toks=512
#                                (Qwen blog: 91.6 ; harness yaml default is 8-shot
#                                — must override num_fewshot=0)
#   ifeval                     : 0-shot strict + chat template + max_gen_toks=1280
#                                (Qwen blog: 71.2)
#   gpqa_diamond_cot_zeroshot  : 0-shot **CoT** zero-shot + chat template
#                                (Qwen blog: 36.4 ; the bare `gpqa_diamond_zeroshot`
#                                does NOT elicit CoT and lands ~29)
#   mmlu_pro                   : 5-shot CoT generation + chat template
#                                (Qwen blog: 56.3 ; task default is 5-shot CoT)
#                                Optional: --mmlu-pro-limit for subsampled comparisons
#   eq_bench                   : 0-shot + chat template
#                                (no Qwen-published target; lm-eval is v2.1, not the
#                                eqbench.com v3 leaderboard — incomparable to v3)
#
# The following three tasks have no Qwen-published Instruct number and are
# log-likelihood multi-choice tasks where chat-template wrapping degrades the
# answer-token LL — we run them WITHOUT --apply_chat_template:
#
#   mmlu, boolq, truthfulqa_mc2 : 0-shot LL, no chat template
#                                (mmlu defaults to 5-shot LL with `mmlu` task)
#
# All steps write to the same --output directory; parse_results.py reads them
# recursively. Each task is its own lm_eval invocation so flags can differ.
#
# Usage:
#   bash eval_8tasks.sh --model <path_or_hf_id> [options]
#
# Options:
#   --model        <path>   HF model ID or local directory (required)
#   --tokenizer    <path>   Separate tokenizer (default: same as --model)
#   --gpu          <N>      CUDA_VISIBLE_DEVICES index (default: 0)
#   --output       <dir>    Result directory (default: eval_results/text_8tasks_<slug>)
#   --tasks        <list>   Comma-separated subset of the 8 (default: all 8)
#   --mmlu-pro-limit <f>    Subsample mmlu_pro to fraction f (default: full set;
#                           use 0.0416 for the historical ~500-item stratified subset)
#   --mmlu-pro-seed  <n>    mmlu_pro subsample seed (default 42; ignored if no --limit)
#
# Environment:
#   LM_EVAL_BIN              Path to lm_eval binary (default: `lm_eval` on PATH)
#   HUGGING_FACE_HUB_TOKEN   Optional HF token for gated models

set -uo pipefail

MODEL=""
TOKENIZER=""
GPU=0
OUTPUT=""
TASKS_REQUESTED="boolq,eq_bench,gpqa_diamond_cot_zeroshot,gsm8k_cot,ifeval,mmlu,truthfulqa_mc2,mmlu_pro"
MMLU_PRO_LIMIT=""
MMLU_PRO_SEED="42"
# `auto` over-conservatively drops to bs=1 with chat template + long max_gen_toks
# (47-hour ETA on a single 7B eval); set an explicit number to actually use the GPU.
BATCH_SIZE="8"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)            MODEL="$2";            shift 2 ;;
        --tokenizer)        TOKENIZER="$2";        shift 2 ;;
        --gpu)              GPU="$2";              shift 2 ;;
        --output)           OUTPUT="$2";           shift 2 ;;
        --tasks)            TASKS_REQUESTED="$2";  shift 2 ;;
        --mmlu-pro-limit)   MMLU_PRO_LIMIT="$2";   shift 2 ;;
        --mmlu-pro-seed)    MMLU_PRO_SEED="$2";    shift 2 ;;
        --batch-size)       BATCH_SIZE="$2";       shift 2 ;;
        -h|--help)
            sed -n '2,55p' "$0"
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
mkdir -p "$OUTPUT"

if [[ -n "$TOKENIZER" ]]; then
    TOK_ARG=",tokenizer=${TOKENIZER}"
else
    TOK_ARG=""
fi
MODEL_ARGS_BASE="pretrained=${MODEL}${TOK_ARG},dtype=bfloat16,trust_remote_code=True"

TASKS_CLEAN=$(echo "$TASKS_REQUESTED" | tr -d ' ' | tr ',' '\n' | awk 'NF')
echo "========================================"
echo "Model:     $MODEL"
echo "Tokenizer: ${TOKENIZER:-"(same as model)"}"
echo "GPU:       $GPU"
echo "Output:    $OUTPUT"
echo "Tasks:     $(echo $TASKS_CLEAN | tr '\n' ',' | sed 's/,$//')"
echo "lm_eval:   $LM_EVAL_BIN"
echo "========================================"

RC=0

# Helper: run one task with given flags.
run_task() {
    local task="$1"
    local fewshot="$2"
    local chat="$3"   # "yes" | "no"
    shift 3
    local extra_flags=("$@")

    local chat_flag=""
    if [[ "$chat" == "yes" ]]; then
        chat_flag="--apply_chat_template"
    fi

    local task_label="${task} (n_fewshot=${fewshot}, chat_template=${chat})"
    echo
    echo "[$(date +%H:%M)] -> ${task_label}"
    echo "  flags: ${chat_flag} --num_fewshot ${fewshot} ${extra_flags[*]}"

    CUDA_VISIBLE_DEVICES=${GPU} \
    "$LM_EVAL_BIN" \
        --model hf \
        --model_args "${MODEL_ARGS_BASE}" \
        --tasks "${task}" \
        --num_fewshot "${fewshot}" \
        --batch_size "${BATCH_SIZE}" \
        ${chat_flag} \
        --output_path "${OUTPUT}" \
        "${extra_flags[@]}" \
        || { echo "[WARN] ${task_label} failed (exit $?)"; RC=1; }
}

want() {
    echo "$TASKS_CLEAN" | grep -qx "$1"
}

# ── 1. gsm8k_cot ─────────────────────────────────────────────────────────────
# 0-shot CoT + chat template + max_gen_toks=512, greedy. Qwen blog: 91.6.
if want "gsm8k_cot"; then
    run_task "gsm8k_cot" 0 "yes" \
        --gen_kwargs "do_sample=False,temperature=0,max_gen_toks=512"
fi

# ── 2. ifeval ────────────────────────────────────────────────────────────────
# 0-shot generation + chat template + max_gen_toks=1280, greedy. Qwen blog: 71.2 (prompt-strict).
if want "ifeval"; then
    run_task "ifeval" 0 "yes" \
        --gen_kwargs "do_sample=False,temperature=0,max_gen_toks=1280"
fi

# ── 3. gpqa_diamond_cot_zeroshot ─────────────────────────────────────────────
# 0-shot CoT zero-shot + chat template. Qwen blog: 36.4.
# (Note: the alternate `gpqa_diamond_zeroshot` task does NOT elicit CoT and scores ~29.)
if want "gpqa_diamond_cot_zeroshot"; then
    run_task "gpqa_diamond_cot_zeroshot" 0 "yes" \
        --gen_kwargs "do_sample=False,temperature=0,max_gen_toks=1024"
fi

# ── 4. mmlu_pro ──────────────────────────────────────────────────────────────
# 5-shot CoT (task default) + chat template. Qwen blog: 56.3 (full set).
# Optionally subsample with --mmlu-pro-limit (legacy: 0.0416 ≈ 500 items, ±2-3 pt noise).
if want "mmlu_pro"; then
    extra_args=( --gen_kwargs "do_sample=False,temperature=0,max_gen_toks=2048" )
    if [[ -n "$MMLU_PRO_LIMIT" ]]; then
        extra_args+=( --limit "$MMLU_PRO_LIMIT" --seed "$MMLU_PRO_SEED" )
    fi
    run_task "mmlu_pro" 5 "yes" "${extra_args[@]}"
fi

# ── 5. eq_bench ──────────────────────────────────────────────────────────────
# 0-shot + chat template. lm-eval is v2.1, not the v3 leaderboard. No Qwen target.
if want "eq_bench"; then
    run_task "eq_bench" 0 "yes" \
        --gen_kwargs "do_sample=False,temperature=0,max_gen_toks=512"
fi

# ── 6. mmlu (LL task — chat template OFF) ───────────────────────────────────
# 5-shot LL. No Qwen 7B-Instruct target for vanilla MMLU (Qwen reports MMLU-redux 75.4).
if want "mmlu"; then
    run_task "mmlu" 5 "no"
fi

# ── 7. boolq (LL task — chat template OFF) ──────────────────────────────────
# 0-shot LL. No Qwen target.
if want "boolq"; then
    run_task "boolq" 0 "no"
fi

# ── 8. truthfulqa_mc2 (LL task — chat template OFF) ─────────────────────────
# 0-shot LL. No Qwen target.
if want "truthfulqa_mc2"; then
    run_task "truthfulqa_mc2" 0 "no"
fi

echo
echo "[$(date +%H:%M)] All requested tasks complete → ${OUTPUT}  (exit code: $RC)"
exit $RC
