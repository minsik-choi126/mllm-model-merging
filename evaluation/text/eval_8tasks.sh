#!/bin/bash
# 8-task text-eval driver. Two protocols are supported via --protocol {default,instruct}.
# Each task is its own lm_eval invocation so flags can differ per task.
#
# --protocol default (community default; lm-eval-harness yaml defaults, no
#                     chat-template forced):
#     gsm8k_cot                 8-shot, no chat tpl, max_gen 512
#     ifeval                    0-shot, no chat tpl, max_gen 1280
#     gpqa_diamond_cot_zeroshot 0-shot, no chat tpl, max_gen 1024
#     mmlu_pro                  5-shot CoT, no chat tpl, max_gen 2048
#     eq_bench                  0-shot, no chat tpl, max_gen 512
#     mmlu                      5-shot LL, no chat tpl
#     boolq                     0-shot LL, no chat tpl
#     truthfulqa_mc2            0-shot LL, no chat tpl
#
# --protocol instruct (instruct-aware: every task 0-shot + chat-template ON;
#                      this matches how chat-templated post-trained models are
#                      actually deployed):
#     all 8 tasks 0-shot, --apply_chat_template ON, generation knobs identical
#     to the `default` protocol where applicable.
#
# Usage:
#   bash eval_8tasks.sh --model <path_or_hf_id> --protocol {default,instruct} [options]
#
# Options:
#   --model        <path>   HF model ID or local directory (required)
#   --protocol     <name>   default | instruct  (required)
#   --tokenizer    <path>   Separate tokenizer (default: same as --model)
#   --gpu          <N>      CUDA_VISIBLE_DEVICES index (default: 0)
#   --output       <dir>    Result directory (default: eval_results/text_8tasks_<slug>_<proto>)
#   --tasks        <list>   Comma-separated subset (default: all 8)
#   --batch-size   <N>      lm_eval batch size (default: 8; "auto" drops to 1 with chat tpl + long max_gen)
#   --mmlu-pro-limit <f>    Subsample mmlu_pro to fraction f (default: full set)
#   --mmlu-pro-seed  <n>    mmlu_pro subsample seed (default 42)
#
# Environment:
#   LM_EVAL_BIN              Path to lm_eval binary (default: `lm_eval` on PATH)
#   HUGGING_FACE_HUB_TOKEN   Optional HF token for gated models

set -uo pipefail

MODEL=""
PROTOCOL=""
TOKENIZER=""
GPU=0
OUTPUT=""
TASKS_REQUESTED="boolq,eq_bench,gpqa_diamond_cot_zeroshot,gsm8k_cot,ifeval,mmlu,truthfulqa_mc2,mmlu_pro"
MMLU_PRO_LIMIT=""
MMLU_PRO_SEED="42"
BATCH_SIZE="8"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)            MODEL="$2";            shift 2 ;;
        --protocol)         PROTOCOL="$2";         shift 2 ;;
        --tokenizer)        TOKENIZER="$2";        shift 2 ;;
        --gpu)              GPU="$2";              shift 2 ;;
        --output)           OUTPUT="$2";           shift 2 ;;
        --tasks)            TASKS_REQUESTED="$2";  shift 2 ;;
        --mmlu-pro-limit)   MMLU_PRO_LIMIT="$2";   shift 2 ;;
        --mmlu-pro-seed)    MMLU_PRO_SEED="$2";    shift 2 ;;
        --batch-size)       BATCH_SIZE="$2";       shift 2 ;;
        -h|--help)
            sed -n '2,40p' "$0"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "Error: --model is required" >&2; exit 1
fi
if [[ "$PROTOCOL" != "default" && "$PROTOCOL" != "instruct" ]]; then
    echo "Error: --protocol must be one of {default,instruct} (got '${PROTOCOL}')" >&2; exit 1
fi

LM_EVAL_BIN="${LM_EVAL_BIN:-lm_eval}"

if [[ -z "$OUTPUT" ]]; then
    SLUG=$(echo "$MODEL" | tr '/.' '__')
    OUTPUT="eval_results/text_8tasks_${SLUG}_${PROTOCOL}"
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
echo "Protocol: $PROTOCOL"
echo "Model:    $MODEL"
echo "GPU:      $GPU"
echo "Output:   $OUTPUT"
echo "Tasks:    $(echo $TASKS_CLEAN | tr '\n' ',' | sed 's/,$//')"
echo "Batch:    $BATCH_SIZE"
echo "========================================"

RC=0

# Resolve per-task (num_fewshot, chat_yes_no) given protocol.
fewshot_for_task() {
    local task="$1"
    if [[ "$PROTOCOL" == "instruct" ]]; then
        echo 0
        return
    fi
    case "$task" in
        gsm8k_cot)                    echo 8 ;;
        mmlu)                         echo 5 ;;
        mmlu_pro)                     echo 5 ;;
        ifeval|gpqa_diamond_cot_zeroshot|eq_bench|boolq|truthfulqa_mc2) echo 0 ;;
        *)                            echo 0 ;;
    esac
}

chat_for_task() {
    if [[ "$PROTOCOL" == "instruct" ]]; then
        echo yes
    else
        echo no
    fi
}

run_task() {
    local task="$1"
    shift
    local extra_flags=("$@")
    local fewshot
    local chat
    fewshot=$(fewshot_for_task "$task")
    chat=$(chat_for_task)

    local chat_flag=""
    if [[ "$chat" == "yes" ]]; then
        chat_flag="--apply_chat_template"
    fi

    echo
    echo "[$(date +%H:%M)] -> ${task}  (n_fewshot=${fewshot}, chat_template=${chat})"

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
        || { echo "[WARN] ${task} failed (exit $?)"; RC=1; }
}

want() { echo "$TASKS_CLEAN" | grep -qx "$1"; }

# Greedy generation knobs are protocol-independent; only num_fewshot / chat-template differ.
if want "gsm8k_cot"; then
    run_task "gsm8k_cot" \
        --gen_kwargs "do_sample=False,temperature=0,max_gen_toks=512"
fi
if want "ifeval"; then
    run_task "ifeval" \
        --gen_kwargs "do_sample=False,temperature=0,max_gen_toks=1280"
fi
if want "gpqa_diamond_cot_zeroshot"; then
    run_task "gpqa_diamond_cot_zeroshot" \
        --gen_kwargs "do_sample=False,temperature=0,max_gen_toks=1024"
fi
if want "mmlu_pro"; then
    extra_args=( --gen_kwargs "do_sample=False,temperature=0,max_gen_toks=2048" )
    if [[ -n "$MMLU_PRO_LIMIT" ]]; then
        extra_args+=( --limit "$MMLU_PRO_LIMIT" --seed "$MMLU_PRO_SEED" )
    fi
    run_task "mmlu_pro" "${extra_args[@]}"
fi
if want "eq_bench"; then
    run_task "eq_bench" \
        --gen_kwargs "do_sample=False,temperature=0,max_gen_toks=512"
fi
if want "mmlu"; then
    run_task "mmlu"
fi
if want "boolq"; then
    run_task "boolq"
fi
if want "truthfulqa_mc2"; then
    run_task "truthfulqa_mc2"
fi

echo
echo "[$(date +%H:%M)] All requested tasks complete → ${OUTPUT}  (exit code: $RC)"
exit $RC
