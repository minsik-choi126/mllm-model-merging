#!/bin/bash
# Run text eval over a matrix of (models, protocols), splitting jobs across
# multiple GPUs in round-robin order. Wrapper around eval_8tasks.sh.
#
# After all jobs finish, runs parse_results.py once per protocol that has at
# least an LLM and a VLM-LM result, producing comparison tables.
#
# Usage:
#   bash run_eval_matrix.sh \
#       --models llm:/path/Qwen2.5-7B-Instruct \
#                vlm_lm:/path/extracted/qwen25vl_7b_lm \
#                merged:/path/merged/qwen25_epull \
#       --protocols default,instruct \
#       --gpus 0,1 \
#       --output-root eval_results
#
# Options:
#   --models      space-separated NAME:PATH entries (required, ≥1)
#                   NAME is used in the output directory; PATH is what
#                   eval_8tasks.sh receives as --model.
#   --protocols   comma-separated subset of {default, instruct} (default both)
#   --gpus        comma-separated GPU indices to round-robin across (default 0)
#   --output-root output root dir; per-job dir is <root>/<NAME>_<protocol>
#                   (default eval_results)
#   --tasks       passthrough to eval_8tasks.sh (default: all 8)
#   --batch-size  passthrough (default 8)
#   --mmlu-pro-limit  passthrough
#   --mmlu-pro-seed   passthrough
#   --no-compare  skip the parse_results.py step at the end
#
# The script is launched once and runs to completion; long runs should be
# invoked in the background by the caller.

set -uo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
SCRIPT="$REPO/evaluation/text/eval_8tasks.sh"
PARSE="$REPO/evaluation/text/parse_results.py"

if [[ ! -f "$SCRIPT" ]]; then
    echo "Error: cannot locate $SCRIPT" >&2; exit 1
fi

MODELS=()
PROTOCOLS="default,instruct"
GPUS="0"
OUT_ROOT="eval_results"
TASKS="boolq,eq_bench,gpqa_diamond_cot_zeroshot,gsm8k_cot,ifeval,mmlu,truthfulqa_mc2,mmlu_pro"
BATCH_SIZE=8
MMLU_PRO_LIMIT=""
MMLU_PRO_SEED="42"
DO_COMPARE=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                MODELS+=("$1"); shift
            done
            ;;
        --protocols)        PROTOCOLS="$2";       shift 2 ;;
        --gpus)             GPUS="$2";            shift 2 ;;
        --output-root)      OUT_ROOT="$2";        shift 2 ;;
        --tasks)            TASKS="$2";           shift 2 ;;
        --batch-size)       BATCH_SIZE="$2";      shift 2 ;;
        --mmlu-pro-limit)   MMLU_PRO_LIMIT="$2";  shift 2 ;;
        --mmlu-pro-seed)    MMLU_PRO_SEED="$2";   shift 2 ;;
        --no-compare)       DO_COMPARE=0;         shift ;;
        -h|--help)          sed -n '2,32p' "$0";  exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "Error: --models requires at least one NAME:PATH entry" >&2; exit 1
fi

mkdir -p "$OUT_ROOT"
QLOG="$OUT_ROOT/queue.log"
echo "[$(date +%F\ %T)] matrix start: ${#MODELS[@]} model(s) x $(echo $PROTOCOLS | tr ',' ' ' | wc -w) protocol(s) on GPUs $GPUS" | tee -a "$QLOG"

IFS=',' read -ra GPU_ARR <<< "$GPUS"
IFS=',' read -ra PROTO_ARR <<< "$PROTOCOLS"

# Build job list: one (model, protocol) per entry.
JOB_NAMES=()
JOB_MODELS=()
JOB_PROTOS=()
for entry in "${MODELS[@]}"; do
    name="${entry%%:*}"
    path="${entry#*:}"
    if [[ -z "$name" || -z "$path" || "$name" == "$path" ]]; then
        echo "Error: bad --models entry '$entry' (expected NAME:PATH)" >&2; exit 1
    fi
    for proto in "${PROTO_ARR[@]}"; do
        JOB_NAMES+=("${name}_${proto}")
        JOB_MODELS+=("$path")
        JOB_PROTOS+=("$proto")
    done
done

# Per-GPU sequential chain (background).
CHAIN_PIDS=()
for gi in "${!GPU_ARR[@]}"; do
    gpu=${GPU_ARR[$gi]}
    chain_log="$OUT_ROOT/.chain_gpu${gpu}.log"
    (
        for ji in "${!JOB_NAMES[@]}"; do
            if [[ $((ji % ${#GPU_ARR[@]})) -ne $gi ]]; then continue; fi
            name=${JOB_NAMES[$ji]}
            model=${JOB_MODELS[$ji]}
            proto=${JOB_PROTOS[$ji]}
            out="$OUT_ROOT/${name}"
            mkdir -p "$out"
            echo "[$(date +%F\ %T)] [GPU${gpu}] start: ${name} -> ${out}" | tee -a "$QLOG"

            # Skip tasks that already have a results JSON in this dir (re-runnable
            # idempotently). lm-eval writes one results-*.json per (task) call;
            # we look for tasks present in any prior JSON in the dir.
            requested_clean=$(echo "$TASKS" | tr -d ' ')
            missing=$(OUT_DIR="$out" REQ="$requested_clean" python -c '
import json, glob, os
out = os.environ["OUT_DIR"]
done = set()
for jf in glob.glob(os.path.join(out, "**", "results*.json"), recursive=True):
    try:
        d = json.load(open(jf))
    except Exception:
        continue
    for t, vals in d.get("results", {}).items():
        if t.startswith("mmlu_pro_"):
            continue
        if t.startswith("mmlu_") and t != "mmlu":
            continue
        if any(isinstance(v, float) and not k.endswith("_stderr")
               for k, v in vals.items() if k != "alias"):
            done.add(t)
req = set(os.environ["REQ"].split(",")) - {""}
print(",".join(sorted(req - done)))
')
            done_tasks=$(OUT_DIR="$out" REQ="$requested_clean" python -c '
import json, glob, os
out = os.environ["OUT_DIR"]
done = set()
for jf in glob.glob(os.path.join(out, "**", "results*.json"), recursive=True):
    try:
        d = json.load(open(jf))
    except Exception:
        continue
    for t, vals in d.get("results", {}).items():
        if t.startswith("mmlu_pro_"):
            continue
        if t.startswith("mmlu_") and t != "mmlu":
            continue
        if any(isinstance(v, float) and not k.endswith("_stderr")
               for k, v in vals.items() if k != "alias"):
            done.add(t)
req = set(os.environ["REQ"].split(",")) - {""}
print(",".join(sorted(done & req)))
')
            if [[ -z "$missing" ]]; then
                echo "[$(date +%F\ %T)] [GPU${gpu}] skip ${name} — all ${requested_clean} already present" | tee -a "$QLOG"
                echo "[$(date +%F\ %T)] [GPU${gpu}] done: ${name}" | tee -a "$QLOG"
                continue
            fi
            if [[ -n "$done_tasks" ]]; then
                echo "[$(date +%F\ %T)] [GPU${gpu}] resume ${name}: skipping done [${done_tasks}], running [${missing}]" | tee -a "$QLOG"
            fi

            extra=()
            if [[ -n "$MMLU_PRO_LIMIT" ]]; then
                extra+=( --mmlu-pro-limit "$MMLU_PRO_LIMIT" --mmlu-pro-seed "$MMLU_PRO_SEED" )
            fi
            bash "$SCRIPT" \
                --model "$model" --protocol "$proto" \
                --gpu "$gpu" --batch-size "$BATCH_SIZE" \
                --output "$out" --tasks "$missing" \
                "${extra[@]}" \
                2>&1 | tee -a "$out/run.log"
            echo "[$(date +%F\ %T)] [GPU${gpu}] done: ${name}" | tee -a "$QLOG"
        done
    ) > "$chain_log" 2>&1 &
    CHAIN_PIDS+=($!)
done

wait "${CHAIN_PIDS[@]}"
echo "[$(date +%F\ %T)] all chains done" | tee -a "$QLOG"

# Per-protocol 3-way comparison (only if we have at least an LLM + VLM-LM pair).
if [[ "$DO_COMPARE" -eq 1 && -f "$PARSE" ]]; then
    for proto in "${PROTO_ARR[@]}"; do
        llm_dir="$OUT_ROOT/llm_${proto}"
        vlm_dir="$OUT_ROOT/vlm_lm_${proto}"
        if [[ ! -d "$llm_dir" || ! -d "$vlm_dir" ]]; then
            continue
        fi
        cmd=(python "$PARSE" --llm "$llm_dir" --vlm "$vlm_dir")
        for entry in "${MODELS[@]}"; do
            name="${entry%%:*}"
            case "$name" in
                llm|vlm_lm) ;;
                *)
                    merged_dir="$OUT_ROOT/${name}_${proto}"
                    [[ -d "$merged_dir" ]] && cmd+=(--merged "${name}:${merged_dir}")
                    ;;
            esac
        done
        out="$OUT_ROOT/comparison_${proto}.txt"
        echo "[$(date +%F\ %T)] compare ${proto} -> $out" | tee -a "$QLOG"
        "${cmd[@]}" 2>&1 | tee "$out"
    done
fi
echo "[$(date +%F\ %T)] matrix complete" | tee -a "$QLOG"
