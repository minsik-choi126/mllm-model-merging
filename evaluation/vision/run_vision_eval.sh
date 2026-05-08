#!/bin/bash
# Vision evaluation via VLMEvalKit.
#
# Usage:
#   bash run_vision_eval.sh --model <hf_id_or_path> [options]
#
# Options:
#   --model      VLM model path or HF ID (required)
#   --gpu        GPU index (default: 0)
#   --tasks      Benchmark group or comma-separated names (default: core)
#                Groups: core | extended | full | <bench1,bench2,...>
#   --output     Output directory (default: eval_results/vision/<model_slug>_<group>)
#
# Environment:
#   VLMEVAL_DIR   Path to VLMEvalKit checkout (required)
#   PYTHON_BIN    Python interpreter to use (default: `python3` on PATH)

# ── Benchmark groups ──────────────────────────────────────────────────────────
TASKS_CORE="MMBench_DEV_EN_V11,SEEDBench_IMG,POPE,MMMU_DEV_VAL,MathVista_MINI,RealWorldQA"
TASKS_EXTENDED="${TASKS_CORE},OCRBench,ChartQA_TEST,AI2D_TEST,HallusionBench,MMVet"
TASKS_FULL="${TASKS_EXTENDED},DocVQA_VAL,SEEDBench2_Plus"
#
# Domain coverage:
#   General perception : MMBench_DEV_EN_V11, SEEDBench_IMG
#   Knowledge/reasoning: MMMU_DEV_VAL, MathVista_MINI, RealWorldQA
#   Hallucination      : POPE, HallusionBench
#   OCR/document       : OCRBench, ChartQA_TEST, DocVQA_VAL
#   Diagram            : AI2D_TEST
#   Open-ended         : MMVet
# ─────────────────────────────────────────────────────────────────────────────

set -e

if [[ -z "${VLMEVAL_DIR:-}" ]]; then
    echo "Error: VLMEVAL_DIR is not set." >&2
    echo "Set it to your VLMEvalKit checkout, e.g.:" >&2
    echo "  export VLMEVAL_DIR=/path/to/VLMEvalKit" >&2
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL=""
GPU=0
TASK_ARG="core"
OUTPUT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)   MODEL="$2";    shift 2 ;;
        --gpu)     GPU="$2";      shift 2 ;;
        --tasks)   TASK_ARG="$2"; shift 2 ;;
        --output)  OUTPUT="$2";   shift 2 ;;
        -h|--help)
            sed -n '2,18p' "$0"
            exit 0
            ;;
        *) echo "Unknown: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "Error: --model required" >&2
    exit 1
fi

case "$TASK_ARG" in
    core)     TASKS="$TASKS_CORE"     ;;
    extended) TASKS="$TASKS_EXTENDED" ;;
    full)     TASKS="$TASKS_FULL"     ;;
    *)        TASKS="$TASK_ARG"       ;;
esac

if [[ -z "$OUTPUT" ]]; then
    MODEL_SLUG=$(echo "$MODEL" | tr '/' '_' | tr ' ' '_')
    OUTPUT="eval_results/vision/${MODEL_SLUG}_${TASK_ARG}"
fi

TASKS_SPACE=$(echo "$TASKS" | tr ',' ' ')

echo "======================================================"
echo "Vision Eval (VLMEvalKit)"
echo "Model:    $MODEL"
echo "Tasks:    $TASKS"
echo "GPU:      $GPU"
echo "Output:   $OUTPUT"
echo "VLMEval:  $VLMEVAL_DIR"
echo "Python:   $PYTHON_BIN"
echo "======================================================"

mkdir -p "$OUTPUT"

CUDA_VISIBLE_DEVICES=$GPU "$PYTHON_BIN" "$VLMEVAL_DIR/run.py" \
    --model "$MODEL" \
    --data $TASKS_SPACE \
    --work-dir "$OUTPUT"
