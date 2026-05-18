#!/usr/bin/env bash
# C3 — Architectural causality experiment at 3B (LLaVA recipe; Stage 1.5 SKIPPED)
#
# Trains two VLMs from the same composed init:
#   A1  vanilla       — no QK-norm injection
#   A2  qknorm        — q_norm/k_norm modules injected (γ=1 at init)
#
# Both share Stage 1 → Stage 2 pipeline. We omit Stage 1.5 (image-recap) per
# the advisor's recommendation to keep this on a 2-GPU node tractable.
#
# Usage:
#   export CKPT_ROOT=/path/to/ckpts/c3
#   bash training/scripts/run_c3_3b.sh
#   # or run only one variant:
#   VARIANT=vanilla bash training/scripts/run_c3_3b.sh
#   VARIANT=qknorm  bash training/scripts/run_c3_3b.sh

set -euo pipefail

: "${CKPT_ROOT:?export CKPT_ROOT to a work dir for checkpoints (~30 GB)}"
NGPU="${NGPU:-2}"
VARIANT="${VARIANT:-both}"           # vanilla | qknorm | both
VL_CKPT="${VL_CKPT:-/131_data/geeho/minsik/Qwen2.5-VL-3B-Instruct}"
LM_CKPT="${LM_CKPT:-/131_data/geeho/minsik/Qwen2.5-3B-Instruct}"

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

mkdir -p "$CKPT_ROOT"
if [[ ! -e "$REPO/ckpts" ]]; then
    ln -sfn "$CKPT_ROOT" "$REPO/ckpts"
fi

INIT_DIR="$CKPT_ROOT/init/qwen25vl_3b_text_lm"

# ---------------------------------------------------------------------------
# Step 1 — compose the from-scratch init
# ---------------------------------------------------------------------------
if [[ ! -f "$INIT_DIR/config.json" ]]; then
    echo "===== STEP 1: composing from-scratch VLM init ====="
    python -m training.scripts.build_init_from_pretrained \
        --vl-ckpt "$VL_CKPT" --lm-ckpt "$LM_CKPT" \
        --output "$INIT_DIR" --dtype bfloat16 --seed 0
else
    echo "[INIT] already exists at $INIT_DIR — skipping"
fi

# ---------------------------------------------------------------------------
# Step 2 — Stage 1 (Align) for each variant
# ---------------------------------------------------------------------------
run_stage() {
    local variant="$1" stage="$2"  # variant ∈ {vanilla,qknorm}, stage ∈ {align,stage2}
    local cfg="training/configs/3b/c3_${variant}_${stage}.yaml"
    local out_marker="ckpts/c3/3b/${variant}_${stage}/config.json"
    if [[ -f "$out_marker" ]]; then
        echo "[$variant/$stage] already done — skipping"
        return
    fi
    echo "===== $variant · $stage ====="
    torchrun --standalone --nproc_per_node="$NGPU" -m training.train.cli \
        --config "$cfg"
}

variants=()
case "$VARIANT" in
    vanilla)  variants=(vanilla) ;;
    qknorm)   variants=(qknorm)  ;;
    both)     variants=(vanilla qknorm) ;;
    *) echo "VARIANT must be vanilla | qknorm | both"; exit 2 ;;
esac

for v in "${variants[@]}"; do
    run_stage "$v" align
    run_stage "$v" stage2
done

# ---------------------------------------------------------------------------
# Step 3 — IFEval on the extracted text backbones
# ---------------------------------------------------------------------------
echo "===== Step 3: IFEval on extracted LM backbones ====="
for v in "${variants[@]}"; do
    src="ckpts/c3/3b/${v}_stage2"
    dst="cache/extracted/c3_3b_${v}_lm"
    if [[ ! -f "$dst/model.safetensors" && ! -f "$dst/model-00001-of-*.safetensors" ]]; then
        python -m extraction.extract_lm \
            --vlm "$src" --llm-template "$LM_CKPT" \
            --output "$dst" --dtype bfloat16
    fi
done

if command -v lm_eval >/dev/null 2>&1; then
    export LM_EVAL_BIN="${LM_EVAL_BIN:-$(command -v lm_eval)}"
    models=( "qwen25_3b_base:${LM_CKPT}" )
    for v in "${variants[@]}"; do
        models+=( "c3_${v}_vlm_lm:cache/extracted/c3_3b_${v}_lm" )
    done
    bash evaluation/text/run_eval_matrix.sh \
        --models "${models[@]}" \
        --protocols instruct \
        --gpus 0,1 \
        --tasks ifeval \
        --output-root eval_results/c3_3b
else
    echo "[WARN] lm_eval not on PATH — skipping IFEval matrix"
fi
echo "===== c3_3b DONE ====="
