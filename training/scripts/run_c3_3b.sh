#!/usr/bin/env bash
# C3 — Architectural causality experiment at 3B (LLaVA recipe; Stage 1.5 SKIPPED)
#
# Trains two VLMs from the same composed init:
#   A1  vanilla       — no QK-norm injection
#   A2  qknorm        — q_norm/k_norm modules injected (γ=1 at init)
#
# Both share Stage 1 → Stage 2 pipeline. We omit Stage 1.5 (image-recap) per
# the advisor's recommendation to keep the run tractable on a small node.
#
# Usage:
#   export CKPT_ROOT=/path/to/ckpts/c3
#   bash training/scripts/run_c3_3b.sh
#   # or run only one variant:
#   VARIANT=vanilla bash training/scripts/run_c3_3b.sh
#   VARIANT=qknorm  bash training/scripts/run_c3_3b.sh
#
# Env-var configuration:
#   NGPU=N             number of GPUs for training (default 2). The script
#                      auto-rescales gradient_accumulation_steps in the
#                      YAML configs so effective batch size stays at 128:
#                      grad_accum = round(128 / (per_device_bs * NGPU))
#   CUDA_VISIBLE_DEVICES=0,1,2,3
#                      which physical GPUs torchrun should see (overrides
#                      auto-discovery). Must list at least NGPU devices.
#   EVAL_GPUS=0,1      GPU indices for the post-training IFEval matrix
#                      (default: 0,...,NGPU-1 of CUDA_VISIBLE_DEVICES)
#   VARIANT=vanilla|qknorm|both
#                      which variant(s) to run (default: both)
#   VL_CKPT, LM_CKPT   paths to the vision-tower-source VLM and text-LM-source
#                      LLM checkpoints (defaults to /131_data/geeho/minsik/...).

set -euo pipefail

: "${CKPT_ROOT:?export CKPT_ROOT to a work dir for checkpoints (~30 GB)}"
NGPU="${NGPU:-2}"
VARIANT="${VARIANT:-both}"           # vanilla | qknorm | both
VL_CKPT="${VL_CKPT:-/131_data/geeho/minsik/Qwen2.5-VL-3B-Instruct}"
LM_CKPT="${LM_CKPT:-/131_data/geeho/minsik/Qwen2.5-3B-Instruct}"

# Default EVAL_GPUS to first NGPU indices if not set
if [[ -z "${EVAL_GPUS:-}" ]]; then
    eval_idx=()
    for ((i=0; i<NGPU; i++)); do eval_idx+=("$i"); done
    EVAL_GPUS="$(IFS=,; echo "${eval_idx[*]}")"
fi
echo "[c3_3b] NGPU=$NGPU  EVAL_GPUS=$EVAL_GPUS  VARIANT=$VARIANT"

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

mkdir -p "$CKPT_ROOT"
if [[ ! -e "$REPO/ckpts" ]]; then
    ln -sfn "$CKPT_ROOT" "$REPO/ckpts"
fi

# Path must match training/configs/_base_/model_qwen25_3b_init.yaml's
# `model.pretrained` field, which resolves via the ckpts -> $CKPT_ROOT symlink:
# YAML says ckpts/c3/init/... so on disk that's $CKPT_ROOT/c3/init/...
INIT_DIR="$CKPT_ROOT/c3/init/qwen25vl_3b_text_lm"

# ---------------------------------------------------------------------------
# Step 0 — generate derived configs (rescaled grad_accum for actual GPU count)
# ---------------------------------------------------------------------------
# Configs target effective batch size 128 = per_device_bs * NGPU * grad_accum.
# Recipe defaults assume NGPU=2; we rescale grad_accum into a *derived* config
# under $CKPT_ROOT/configs_derived/3b/ rather than mutating git-tracked files.
# The derived dir uses absolute repo paths for `defaults:` so the load_yaml
# inheritance resolver doesn't have to climb out of $CKPT_ROOT.
export DERIVED_CFG_DIR="$CKPT_ROOT/configs_derived/3b"
mkdir -p "$DERIVED_CFG_DIR"
REPO_ABS="$REPO" python - <<'EOF'
import os, re, shutil
from pathlib import Path
TARGET_EFF_BS = 128
NGPU = int(os.environ.get("NGPU", "2"))
REPO = Path(os.environ["REPO_ABS"]).resolve()
BASE_ABS = REPO / "training" / "configs" / "_base_"
src_dir = REPO / "training" / "configs" / "3b"
dst_dir = Path(os.environ["DERIVED_CFG_DIR"]).resolve()
for src in src_dir.glob("c3_*.yaml"):
    txt = src.read_text()
    m_bs = re.search(r"per_device_train_batch_size:\s*(\d+)", txt)
    m_ga = re.search(r"gradient_accumulation_steps:\s*(\d+)", txt)
    if m_bs and m_ga:
        bs = int(m_bs.group(1))
        desired_ga = max(1, round(TARGET_EFF_BS / (bs * NGPU)))
        cur_ga = int(m_ga.group(1))
        if cur_ga != desired_ga:
            txt = re.sub(r"(gradient_accumulation_steps:\s*)\d+",
                         rf"\g<1>{desired_ga}", txt)
            print(f"  [derived] {src.name}: ga {cur_ga} -> {desired_ga} "
                  f"(bs={bs} * ngpu={NGPU} * ga={desired_ga} "
                  f"= {bs*NGPU*desired_ga})")
    # Rewrite `- ../_base_/<file>` -> absolute repo path so load_yaml
    # resolves correctly from anywhere on disk.
    txt = re.sub(r"(- )\.\./_base_/(\S+)",
                 lambda m: f"{m.group(1)}{BASE_ABS}/{m.group(2)}", txt)
    # If a derived data manifest exists in the same dir, point data.root
    # at it instead of the git-tracked original.
    for orig, derived_name in [
        ("training/configs/data/align_llava_pretrain.json",
         "align_llava_pretrain.json"),
        ("training/configs/data/stage2_llava_mix665k.json",
         "stage2_llava_mix665k.json"),
    ]:
        derived_path = dst_dir.parent / "data" / derived_name
        if derived_path.exists():
            txt = txt.replace(orig, str(derived_path.resolve()))
    (dst_dir / src.name).write_text(txt)
EOF

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
    local cfg="$DERIVED_CFG_DIR/c3_${variant}_${stage}.yaml"
    # Use a per-stage marker — Stage 1 only saves mm_projector.bin, so
    # don't expect config.json.
    local out_dir="ckpts/c3/3b/${variant}_${stage}"
    if [[ ( "$stage" == "align"  && -f "$out_dir/mm_projector.bin" ) ||
          ( "$stage" == "stage2" && -f "$out_dir/config.json"      ) ]]; then
        echo "[$variant/$stage] already done — skipping"
        return
    fi
    echo "===== $variant · $stage ====="
    if [[ "$stage" == "stage2" ]]; then
        # Stage 2 needs the Stage 1 projector (and qknorm γ if applicable).
        # See TODO.md §5 #1 — Stage 2 YAML currently overrides `pretrained:`
        # to the Stage 1 dir, which is incorrect; that override should be
        # removed before the run, then --pretrain-projector applies the
        # projector overlay from Stage 1.
        torchrun --standalone --nproc_per_node="$NGPU" -m training.train.cli \
            --config "$cfg" \
            --pretrain-projector "ckpts/c3/3b/${variant}_align"
    else
        torchrun --standalone --nproc_per_node="$NGPU" -m training.train.cli \
            --config "$cfg"
    fi
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
        --gpus "$EVAL_GPUS" \
        --tasks ifeval \
        --output-root eval_results/c3_3b
else
    echo "[WARN] lm_eval not on PATH — skipping IFEval matrix"
fi
echo "===== c3_3b DONE ====="
