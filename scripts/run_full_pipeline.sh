#!/usr/bin/env bash
# run_full_pipeline.sh — orchestrator for the analysis matrix reproduction.
#
# IMPORTANT: this script reproduces stages 1-8 (model downloads, analysis,
# IFEval matrix). Stages 9-12 (C3 training) are GATED behind code-level
# blockers documented in TODO.md §5; the script refuses to launch C3
# unless I_KNOW_C3_BLOCKERS=1 is set. See TODO.md for the full list of
# trainer changes needed before C3 can run end-to-end.
#
# Stages this script CAN run end-to-end today:
#   1. Download all models (NOT datasets — those need manual prep, see TODO.md §2)
#   2. Build Qwen3-8B-nothink chat-template overlay
#   3. Extract text backbones from 5 VLMs (Qwen2.5-VL, Qwen3-VL, InternVL3,
#      InternVL3.5, LLaVA-LLaMA3)
#   4. Weight-level analysis (diff geometry, SVD, γ, T, k_proj rows)
#   5. C1 sink-ablation overlays + IFEval
#   6. C2 SAS overlay + IFEval
#   7. E2 random-W perturb + IFEval
#   8. Generalization IFEval matrix (5 pairs)
#
# Stages this script DOES NOT run today (without manual fixes):
#   9. C3 init build (OK once datasets prepped)
#  10-11. C3 Stage 1/2 training (gated — see TODO.md §5 blockers)
#  12. C3 extract LM + IFEval (needs trainer integration first)
#
# Stages can be skipped via env-vars:
#   SKIP_DOWNLOADS=1   skip step 1
#   SKIP_OVERLAY=1     skip step 2
#   SKIP_ANALYSIS=1    skip steps 3-8
#   SKIP_C3=1          skip C3 launch attempt (default behavior anyway —
#                      C3 requires I_KNOW_C3_BLOCKERS=1 to attempt)
#
# Single-stage entry points (set one of these to a non-empty value):
#   ONLY_DOWNLOADS=1   only step 1
#   ONLY_ANALYSIS=1    only steps 3-8 (assumes step 1-2 done)
#   ONLY_C3=1          only attempt C3 (requires I_KNOW_C3_BLOCKERS=1)
#
# Required env-vars:
#   ROOT=/path/to/models               where downloaded models live
#   CKPT_ROOT=/path/to/c3_ckpts        C3 training output (~60 GB)
#   DATA_ROOT=/path/to/data            LLaVA-Pretrain + Mix-665K datasets
#   HF_TOKEN=...                       HuggingFace token (gated models)
#
# Optional:
#   NGPU=2                             #GPUs for C3 training (default 2)
#   ANALYSIS_GPU=0                     which GPU for analysis steps (default 0)
#   EVAL_GPUS=0,1                      GPUs for IFEval matrix (default 0,1)
#   PYTHON=python                      python binary
#
# Hard requirements (must be installed beforehand):
#   - pytorch ≥ 2.4, transformers ≥ 4.45
#   - huggingface_hub, safetensors
#   - lm-eval-harness 0.4.5 with ifeval extra: `pip install "lm_eval[ifeval]==0.4.5"`
#   - deepspeed (for C3)

set -euo pipefail

# ---------------------------------------------------------------------------
# Env defaults
# ---------------------------------------------------------------------------
: "${ROOT:?export ROOT=/path/to/models (where downloaded models live)}"
: "${HF_TOKEN:?export HF_TOKEN with a HF user token for gated repos}"

CKPT_ROOT="${CKPT_ROOT:-$ROOT/c3_ckpts}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
NGPU="${NGPU:-2}"
ANALYSIS_GPU="${ANALYSIS_GPU:-0}"
EVAL_GPUS="${EVAL_GPUS:-0,1}"
PYTHON="${PYTHON:-python}"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# Stage gating
# ---------------------------------------------------------------------------
# Defaults: analysis matrix runs end-to-end; C3 stays OFF because it has
# unresolved blockers (see TODO.md §5). Opt-in via WANT_C3=1.
RUN_DOWNLOADS=1
RUN_OVERLAY=1
RUN_ANALYSIS=1
RUN_C3=0
[[ -n "${WANT_C3:-}" ]] && RUN_C3=1

[[ -n "${SKIP_DOWNLOADS:-}" ]] && RUN_DOWNLOADS=0
[[ -n "${SKIP_OVERLAY:-}"   ]] && RUN_OVERLAY=0
[[ -n "${SKIP_ANALYSIS:-}"  ]] && RUN_ANALYSIS=0
[[ -n "${SKIP_C3:-}"        ]] && RUN_C3=0

if [[ -n "${ONLY_DOWNLOADS:-}" ]]; then
    RUN_DOWNLOADS=1; RUN_OVERLAY=0; RUN_ANALYSIS=0; RUN_C3=0
elif [[ -n "${ONLY_ANALYSIS:-}" ]]; then
    RUN_DOWNLOADS=0; RUN_OVERLAY=0; RUN_ANALYSIS=1; RUN_C3=0
elif [[ -n "${ONLY_C3:-}" ]]; then
    RUN_DOWNLOADS=0; RUN_OVERLAY=0; RUN_ANALYSIS=0; RUN_C3=1
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()    { printf '\n\e[1;36m[%s] %s\e[0m\n' "$(date +%H:%M:%S)" "$*"; }
section(){ printf '\n\e[1;33m===== %s =====\e[0m\n' "$*"; }

hf_download() {
    local repo="$1" dst="$2"
    if [[ -d "$dst" && -n "$(ls "$dst" 2>/dev/null)" ]]; then
        log "skip $repo (already at $dst)"; return
    fi
    log "downloading $repo → $dst"
    HF_TOKEN="$HF_TOKEN" "$PYTHON" -c "
from huggingface_hub import snapshot_download
snapshot_download('$repo', local_dir='$dst', token='$HF_TOKEN', max_workers=4)
"
}

# ---------------------------------------------------------------------------
# Legacy path fix-up (must come AFTER log() is defined)
# ---------------------------------------------------------------------------
# Several analysis scripts (c1_*, c2_*, e2_*) hardcode /131_data/geeho/minsik/.
# If $ROOT differs, try symlink; fall back to sed (analysis scripts are not
# heavily git-edited — sed-modifying them is acceptable for a clean re-run).
LEGACY_ROOT="/131_data/geeho/minsik"
if [[ "$ROOT" != "$LEGACY_ROOT" && ! -e "$LEGACY_ROOT" ]]; then
    if mkdir -p "$(dirname "$LEGACY_ROOT")" 2>/dev/null \
       && ln -sfn "$ROOT" "$LEGACY_ROOT" 2>/dev/null; then
        log "created compatibility symlink $LEGACY_ROOT -> $ROOT"
    else
        log "WARN: cannot create $LEGACY_ROOT symlink; sed-patching analysis script paths"
        for f in analysis/sibling_diff/c1_*.py \
                 analysis/sibling_diff/c2_*.py \
                 analysis/sibling_diff/e2_*.py; do
            [[ -f "$f" ]] && sed -i "s|/131_data/geeho/minsik|${ROOT}|g" "$f"
        done
    fi
fi

# ---------------------------------------------------------------------------
# Stage 1 — Model downloads
# ---------------------------------------------------------------------------
if [[ $RUN_DOWNLOADS -eq 1 ]]; then
    section "STAGE 1: model downloads (≈ 250 GB)"
    mkdir -p "$ROOT"

    hf_download Qwen/Qwen2.5-7B-Instruct        "$ROOT/Qwen2.5-7B-Instruct"
    hf_download Qwen/Qwen2.5-VL-7B-Instruct     "$ROOT/Qwen2.5-VL-7B-Instruct"
    hf_download Qwen/Qwen3-8B                   "$ROOT/Qwen3-8B"
    hf_download Qwen/Qwen3-VL-8B-Instruct       "$ROOT/Qwen3-VL-8B-Instruct"
    hf_download Qwen/Qwen2.5-3B-Instruct        "$ROOT/Qwen2.5-3B-Instruct"
    hf_download Qwen/Qwen2.5-VL-3B-Instruct     "$ROOT/Qwen2.5-VL-3B-Instruct"
    hf_download meta-llama/Meta-Llama-3-8B-Instruct  "$ROOT/Meta-Llama-3-8B-Instruct"
    hf_download lmms-lab/llama3-llava-next-8b   "$ROOT/llama3-llava-next-8b"
    hf_download OpenGVLab/InternVL3-8B          "$ROOT/InternVL3-8B"
    hf_download OpenGVLab/InternVL3_5-8B        "$ROOT/InternVL3_5-8B"
fi

# ---------------------------------------------------------------------------
# Stage 2 — Qwen3-8B-nothink chat-template overlay
# ---------------------------------------------------------------------------
if [[ $RUN_OVERLAY -eq 1 ]]; then
    section "STAGE 2: build Qwen3-8B-nothink overlay"
    OVERLAY="$ROOT/Qwen3-8B-nothink"
    if [[ ! -f "$OVERLAY/chat_template.jinja" ]]; then
        mkdir -p "$OVERLAY"
        SRC="$ROOT/Qwen3-8B"
        for f in "$SRC"/*; do
            base="$(basename "$f")"
            case "$base" in
                chat_template.jinja|tokenizer_config.json) ;;
                *) ln -sfn "$f" "$OVERLAY/$base" ;;
            esac
        done
        # Patch the chat template: force-emit empty <think>...</think> block.
        # Qwen3 default template branches on `enable_thinking`; we hard-code
        # the no-think branch so IFEval format-checkers always see clean output.
        "$PYTHON" - <<EOF
import json, re, shutil
from pathlib import Path

src = Path("$SRC")
dst = Path("$OVERLAY")

# tokenizer_config.json — copy, but set chat_template ref to local jinja
tc = json.load(open(src / "tokenizer_config.json"))
tc["chat_template"] = open(src / "chat_template.jinja").read()
# Inject the no-think override: replace any conditional emit of <think>...</think>
# with always-empty <think>\n\n</think>\n\n on every assistant turn.
t = tc["chat_template"]
# crude but effective: strip the `{% if enable_thinking %}` / endif pair and
# replace any '<think>{{ thinking_content }}</think>' with empty form.
t = re.sub(r"\{%- ?if enable_thinking ?%\}.*?\{%- ?endif ?%\}", "", t, flags=re.S)
t = t.replace("{{ thinking_content }}", "")
tc["chat_template"] = t
json.dump(tc, open(dst / "tokenizer_config.json", "w"), indent=2)
open(dst / "chat_template.jinja", "w").write(t)
print("[overlay] patched chat template")
EOF
    else
        log "skip overlay (already at $OVERLAY)"
    fi
fi

# ---------------------------------------------------------------------------
# Stage 3 — Extract text backbones from 5 VLMs
# ---------------------------------------------------------------------------
if [[ $RUN_ANALYSIS -eq 1 ]]; then
    section "STAGE 3: extract VLM text backbones"
    mkdir -p cache/extracted

    extract_via_lm() {
        local pair="$1" out="$2"
        if [[ -d "$out" && -n "$(ls "$out" 2>/dev/null)" ]]; then
            log "skip extract $pair (exists)"; return
        fi
        CUDA_VISIBLE_DEVICES="$ANALYSIS_GPU" "$PYTHON" -m extraction.extract_lm \
            --pair "$pair" --output "$out"
    }
    extract_via_lm qwen25vl_7b cache/extracted/qwen25vl_7b_lm
    extract_via_lm qwen3vl_8b  cache/extracted/qwen3vl_8b_lm

    # Also build a "no-think" alias for Qwen3-VL-LM (config copied from overlay)
    if [[ ! -d cache/extracted/qwen3vl_8b_lm_nothink ]]; then
        mkdir -p cache/extracted/qwen3vl_8b_lm_nothink
        for f in cache/extracted/qwen3vl_8b_lm/*; do
            ln -sfn "$(realpath "$f")" "cache/extracted/qwen3vl_8b_lm_nothink/$(basename "$f")"
        done
        # Override tokenizer_config + chat_template to no-think variant
        cp "$ROOT/Qwen3-8B-nothink/tokenizer_config.json" cache/extracted/qwen3vl_8b_lm_nothink/
        cp "$ROOT/Qwen3-8B-nothink/chat_template.jinja"   cache/extracted/qwen3vl_8b_lm_nothink/
    fi

    extract_direct() {
        local vlm="$1" out="$2" tok_src="$3" cfg_src="${4:-}"
        if [[ -d "$out" && -n "$(ls "$out" 2>/dev/null)" ]]; then
            log "skip extract_direct $vlm (exists)"; return
        fi
        local extra=""
        [[ -n "$cfg_src" ]] && extra="--config-src $cfg_src"
        CUDA_VISIBLE_DEVICES="$ANALYSIS_GPU" "$PYTHON" \
            analysis/sibling_diff/extract_direct.py \
            --vlm "$vlm" --output "$out" --tokenizer-src "$tok_src" $extra
    }

    # InternVL3-8B uses a custom Qwen2.5-7B variant (vocab_size=151674)
    INTERNVL3_CFG="/tmp/internvl3_cfg_src"
    if [[ ! -f "$INTERNVL3_CFG/config.json" ]]; then
        mkdir -p "$INTERNVL3_CFG"
        "$PYTHON" - <<EOF
import json
src = json.load(open("$ROOT/InternVL3-8B/config.json"))
llm = src.get("llm_config", src.get("text_config", {}))
llm.setdefault("model_type", "qwen2")
json.dump(llm, open("$INTERNVL3_CFG/config.json", "w"), indent=2)
EOF
    fi
    extract_direct "$ROOT/InternVL3-8B" cache/extracted/internvl3_8b_lm \
        "$ROOT/Qwen2.5-7B-Instruct" "$INTERNVL3_CFG"

    # InternVL3.5-8B has intermediate_size=12288 (custom Qwen3 variant)
    INTERNVL35_CFG="/tmp/internvl35_cfg_src"
    if [[ ! -f "$INTERNVL35_CFG/config.json" ]]; then
        mkdir -p "$INTERNVL35_CFG"
        "$PYTHON" - <<EOF
import json
src = json.load(open("$ROOT/InternVL3_5-8B/config.json"))
llm = src.get("llm_config", src.get("text_config", {}))
llm.setdefault("model_type", "qwen3")
json.dump(llm, open("$INTERNVL35_CFG/config.json", "w"), indent=2)
EOF
    fi
    extract_direct "$ROOT/InternVL3_5-8B" cache/extracted/internvl35_8b_lm \
        "$ROOT/Qwen3-8B-nothink" "$INTERNVL35_CFG"

    # LLaVA-LLaMA3 — Llama-3 tokenizer
    extract_direct "$ROOT/llama3-llava-next-8b" cache/extracted/llava_llama3_lm \
        "$ROOT/Meta-Llama-3-8B-Instruct"

# ---------------------------------------------------------------------------
# Stage 4 — Weight-level analysis
# ---------------------------------------------------------------------------
    section "STAGE 4: weight-level analysis (diff geometry, SVD, γ, T)"
    cd "$REPO/analysis/sibling_diff"

    CUDA_VISIBLE_DEVICES="$ANALYSIS_GPU" "$PYTHON" diff_geometry.py
    CUDA_VISIBLE_DEVICES="$ANALYSIS_GPU" "$PYTHON" svd_and_gamma.py
    CUDA_VISIBLE_DEVICES="$ANALYSIS_GPU" "$PYTHON" sink_in_projections.py
    CUDA_VISIBLE_DEVICES="$ANALYSIS_GPU" "$PYTHON" measure_sink_T.py
    cd "$REPO"

# ---------------------------------------------------------------------------
# Stage 5 — C1 sink-ablation overlays + IFEval
# ---------------------------------------------------------------------------
    section "STAGE 5: C1 sink-ablation overlays"
    CUDA_VISIBLE_DEVICES="$ANALYSIS_GPU" "$PYTHON" analysis/sibling_diff/c1_kill_sink_qwen3.py
    CUDA_VISIBLE_DEVICES="$ANALYSIS_GPU" "$PYTHON" analysis/sibling_diff/c1_ablate_per_norm.py

    log "C1 IFEval matrix"
    bash evaluation/text/run_eval_matrix.sh \
        --models llm:"$ROOT/Qwen3-8B-nothink" \
                 c1_all:"$ROOT/Qwen3-8B-nosink-nothink" \
                 c1_lnnorm:"$ROOT/Qwen3-8B-nosink-lnnorm-nothink" \
                 c1_qnorm:"$ROOT/Qwen3-8B-nosink-qnorm-nothink" \
                 c1_knorm:"$ROOT/Qwen3-8B-nosink-knorm-nothink" \
        --protocols instruct --gpus "$EVAL_GPUS" --tasks ifeval \
        --output-root eval_results/c1_ablation

# ---------------------------------------------------------------------------
# Stage 6 — C2 SAS + IFEval
# ---------------------------------------------------------------------------
    section "STAGE 6: C2 SAS overlay + IFEval"
    CUDA_VISIBLE_DEVICES="$ANALYSIS_GPU" "$PYTHON" analysis/sibling_diff/c2_sas_qwen25_recover.py
    bash evaluation/text/run_eval_matrix.sh \
        --models llm:"$ROOT/Qwen2.5-7B-Instruct" \
                 vlm_lm:cache/extracted/qwen25vl_7b_lm \
                 sas:cache/extracted/qwen25vl_7b_lm_sas \
        --protocols instruct --gpus "$EVAL_GPUS" --tasks ifeval \
        --output-root eval_results/qwen25_sas_test

# ---------------------------------------------------------------------------
# Stage 7 — E2 random-W + IFEval
# ---------------------------------------------------------------------------
    section "STAGE 7: E2 random-W perturbation"
    CUDA_VISIBLE_DEVICES="$ANALYSIS_GPU" "$PYTHON" analysis/sibling_diff/e2_random_w_perturb.py \
        --seed 0 --rel-scale 1.0
    bash evaluation/text/run_eval_matrix.sh \
        --models e2_qwen25_pert:"$ROOT/Qwen2.5-7B-Instruct-Wperturb-s0-r1.0" \
        --protocols instruct --gpus "$EVAL_GPUS" --tasks ifeval \
        --output-root eval_results/generalization

# ---------------------------------------------------------------------------
# Stage 8 — Generalization IFEval matrix (5 pairs)
# ---------------------------------------------------------------------------
    section "STAGE 8: generalization IFEval (5 pairs)"
    bash evaluation/text/run_eval_matrix.sh \
        --models llama3_llm:"$ROOT/Meta-Llama-3-8B-Instruct" \
                 llava_llama3_vlm:cache/extracted/llava_llama3_lm \
                 internvl3_vlm:cache/extracted/internvl3_8b_lm \
                 internvl35_vlm:cache/extracted/internvl35_8b_lm \
                 qwen25_llm:"$ROOT/Qwen2.5-7B-Instruct" \
                 qwen25vl_lm:cache/extracted/qwen25vl_7b_lm \
                 qwen3_llm:"$ROOT/Qwen3-8B-nothink" \
                 qwen3vl_lm:cache/extracted/qwen3vl_8b_lm_nothink \
        --protocols instruct --gpus "$EVAL_GPUS" --tasks ifeval \
        --output-root eval_results/generalization
fi  # RUN_ANALYSIS

# ---------------------------------------------------------------------------
# Stage 9-12 — C3 training (GATED; see TODO.md §5 for blockers)
# ---------------------------------------------------------------------------
if [[ $RUN_C3 -eq 1 ]]; then
    section "STAGE 9-12: C3 architectural causality training"

    if [[ -z "${I_KNOW_C3_BLOCKERS:-}" ]]; then
        cat >&2 <<EOM

C3 training is gated. The following blockers still need resolution
(see TODO.md §5 for details). Stage 2 pretrained-override (§5.1) has
already been fixed; what remains:

  • §5.2 The post-training extraction step (extraction/extract_lm.py)
        does not know about inject_qknorm. Extracting from a qknorm-
        trained VLM will drop the q_norm/k_norm parameters and
        evaluate as a vanilla Qwen2 LM. Fix: add --inject-qknorm flag
        to extract_lm CLI; save q_norm/k_norm.weight tensors and a
        minimal load script.

  • §5.3 qknorm γ continuity stage1->stage2: load_qknorm_state_if_present
        currently loads from \`pretrained\` (now the init dir, no γ).
        Add an explicit qknorm_state_dir config field or pass via CLI.

  • §5.4 training/data/__init__.py eagerly imports lmdb-dependent
        modules; DeepSpeed Zero-2 is in the YAMLs. Install before
        launching: pip install lmdb deepspeed

  • §5.5 transformers 4.45+ DDP-safe patch in
        training/models/qwen25vl.py calls .to() on a torch.split()
        result (tuple in newer versions). Smoke-test with 1 step first.

  • §5.6 Datasets are manual (see TODO.md §2):
            \$DATA_ROOT/LLaVA-Pretrain/llava_pretrain_558k.jsonl + images/
            \$DATA_ROOT/LLaVA-Instruct/llava_v1_5_mix665k.jsonl + image_root/
        This script downloads MODELS only.

If you have addressed these and want to attempt the C3 run:
    WANT_C3=1 I_KNOW_C3_BLOCKERS=1 bash scripts/run_full_pipeline.sh ...

EOM
        exit 3
    fi

    : "${DATA_ROOT:?export DATA_ROOT (LLaVA-Pretrain + Mix-665K). See TODO.md §2.}"

    # Sanity checks for required data files
    if [[ ! -f "$DATA_ROOT/LLaVA-Pretrain/llava_pretrain_558k.jsonl" ]]; then
        echo "ERROR: missing $DATA_ROOT/LLaVA-Pretrain/llava_pretrain_558k.jsonl" >&2
        echo "       see TODO.md §2 (Dataset downloads) for prep instructions" >&2
        exit 2
    fi
    if [[ ! -f "$DATA_ROOT/LLaVA-Instruct/llava_v1_5_mix665k.jsonl" ]]; then
        echo "ERROR: missing $DATA_ROOT/LLaVA-Instruct/llava_v1_5_mix665k.jsonl" >&2
        echo "       see TODO.md §2" >&2
        exit 2
    fi

    # Write derived manifests to a $CKPT_ROOT/configs_derived/ dir so we
    # don't mutate the git-tracked YAML/JSON.
    derived="$CKPT_ROOT/configs_derived/data"
    mkdir -p "$derived"
    "$PYTHON" - <<EOF
import json
from pathlib import Path
for f in ["training/configs/data/align_llava_pretrain.json",
          "training/configs/data/stage2_llava_mix665k.json"]:
    d = json.load(open(f))
    for s in d["sources"]:
        s["jsonl"]      = s["jsonl"].replace("/131_data/geeho/minsik/data", "$DATA_ROOT")
        s["image_root"] = s["image_root"].replace("/131_data/geeho/minsik/data", "$DATA_ROOT")
    out = Path("$derived") / Path(f).name
    out.write_text(json.dumps(d, indent=2))
    print(f"  wrote derived manifest: {out}")
EOF
    # NOTE: the YAML configs reference these manifest paths *relative to repo
    # root* (training/configs/data/*.json). Symlink derived versions in place
    # so the training run picks them up without mutating tracked files.
    for f in align_llava_pretrain.json stage2_llava_mix665k.json; do
        ln -sfn "$derived/$f" "training/configs/data/$f.derived"
    done
    echo "  (data manifests prepared under $derived — symlinks placed alongside tracked files)"

    # The existing C3 driver handles init → align → stage2 → extract → IFEval.
    CKPT_ROOT="$CKPT_ROOT" \
        VL_CKPT="$ROOT/Qwen2.5-VL-3B-Instruct" \
        LM_CKPT="$ROOT/Qwen2.5-3B-Instruct" \
        NGPU="$NGPU" \
        EVAL_GPUS="$EVAL_GPUS" \
        bash training/scripts/run_c3_3b.sh
fi

section "ALL DONE"
echo "Outputs:"
if [[ $RUN_ANALYSIS -eq 1 ]]; then
    echo "  Analysis CSVs   : analysis/sibling_diff/*.csv"
    echo "  Analysis figures: analysis/sibling_diff/figures/"
    echo "  Eval results    : eval_results/"
fi
if [[ $RUN_C3 -eq 1 ]]; then
    echo "  C3 ckpts        : $CKPT_ROOT/c3/3b/{vanilla,qknorm}_stage2/"
    echo "  C3 eval         : eval_results/c3_3b/"
    echo
    echo "Next: inspect eval_results/c3_3b/ and decide per TODO.md §7"
elif [[ $RUN_ANALYSIS -eq 1 ]]; then
    echo
    echo "Analysis matrix done. To launch C3 training, resolve the §5"
    echo "blockers in TODO.md and re-run with:"
    echo "  WANT_C3=1 I_KNOW_C3_BLOCKERS=1 ONLY_C3=1 bash scripts/run_full_pipeline.sh"
fi
