# TODO — restart from a clean node

The current node will be wiped (models + data deleted). This file is the
**resumption guide** for the next node.

[`scripts/run_full_pipeline.sh`](scripts/run_full_pipeline.sh)
**automates the analysis matrix** (stages 1–8: model downloads →
text-backbone extraction → weight analysis → C1/C2/E2 ablations →
5-pair IFEval matrix). **C3 training (stages 9–12) is opt-in and
gated** behind `WANT_C3=1` + `I_KNOW_C3_BLOCKERS=1` because it still
has unresolved blockers documented in §5 below. Datasets are NOT
auto-downloaded; see §2 for manual prep.

This file lists every step so a human can inspect / interrupt /
re-run individual stages independent of the orchestrator.

For details on what each measurement *means*, see
[`analysis/sibling_diff/README.md`](analysis/sibling_diff/README.md).

---

## 0. Prerequisites

- ≥ 2× A6000 / A100 (48 GB+ each) for C3 training. More GPUs are fine:
  `NGPU=4 bash scripts/run_full_pipeline.sh` runs with 4 GPUs and the
  driver auto-rescales `gradient_accumulation_steps` so effective batch
  stays at 128. Set `CUDA_VISIBLE_DEVICES=0,2,4,6` to pick specific GPUs.
- 1× any 24 GB+ GPU for analysis / IFEval matrix
- ~500 GB free disk (models 250 GB, datasets 100 GB, ckpts 60 GB, work 90 GB)
- Conda / pip env with CUDA 12.1+; PyTorch ≥ 2.4
- HuggingFace token at `$HF_TOKEN` (gated models: Llama-3, Qwen)
- GitHub token at `$GH_TOKEN` (for `gh` / private repo access; keep secret)

```bash
export ROOT=/your/work/dir            # e.g. /131_data/geeho/minsik (legacy default)
export HF_TOKEN=$(cat /path/to/hf_token)
export GH_TOKEN=$(cat /path/to/gh_token)
export NGPU=4                          # optional: number of training GPUs (default 2)
export EVAL_GPUS=0,1,2,3               # optional: GPUs for IFEval matrix (default 0,1)
```

---

## 1. Model downloads (≈ 250 GB)

Required for the analysis matrix + C3:

| Model | HF repo | Size | Used for |
|---|---|---:|---|
| Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | 15 GB | LLM base for sibling pair |
| Qwen2.5-VL-7B-Instruct | `Qwen/Qwen2.5-VL-7B-Instruct` | 16 GB | sibling VLM |
| Qwen3-8B | `Qwen/Qwen3-8B` | 17 GB | LLM base for sibling pair |
| Qwen3-VL-8B-Instruct | `Qwen/Qwen3-VL-8B-Instruct` | 18 GB | sibling VLM |
| Qwen2.5-3B-Instruct | `Qwen/Qwen2.5-3B-Instruct` | 6 GB | C3 text LM source |
| Qwen2.5-VL-3B-Instruct | `Qwen/Qwen2.5-VL-3B-Instruct` | 7 GB | C3 vision-tower source |
| Meta-Llama-3-8B-Instruct | `meta-llama/Meta-Llama-3-8B-Instruct` | 16 GB | gen test LLM |
| llama3-llava-next-8b | `lmms-lab/llama3-llava-next-8b` | 16 GB | gen test VLM |
| InternVL3-8B | `OpenGVLab/InternVL3-8B` | 18 GB | gen test VLM (Qwen2.5 base) |
| InternVL3_5-8B | `OpenGVLab/InternVL3_5-8B` | 18 GB | gen test VLM (Qwen3 base) |

Auto-download: stage 1 of `run_full_pipeline.sh`.

Then build the **no-thinking overlay** for Qwen3-8B (see §2.5 of analysis
README): patches the chat template to always emit empty `<think>` block
so IFEval format-checkers work. The script does this via symlinks
+ overridden `chat_template.jinja`.

---

## 2. Dataset downloads (~100 GB)

For C3 LLaVA training only. Skip if you're only running the analysis
matrix.

### Stage 1 (Align) — LLaVA-Pretrain-558K (~12 GB)
- HF: `liuhaotian/LLaVA-Pretrain`
- Convert `blip_laion_cc_sbu_558k.json` → JSONL (one JSON per line):
  ```
  python -c "import json; r=json.load(open('blip_laion_cc_sbu_558k.json')); open('llava_pretrain_558k.jsonl','w').writelines(json.dumps(x)+'\n' for x in r)"
  ```
- Unzip `images.zip` into `images/`.
- Edit [`training/configs/data/align_llava_pretrain.json`](training/configs/data/align_llava_pretrain.json) paths.

### Stage 2 — LLaVA-1.5-mix665k (~70 GB images)
- `llava_v1_5_mix665k.json` from `liuhaotian/LLaVA-Instruct-150K` →
  convert to JSONL same way.
- Image sources (~70 GB total):
  - COCO train2017 (19 GB) — `http://images.cocodataset.org/zips/train2017.zip`
  - GQA (21 GB) — `https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip`
  - Visual Genome (17 GB) — VG_100K + VG_100K_2 parts
  - OCR-VQA (4 GB) — `https://ocr-vqa.github.io/`
  - TextVQA (8 GB) — `https://textvqa.org/dataset/`
- Layout: unify under `image_root` with sub-dirs `coco/train2017/`,
  `gqa/images/`, `vg/VG_100K(_2)/`, `ocr_vqa/images/`, `textvqa/train_images/`.
  (Paths inside JSON are *relative* to image_root.)
- Edit [`training/configs/data/stage2_llava_mix665k.json`](training/configs/data/stage2_llava_mix665k.json) paths.

---

## 3. Run order

```bash
bash scripts/run_full_pipeline.sh
```

The script chains:

| Stage | Wall-clock (2 GPU) | Output |
|---|---:|---|
| 1. Model downloads | ~2 h network | `$ROOT/{model_dirs}` |
| 2. Build Qwen3-8B-nothink overlay | <1 min | `$ROOT/Qwen3-8B-nothink` |
| 3. Extract text backbones from 5 VLMs | ~10 min | `cache/extracted/*_lm` |
| 4. Run analysis (diff geometry, SVD, γ, T, k_proj rows) | ~30 min | `analysis/sibling_diff/*.csv`, `figures/*.png` |
| 5. C1 sink-ablation overlays + IFEval | ~3 h | `eval_results/c1_*` |
| 6. C2 SAS overlay + IFEval | ~1 h | `eval_results/qwen25_sas_test` |
| 7. E2 random-W perturb + IFEval | ~1 h | `eval_results/generalization/e2_*` |
| 8. Generalization IFEval matrix (3 VLMs + 1 LLM base) | ~4 h | `eval_results/generalization/*` |
| 9. C3 dataset prep (Stage 1 + Stage 2) | ~3 h disk I/O | `$DATA_ROOT/LLaVA-*` |
| 10. C3 init build (compose Qwen2.5-VL vision + Qwen2.5 text LM) | ~10 min | `$CKPT_ROOT/init/*` |
| 11. C3 Stage 1 align (×2 variants) | ~16 h total | `$CKPT_ROOT/3b/{vanilla,qknorm}_align` |
| 12. C3 Stage 2 instruct (×2 variants) | ~80 h total | `$CKPT_ROOT/3b/{vanilla,qknorm}_stage2` |
| 13. Extract LM from C3 VLMs + IFEval | ~30 min | `eval_results/c3_3b/` |

Steps 4–8 (analysis) take ~10 h total; steps 11–12 (C3 training) take
~4 days on 2 GPUs (≈ 2 days on 4 GPUs, ≈ 1 day on 8 GPUs — wall-clock
scales near-linearly with NGPU since the auto grad-accum rescale keeps
effective batch constant). Stages 3–8 are independent of 9–13 so they
can run in parallel.

---

## 4. Per-stage manual entry points

Default behavior: stages 1–8 (model downloads + analysis + IFEval
matrix) run end-to-end. C3 training is OFF by default and only runs
when both `WANT_C3=1` and `I_KNOW_C3_BLOCKERS=1` are set (see §5).

```bash
# default: stages 1-8 (analysis matrix)
bash scripts/run_full_pipeline.sh

# C3 training only — once §5 blockers are resolved
WANT_C3=1 I_KNOW_C3_BLOCKERS=1 \
    SKIP_DOWNLOADS=1 SKIP_ANALYSIS=1 \
    bash scripts/run_full_pipeline.sh

# just download
ONLY_DOWNLOADS=1 bash scripts/run_full_pipeline.sh

# just analysis matrix
ONLY_ANALYSIS=1 bash scripts/run_full_pipeline.sh

# just C3 training (assumes models + datasets in place + §5 fixed)
WANT_C3=1 I_KNOW_C3_BLOCKERS=1 ONLY_C3=1 \
    bash scripts/run_full_pipeline.sh

# C3 with custom GPU layout
NGPU=4 EVAL_GPUS=0,1,2,3 \
    WANT_C3=1 I_KNOW_C3_BLOCKERS=1 ONLY_C3=1 \
    bash scripts/run_full_pipeline.sh

# C3 driver alone (bypassing the wrapper)
NGPU=4 EVAL_GPUS=0,1,2,3 CKPT_ROOT=$ROOT/c3_ckpts \
    bash training/scripts/run_c3_3b.sh
```

The C3 driver (`training/scripts/run_c3_3b.sh`) reads `NGPU`, auto-rescales
`gradient_accumulation_steps` in `training/configs/3b/c3_*.yaml` so the
effective batch size stays at 128 regardless of GPU count, then runs
`torchrun --nproc_per_node=$NGPU`. It also handles the post-training
eval (extract LM backbones → IFEval on `Qwen2.5-3B-Instruct` baseline +
both C3 variants) automatically, using `EVAL_GPUS` for the eval matrix.

---

## 5. Known blockers — C3 training cannot launch end-to-end today

These are real code-level issues caught in code-review **before** the
node was wiped. Each must be resolved before `run_full_pipeline.sh`
will run C3 to completion. The script refuses to attempt C3 unless
`I_KNOW_C3_BLOCKERS=1`.

### 5.1 Stage 2 YAML `pretrained:` override — FIXED ✓
Previously: `training/configs/3b/c3_{vanilla,qknorm}_stage2.yaml` set
`model.pretrained: ckpts/c3/3b/{variant}_align` (Stage 1 output), but
Stage 1 only saves `mm_projector.bin` (see [`training/train/stage1_trainer.py:79`](training/train/stage1_trainer.py#L79))
— no config.json, no model weights, so the Stage 2 `from_pretrained`
call would fail before `--pretrain-projector` could apply the overlay.

**Resolution**: dropped the `pretrained:` line from both stage2 YAMLs.
They now inherit `pretrained: ckpts/c3/init/qwen25vl_3b_text_lm` from
the init base. `run_c3_3b.sh` passes
`--pretrain-projector ckpts/c3/3b/<variant>_align` to apply the Stage 1
projector overlay (see [`training/train/cli.py:28`](training/train/cli.py#L28) and
[`training/train/stage2_trainer.py:38`](training/train/stage2_trainer.py#L38)).

### 5.2 QK-norm γ is lost at post-training extraction
`extraction/extract_lm.py` produces a generic Qwen2 text-only LM. For
the qknorm variant, the trained `q_norm.weight` / `k_norm.weight`
tensors are dropped silently — the IFEval-evaluated LM has the same
architecture as the vanilla one. This invalidates the C3 comparison.

**Fix needed**: (a) add `--inject-qknorm` flag to `extract_lm` CLI;
(b) in the extracted dir, save the γ tensors and a small `modeling.py`
that re-injects them at load time; (c) optionally add a `qknorm`
attention-implementation variant that lm-eval can load.

### 5.3 Stage 1 → Stage 2 qknorm γ continuity
Stage 1 saves only mm_projector.bin; if qknorm γs moved during Stage 1
(they shouldn't — LLM is frozen — but the safety check matters), they
would be lost. `load_qknorm_state_if_present` exists in the loader,
but only triggers when `pretrained` is the Stage 1 dir. With Fix 5.1
above (pretrained = init dir), we need to add an explicit
`qknorm_state_dir` config field or pass via CLI.

### 5.4 Missing training deps
`training/data/__init__.py` eagerly imports lmdb-dependent modules,
and DeepSpeed Zero-2 is in the YAMLs. Neither is in `requirements.txt`.

**Fix**: `pip install lmdb deepspeed` before launching, or move
lmdb-dependent imports to be lazy / conditional.

### 5.5 transformers 4.45+ DDP-safe patch may be stale
`training/models/qwen25vl.py` calls `.to()` on `get_image_features()`'s
return value, but newer transformers makes that a tuple from
`torch.split()`. The text-only-batch collator path in
`training/data/collator.py:141` may hit this.

**Fix**: smoke-test with 1 step on text-only + image-batch mixed
before the full ~80h Stage 2.

### 5.6 Datasets are NOT downloaded by the pipeline script
`run_full_pipeline.sh` stage 1 only downloads models. The LLaVA
datasets must be prepped manually (see §2 above). The
`.jsonl` conversion and image-zip unpacking are one-shot manual steps.

---

## 6. Known gotchas (analysis-stage, less critical)

1. **Qwen3 chat-template `<think>` default breaks IFEval** — pipeline
   uses `$ROOT/Qwen3-8B-nothink` overlay; do *not* eval against raw
   Qwen3-8B without it. See `analysis/sibling_diff/README.md` §2.5.
2. **InternVL3 / 3.5 / LLaVA-LLaMA3 need `extract_direct.py`** —
   HF `AutoModel` fails (custom `trust_remote_code` config classes).
   Both `--config-src` and `--tokenizer-src` flags must point at the
   native LLM config (InternVL3 → Qwen2.5-7B-native; InternVL3.5 →
   needs InternVL-native `llm_config.json` extracted via the helper).
3. **Phi-3.5-Vision is SKIPPED** (see README §2.6 — `DynamicCache` API
   removed in transformers 4.45+ breaks the trust_remote_code class).
4. **lm-eval-harness silent crashes** — wrap with the eval-watcher
   pattern (`/tmp/eval_watcher.sh`) that re-launches with
   `--skip-existing`.
5. **Co-Authored-By: Claude commit trailer must NOT be appended** — see
   `memory/feedback_no_coauthor.md`.
6. **Stage 1.5 image-recap is SKIPPED** (advisor guidance — 2 GPU
   budget, ~3M private samples we can't access).

---

## 7. Decision points after this node

These are *not* automated by the pipeline because they need a human
judgment call on the result:

- **C3 result interpretation** (after step 13):
  - `Δ(A2) − Δ(A1) ≥ 5 pt` → strong positive → write paper.
  - `Δ(A2) − Δ(A1) ∈ [3, 5)` → moderate positive → run R2 mitigation
    (γ warm-start from Qwen3-4B) before paper.
  - `Δ(A2) − Δ(A1) < 3 pt` → null. Investigate: (a) scale to 7B?
    (b) IFEval baseline at 3B too weak to differentiate? (c) γ never
    leaves identity — try longer Stage 2?
- **R2 mitigation (γ warm-start)** if A2 underperforms — see
  `c3_training_setup.md` (recovered from git history if deleted).
- **Round 2 confirmation at 7B** if Round 1 (3B) shows positive signal.

---

## 8. Status summary (as of node-shutdown)

Done ✓
- Phenomenon (Qwen2.5 vs Qwen3, prompt-strict −9.4 vs −3.0)
- Weight diagnosis (§3 of analysis README)
- C1 sufficiency: 83.2 → 38.8 (−44.4) when all 3 norm γs killed
- C2 SAS: TRR ≈ 18 % (weak positive, mechanism-confirming)
- T measurement (sink logit gap)
- Math foundation (Theorem A + Lemma B/C, with V12 correction)
- E2 random-W perturbation (mechanism control, −61.4 pt)
- Generalization: LLaVA-LLaMA3 (−26.6), InternVL3 (−8.7), InternVL3.5
  (−1.1) — *every prediction confirmed*
- Trustworthiness sanity-check vs official refs (§2.4)
- C3 training code skeleton (loader, configs, driver) — drafted but
  has the §5 blockers above; NOT yet runnable end-to-end

Pending (this TODO covers them)
- Re-download models on new node (script: `scripts/run_full_pipeline.sh`)
- C3 training data prep (LLaVA-Pretrain + Mix-665K image sources) —
  manual; see §2
- **Fix §5.1–5.5 C3 blockers** before launching training
- C3 Stage 1 + Stage 2 on 3B (vanilla + qknorm) — ~3-5 days wall-clock
  on 2× A6000 (linearly faster with more GPUs)
- Round 2 7B confirmation (if 3B positive)
- Paper draft
