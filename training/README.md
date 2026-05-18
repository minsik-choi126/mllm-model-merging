# training/ — LLaVA-style VLM training for the architectural causality experiment (C3)

This module trains Qwen2.5-VL-style VLMs from a from-scratch composed init
(`Qwen2.5-VL vision tower` + `Qwen2.5 text LM` + `fresh projector`) so we can
run a controlled architectural-causality experiment: do the *same training
recipe and data* on (a) the vanilla LLM and (b) the LLM with QK-RMSNorm
modules injected, and compare the IFEval drops.

The code is derived from
[`MERIT`](https://github.com/minsik-choi126/MERIT) (Choi & Kim 2026,
"Decentralized Instruction Tuning"), refactored under namespace `training.*`
and extended with a `--inject-qknorm` switch.

## What this experiment tests

**Hypothesis (C3)** — adding per-head QK-RMSNorm modules to a non-QK-norm
LLM and training as a VLM with the same recipe should yield a smaller
IFEval drop than the vanilla baseline. If true, this isolates QK-norm as
an architectural cause of VLM IFEval preservation (controlled for data,
RLHF, schedule, etc.).

| Variant | Architecture | Training | Expected IFEval Δ |
|---|---|---|---|
| L0 (base) | Qwen2.5-3B-Instruct (text-only) | none | 0 (baseline) |
| A1 (vanilla) | Qwen2.5-3B-Instruct | LLaVA Stage 1 + Stage 2 | predict −6 ~ −10 pt |
| **A2 (qknorm)** | Qwen2.5-3B + q_norm/k_norm inject (γ=1) | same recipe | **predict −2 ~ −5 pt** |

Critical comparison: **A2 − A1**. If ≥ 3 pt, architectural causality
supported.

## Layout

```
training/
├── README.md                     # this file
├── version.py
├── __init__.py
├── models/
│   ├── loader.py                 # load_merit_model(cfg, stage); honours inject_qknorm flag
│   ├── attention.py              # resolve_attn_implementation
│   ├── projector.py              # projector_parameters helper
│   ├── qwen25vl.py               # Qwen2.5-VL HF loader + DDP-safe patch
│   ├── vision_tower.py           # freeze_vision_tower
│   ├── checkpoint.py
│   └── qknorm_injection.py       # HeadDimRMSNorm + inject_qknorm() + load_qknorm_state_if_present
├── data/                         # LLaVA / FLAN / VFLAN / Mix-176 loaders + DDP-safe collator
├── train/
│   ├── cli.py                    # entry: python -m training.train.cli --config <yaml>
│   ├── arguments.py              # config dataclasses
│   ├── stage1_trainer.py
│   ├── stage2_trainer.py
│   ├── trainer_utils.py
│   └── deepspeed_configs/zero2.json
├── utils/                        # logging / IO / dist / seed / yaml helpers
├── scripts/
│   ├── build_init_from_pretrained.py   # compose Qwen2.5-VL vision + Qwen2.5 LM + fresh projector
│   └── run_c3_3b.sh                    # full C3 pipeline driver
└── configs/
    ├── _base_/
    │   ├── model_qwen25_3b.yaml         # pre-aligned Qwen2.5-VL-3B-Instruct
    │   ├── model_qwen25_3b_init.yaml    # composed-init backbone (ckpts/c3/init/...)
    │   └── model_qwen25_7b.yaml
    ├── 3b/
    │   ├── c3_vanilla_align.yaml        # A1 Stage 1
    │   ├── c3_vanilla_stage2.yaml       # A1 Stage 2
    │   ├── c3_qknorm_align.yaml         # A2 Stage 1 (inject_qknorm: true)
    │   ├── c3_qknorm_stage2.yaml        # A2 Stage 2
    │   ├── stage1.yaml                  # MERIT-paper Stage 1 (VFLAN data)
    │   └── stage2.yaml                  # MERIT-paper Stage 2
    └── data/
        ├── align_llava_pretrain.json    # LLaVA-Pretrain-558K manifest
        └── stage2_llava_mix665k.json    # LLaVA-1.5-mix665k manifest
```

## QK-norm injection mechanism

`models/qknorm_injection.py` adds per-head RMSNorm modules (`q_norm`,
`k_norm`) of dimension `head_dim` to every attention block of the LLM.
γ is initialized to 1 (identity normalization) and registered as a
*trainable parameter* — Stage 1 keeps it frozen (LLM frozen anyway),
Stage 2 lets it move along with the rest of the LLM.

```python
from training.models.qknorm_injection import inject_qknorm
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(...)
inject_qknorm(model)
# model.model.language_model.layers[i].self_attn now has q_norm, k_norm
```

In `loader.py`, set `model.inject_qknorm: true` in the YAML config to
auto-inject after loading. The loader also restores `q_norm.weight` /
`k_norm.weight` values from disk if the checkpoint is from a previous
training stage (so Stage 2 continues the γ learned during Stage 1).

## How to run (C3 at 3B)

```bash
# Prerequisites:
#   - Qwen2.5-VL-3B-Instruct (HF: Qwen/Qwen2.5-VL-3B-Instruct) — vision tower source
#   - Qwen2.5-3B-Instruct    (HF: Qwen/Qwen2.5-3B-Instruct)   — text LM source
#   - LLaVA-Pretrain-558K    (liuhaotian/LLaVA-Pretrain)      — Stage 1 data
#   - LLaVA-Mix-665K         (LLaVA-Instruct-150K + image directories) — Stage 2 data
#   - Edit training/configs/data/*.json with your local paths
#
# Hardware: 2× A6000/A100 (per advisor's "2-card LLaVA recipe" guidance)
#
# Run both variants end-to-end (~3-4 days):
export CKPT_ROOT=/your/work/dir/c3_3b
bash training/scripts/run_c3_3b.sh

# Or one at a time:
VARIANT=vanilla bash training/scripts/run_c3_3b.sh
VARIANT=qknorm  bash training/scripts/run_c3_3b.sh
```

## What we skip vs MERIT's full pipeline

| MERIT 7B pipeline | C3 3B (this module) |
|---|---|
| Stage Align (LLaVA-Pretrain) | ✓ |
| Stage 1.5 (image-recap corpus, 2.86M private samples) | **skipped** (per advisor guidance, fits 2 GPUs) |
| Stage 2 (LLaVA-Mix-665K) | ✓ |
| Gradient extraction + PCA split (MERIT method) | n/a (we're not running MERIT branches) |
| Branch training + token-weighted merge | n/a |
| paper8 lmms-eval (vision benchmarks) | optional — we focus on IFEval drop |

## Evaluation

The end of `scripts/run_c3_3b.sh` automatically:
1. Extracts the LM text backbone from each trained VLM (via
   `extraction/extract_lm.py` in this same repo).
2. Runs IFEval on `Qwen2.5-3B-Instruct` (baseline) and on each extracted
   VLM-LM via `evaluation/text/run_eval_matrix.sh`.
3. Stores comparison in `eval_results/c3_3b/`.

## Caveats

- This is *not* a faithful reproduction of Qwen team's VLM training recipe.
  The IFEval drop magnitude in A1 may differ from the −9.4 pt observed in
  the public Qwen2.5-VL-7B-Instruct. What matters for C3 is the
  *differential* between A1 and A2 with the SAME recipe.
- Stage 1.5 omission may reduce both A1 and A2 absolute IFEval scores
  uniformly; the differential should be preserved.
- γ-only-trained-in-Stage-2 may not develop heavy-tailed amplifier
  structure as fully as Qwen3's natively-co-trained γ. Mitigation if A2
  underperforms expectation: warm-start γ from Qwen3-4B-Instruct's γ
  values (see `c3_training_setup.md` R2 mitigation).
