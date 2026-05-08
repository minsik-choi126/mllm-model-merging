# Overview

Why this repo exists, what it has, and what is intentionally left out.

## The problem

A vision-language model is usually built by taking a strong text LLM (Qwen,
Llama, Phi, InternLM…) and fine-tuning it on image–text data. The result has
two halves:

- the **vision tower / cross-modal connector**, which is genuinely new
- the **text backbone**, which is the same LLM, but its weights have moved

Whether you do full fine-tuning (Qwen-VL, Phi-Vision) or parameter-efficient
LoRA (LLaVA-LLaMA3), the text backbone after multimodal training is no
longer the same model that the vision team started from. Across the three
architecture families we tested, **the text-only capability of the post-VLM
LM regresses noticeably**, sometimes by 30+ points on a single benchmark
(see `results/text_degradation.md`).

This is interesting for two reasons:

1. **Practical**: VLMs are increasingly used as drop-in replacements for
   their base LLMs. The regression silently degrades downstream behavior.
2. **Methodological**: it sets up a clean weight-merge problem — given the
   original LLM and the post-VLM text backbone, can we recover the LLM's
   text quality without breaking the VLM's vision capability?

## What this repo provides

- **`extraction/`**: a small loader that reads any VLM checkpoint, walks
  whatever proprietary key prefix that family uses
  (`model.language_model.*`, `language_model.model.*`,
  `language_model.output.weight`, …) and rewrites it onto the standard
  HuggingFace causal-LM schema. The resulting directory is a real
  text-only model — `lm-evaluation-harness` accepts it as-is.

- **`evaluation/text/`**: an opinionated 8-task harness wrapper. Six tasks
  on full splits; `mmlu_pro` on a 4.16% stratified subsample (the full set
  is impractical at this scale); standardized chat-template handling.

- **`evaluation/vision/`**: a 6-task `VLMEvalKit` wrapper for the core
  vision benchmark set used by the merging-paper baselines.

- **`results/`**: the published numbers, plus a slot for figures.

- **`method/`**: reserved for the merging algorithm. Currently TBD.

## What is intentionally NOT in this repo

- The merging algorithm itself (still in development).
- Failed experiments, ablation logs, hyperparameter sweeps.
- Any internal training code or data.

The repo is meant to be a stable reference for the **measurement** problem
and the **infrastructure** around it. The method is published separately
once it stabilizes.

## Key design choices

- **Schema normalization at load time, not at merge time.** Different VLM
  families use radically different state-dict layouts. Doing the
  normalization once, when reading weights from disk, lets every other piece
  of code (eval, inspection, and eventually merging) treat any VLM's text
  path as a plain HF causal LM.

- **Extracted LM uses the LLM's tokenizer and config.** When you pass
  `--llm-template`, the extracted directory is a perfect drop-in for the
  base LLM — same vocab, same chat template, same context length. This is
  what makes the LLM ↔ VLM-LM comparison fair.

- **`mmlu_pro --limit 0.0416 --seed 42`.** This subsample (≈500 items
  proportional across subjects) is the standard we use for *all* MMLU-Pro
  numbers in this repo, including comparisons to merged models. Don't
  change it without re-running every prior baseline.

- **Vision core set = 6 tasks.** Matches the ACOM-style merging-paper
  baselines, so vision results from this repo are directly comparable to
  those.

## Pointers into the code

- `extraction/loader.py:normalize_text_backbone_state_dict` — the key
  remapping logic. Add new VLM families by extending the prefix branches
  here.
- `extraction/extract_lm.py:extract_lm_from_vlm` — the main entry point.
- `extraction/models.yaml` — registered VLM↔LLM pairs and their text-side
  shape parameters.
- `evaluation/text/parse_results.py` — score table + delta + TRR (Text
  Retention Rate) summary.
