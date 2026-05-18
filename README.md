# mllm-model-merging

Tools for measuring **text-side degradation** when an LLM is fine-tuned into
a vision-language model, plus the eval infrastructure used in our merging
experiments.

The merging method itself is in development and will land under
[`method/`](method/) when ready.

> **Current research focus** — see [`analysis/sibling_diff/`](analysis/sibling_diff/)
> for the investigation of VLM text degradation as an
> **attention-sink corruption** phenomenon. The cross-vendor natural experiment
> (Qwen3 with QK-RMSNorm vs Qwen2.5 without) suggests that VL training's
> damage to instruction-following is concentrated in a small set of
> sink-encoding weight columns, and that **QK-RMSNorm structurally protects
> the sink encoding** from being touched by VL updates.
>
> **To reproduce from a clean machine**: see [`TODO.md`](TODO.md) for
> the restart guide. [`scripts/run_full_pipeline.sh`](scripts/run_full_pipeline.sh)
> orchestrates the analysis matrix (model downloads → text-backbone
> extraction → weight analysis → C1/C2/E2 ablations → 5-pair IFEval).
> C3 training (architectural-causality VLM fine-tune) is OFF by
> default and has known blockers; see TODO.md §5.

## Headline finding

VLMs lose text capability across architecture families and training regimes.
Numbers are VLM-text-backbone − base-LLM, in points (negative = regression).

| Model            | Training | MMLU | MMLU-Pro | GSM8K | TruthfulQA | BoolQ | IFEval | GPQA | EQ-Bench |
|------------------|----------|-----:|---------:|------:|-----------:|------:|-------:|-----:|---------:|
| Qwen2.5-VL-7B    | Full FT  | −3.1 | **−12.7** | −10.3 | −11.0 | −1.8 | −13.7 | −11.8 | −6.2 |
| LLaVA-LLaMA3-8B  | LoRA     | +0.4 | −2.6 | **−24.1** | −7.8 | −1.8 | **−35.2** | −3.2 | **−33.6** |
| Phi-3.5-Vision   | Full FT  | −7.4 | **−37.9** | −11.3 | **−18.0** | −2.0 | **−29.6** | −11.5 | −11.8 |

**Bold** = drop ≥ 15 pt. Per-model notes and reproduction commands:
[`results/text_degradation.md`](results/text_degradation.md).

## Layout

```
extraction/                VLM → text-only HF model (loader, registry, CLI)
evaluation/
  text/                    8-task eval (lm-evaluation-harness)
  vision/                  6-task core eval (VLMEvalKit)
method/                    TBD — merging algorithm
analysis/sibling_diff/     Sibling-pair (Qwen2.5↔Qwen3) sink-mechanism study
results/                   Published numbers + figures
docs/                      Design notes
```

## Quick start

### Extract a VLM's text backbone

```bash
python -m extraction.extract_lm --pair qwen25vl_7b --output cache/extracted/qwen25vl_7b_lm
```

Pairs available: `qwen25vl_7b`, `qwen3vl_8b`, `internvl25_8b`,
`llava_llama3_8b`, `phi35_vision`. Or pass `--vlm` and `--llm-template`
directly.

### Text eval (8 tasks)

```bash
export LM_EVAL_BIN=/path/to/.venv/bin/lm_eval

bash evaluation/text/eval_8tasks.sh \
    --model cache/extracted/qwen25vl_7b_lm \
    --output eval_results/qwen25vl_7b_lm
```

### Vision eval (6 tasks)

```bash
export VLMEVAL_DIR=/path/to/VLMEvalKit
export PYTHON_BIN=/path/to/.venv/bin/python

bash evaluation/vision/run_vision_eval.sh \
    --model Qwen/Qwen2.5-VL-7B-Instruct
```

### Compare LLM vs VLM-LM

```bash
python evaluation/text/parse_results.py \
    --llm eval_results/qwen25_7b_llm \
    --vlm eval_results/qwen25vl_7b_lm
```

## Requirements

`torch`, `transformers`, `safetensors`, `pyyaml`. Eval pipelines additionally
need `lm-evaluation-harness` and `VLMEvalKit`. See `requirements.txt`.

See [`docs/overview.md`](docs/overview.md) for design notes.
