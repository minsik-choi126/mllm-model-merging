# Text eval (8 tasks)

Wrapper around [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness)
for the 8-task degradation set used throughout this repo.

## Tasks

| Task                    | Split     | Metric                          | Notes                              |
|-------------------------|-----------|---------------------------------|------------------------------------|
| `mmlu`                  | full      | acc                             | 14k items, broad knowledge         |
| `mmlu_pro`              | **4.16%** | exact_match (custom-extract)    | ≈500 items via `--limit 0.0416 --seed 42` (full set is too slow) |
| `gsm8k_cot`             | full      | exact_match (flexible-extract)  | grade-school math, CoT             |
| `truthfulqa_mc2`        | full      | acc                             | factuality                         |
| `boolq`                 | full      | acc                             | yes/no QA                          |
| `ifeval`                | full      | prompt_level_strict_acc         | instruction following              |
| `gpqa_diamond_zeroshot` | full      | acc                             | hard graduate-level QA             |
| `eq_bench`              | full      | score                           | emotional intelligence             |

Only `mmlu_pro` is sub-sampled. All other tasks use their full evaluation
splits — partial sampling there gives misleading deltas when comparing
LLM ↔ VLM-LM ↔ merged models.

## Setup

Install `lm-evaluation-harness` in a venv:

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

Make sure `lm_eval` is on `PATH`, or point `LM_EVAL_BIN` at the binary:

```bash
export LM_EVAL_BIN=/path/to/your/.venv/bin/lm_eval
```

## Run

```bash
bash eval_8tasks.sh \
    --model Qwen/Qwen2.5-7B-Instruct \
    --gpu 0 \
    --output eval_results/llm_qwen25_7b
```

For an extracted VLM backbone:

```bash
bash eval_8tasks.sh \
    --model cache/extracted/qwen25vl_7b_lm \
    --gpu 0 \
    --output eval_results/vlm_qwen25vl_7b
```

The script runs in two stages — Step 1 covers the seven full-set tasks,
Step 2 covers `mmlu_pro` with the stratified subsample. Both write into the
same `--output` directory.

## Compare

```bash
python parse_results.py \
    --llm eval_results/llm_qwen25_7b \
    --vlm eval_results/vlm_qwen25vl_7b
```

Prints per-task scores, deltas, count of tasks with ≥1.5 pt degradation, and
average delta. With `--merged name:path/to/dir` you also get TRR (Text
Retention Rate) for each merged model.

## Important caveats

- Always run all 8 tasks the same way for the LLM and the VLM-LM. If you
  re-run only some tasks, the comparison table silently picks the most
  recent JSON per task.
- `mmlu_pro` `--limit 0.0416` ≈ 500 items proportionally across subjects;
  changing this breaks comparability with our published numbers.
- `--apply_chat_template` is on by default. Disable it (with
  `--no-chat-template`) only when both the LLM and the VLM-LM are *base*
  models, not instruct-tuned.
