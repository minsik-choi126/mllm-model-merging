# Text eval (8 tasks)

Wrapper around [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness)
for the 8-task degradation set used throughout this repo. **Per-task protocols
are tuned to reproduce Qwen2.5-Instruct's published numbers within the
±1–5 pt drift inherent to lm-eval-harness vs Qwen's eval stacks** (OpenCompass /
TIGER-Lab / Google-IFEval-reference). For every comparison we make
(LLM ↔ VLM-LM ↔ merged), all three models are run through the same protocols.

## Per-task protocol

| Task                          | n-shot | Chat tpl. | Generation                    | Qwen2.5-7B-Instruct (official) | Source           |
|-------------------------------|:------:|:---------:|-------------------------------|:------------------------------:|------------------|
| `gsm8k_cot`                   |   0    |   yes     | greedy, max_gen_toks 512      | **91.6**                       | Qwen2.5 LLM blog |
| `ifeval`                      |   0    |   yes     | greedy, max_gen_toks 1280     | **71.2** (prompt-strict)       | Qwen2.5 LLM blog |
| `gpqa_diamond_cot_zeroshot`   |   0    |   yes     | greedy, max_gen_toks 1024 CoT | **36.4**                       | Qwen2.5 LLM blog |
| `mmlu_pro`                    |   5    |   yes     | greedy CoT, max_gen_toks 2048 | **56.3** (full set)            | Qwen2.5 LLM blog |
| `eq_bench`                    |   0    |   yes     | greedy, max_gen_toks 512      | — (lm-eval is v2.1, eqbench.com is v3 → incomparable) | — |
| `mmlu`                        |   5    |   **no**  | log-likelihood                | — (Qwen reports MMLU-redux 75.4, not vanilla MMLU) | — |
| `boolq`                       |   0    |   **no**  | log-likelihood                | — (no Qwen number)             | — |
| `truthfulqa_mc2`              |   0    |   **no**  | log-likelihood                | — (no Qwen number)             | — |

### Why these flags

- **0-shot CoT for `gsm8k_cot`**: lm-eval's task yaml defaults to **8-shot**, but
  Qwen blog reports `91.6` from a **0-shot CoT + chat-template** setup. Override
  required to land within drift.
- **`gpqa_diamond_cot_zeroshot`, not `gpqa_diamond_zeroshot`**: the bare `_zeroshot`
  variant does not elicit a CoT and lands ≈29 (matches Open-LLM-Leaderboard); the
  `_cot_zeroshot` variant matches Qwen's reported 36.4.
- **`max_gen_toks=1280` for IFEval**: matches Google's reference implementation;
  shorter caps truncate multi-paragraph instruction outputs and lower the
  prompt-strict score.
- **`mmlu_pro` 5-shot CoT generation**: TIGER-Lab's official protocol. With
  `--mmlu-pro-limit 0.0416` you get a ≈500-item stratified subsample that adds
  ±2-3 pt noise vs the full-set Qwen number; default is full set.
- **No chat template for `mmlu`/`boolq`/`truthfulqa_mc2`**: these are
  log-likelihood tasks (`acc` over fixed completions). Wrapping the prompt in a
  chat template shifts the LL onto answer tokens that come after the
  `<|im_start|>assistant` marker, which is unstable and degrades scores 1-3 pt.
  Qwen does not chat-template their LL evaluations.

### Cross-cutting drift you cannot eliminate

- Qwen evaluates with **OpenCompass / TIGER-Lab / Google-IFEval-reference** —
  not lm-eval-harness. Even with matched protocols, prompt-template differences,
  regex extraction, and few-shot example sampling drift by 1-5 pt. Treat
  reproductions within **±3 pt** of Qwen blog as success.
- Qwen2.5-VL-7B-Instruct **does not publish text-only numbers** for any of these
  8 tasks (the VL paper covers only image/video/agent benchmarks for the 7B;
  the 72B paper claims "complete capability alignment with the pure text
  Qwen2.5-72B" without a 7B analogue table). The VLM-LM extracted from the
  vision model is therefore the first-party reference here.

## Setup

```bash
pip install "lm_eval[ifeval,math]==0.4.5"
# (or git clone & pip install -e . from EleutherAI/lm-evaluation-harness)
export LM_EVAL_BIN=/opt/conda/bin/lm_eval
```

## Run

```bash
bash eval_8tasks.sh \
    --model /path/to/Qwen2.5-7B-Instruct \
    --gpu 0 \
    --output eval_results/llm_qwen25_7b
```

For an extracted VLM backbone:

```bash
bash eval_8tasks.sh \
    --model /path/to/extracted/qwen25vl_7b_lm \
    --gpu 0 \
    --output eval_results/vlm_qwen25vl_7b
```

For the legacy ≈500-item `mmlu_pro` subsample (saves time at the cost of ±2-3 pt):

```bash
bash eval_8tasks.sh --model ... --output ... --mmlu-pro-limit 0.0416
```

Each task is its own `lm_eval` invocation (the per-task protocols differ), all
writing into the same `--output` directory. `parse_results.py` reads them
recursively.

## Compare

```bash
python parse_results.py \
    --llm eval_results/llm_qwen25_7b \
    --vlm eval_results/vlm_qwen25vl_7b
```

Prints per-task scores, deltas, count of tasks with ≥1.5 pt degradation, and
average delta. With `--merged name:path/to/dir` you also get TRR (Text Retention
Rate) for each merged model.

## Important caveats

- Always run all 8 tasks the same way across LLM, VLM-LM, and merged.
  Re-running only some tasks silently mixes JSONs (the parser picks the most
  recent per task).
- `--mmlu-pro-limit` changes the comparison set; pick once per study and stick to
  it across all three models.
- The `mmlu_redux` task is a closer match to what Qwen reports for "MMLU"; we
  keep `mmlu` here for continuity with prior literature, with the protocol-
  honest LL setup (no chat template).
