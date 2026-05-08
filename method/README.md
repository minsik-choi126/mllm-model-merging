# method/

**Status: TBD.**

This slot will hold the merging algorithm — the part that takes a base LLM and
the corresponding extracted VLM-LM and produces a merged model that recovers
text capability while preserving vision performance.

Until the method lands, the rest of the repo is fully usable on its own:

- `extraction/` — extract any VLM's text backbone into a clean HF model.
- `evaluation/text/` — run the 8-task text degradation eval.
- `evaluation/vision/` — run the 6-task vision eval.

What will live here:

- the merge implementation (per-weight Fisher / signature / shrinkage logic)
- a CLI entry point: `python -m method.merge --pair <key> --output <dir>`
- ablation configs that recover known baselines as degenerate cases
- unit tests on synthetic data
