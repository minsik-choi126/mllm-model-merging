#!/usr/bin/env python3
"""Parse lm-eval result directories and print degradation / comparison tables.

Usage:
    # Compare LLM vs VLM-text-backbone
    python parse_results.py --llm <dir> --vlm <dir>

    # Compare LLM, VLM, and one or more merged models
    python parse_results.py --llm <dir> --vlm <dir> --merged exp1:<dir> exp2:<dir>

    # JSON output (for downstream scripts)
    python parse_results.py --llm <dir> --vlm <dir> --json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path


# Primary metric per task ────────────────────────────────────────────────────
TASK_METRICS: dict[str, str] = {
    "mmlu":               "acc,none",
    "mmlu_pro":           "exact_match,custom-extract",
    "gsm8k":              "exact_match,flexible-extract",
    "gsm8k_cot":          "exact_match,flexible-extract",
    "truthfulqa_mc2":     "acc,none",
    "boolq":              "acc,none",
    "ifeval":             "prompt_level_strict_acc,none",
    "arc_challenge":      "acc_norm,none",
    "arc_easy":           "acc,none",
    "hellaswag":          "acc_norm,none",
    "openbookqa":         "acc_norm,none",
    "winogrande":         "acc,none",
    "commonsense_qa":     "acc,none",
    "piqa":               "acc,none",
    "sciq":               "acc,none",
    "medqa_4options":     "acc,none",
    "race":               "acc,none",
    "drop":               "f1,none",
    "nq_open":            "exact_match,none",
    "aime24":             "exact_match,none",
    "minerva_math500":    "exact_match,none",
    "gpqa_diamond_zeroshot":         "acc,none",
    "gpqa_diamond_cot_zeroshot":     "exact_match,flexible-extract",
    "humaneval_instruct": "pass@1",
    "leaderboard_musr":   "acc_norm,none",
    "bbh_cot_fewshot":    "acc_norm,none",
}


def load_results(result_dirs: str | list[str]) -> dict[str, float]:
    """Load scores from one or more lm-eval output directories.

    Walks each directory for results*.json and extracts the primary metric
    per task according to TASK_METRICS. Later dirs fill in missing tasks.
    Subtasks like `mmlu_pro_biology` are skipped — only the aggregate row.
    """
    if isinstance(result_dirs, str):
        result_dirs = [result_dirs]

    scores: dict[str, float] = {}
    for result_dir in result_dirs:
        jsons = sorted(
            [Path(p) for p in glob.glob(
                str(Path(result_dir) / "**" / "results*.json"), recursive=True
            )],
            key=os.path.getmtime,
            reverse=True,
        )
        for jf in jsons:
            d = json.loads(jf.read_text())
            for task, vals in d.get("results", {}).items():
                if task.startswith("mmlu_pro_"):
                    continue
                if task in scores:
                    continue
                preferred = TASK_METRICS.get(task)
                if preferred and preferred in vals and isinstance(vals[preferred], (int, float)):
                    scores[task] = round(vals[preferred] * 100, 2)
                else:
                    for k, v in vals.items():
                        if isinstance(v, float) and not k.endswith("_stderr"):
                            scores[task] = round(v * 100, 2)
                            break
    return scores


def trr(merged: float, vlm: float, llm: float) -> float:
    """Text Retention Rate: 100% means merged matches LLM, 0% means matches VLM."""
    gap = llm - vlm
    return (merged - vlm) / gap * 100 if abs(gap) > 1e-6 else 0.0


def print_table(
    llm: dict[str, float],
    vlm: dict[str, float],
    merged: dict[str, dict[str, float]] | None = None,
    as_json: bool = False,
):
    tasks = sorted(set(list(llm) + list(vlm)))
    rows = []
    for t in tasks:
        l, v = llm.get(t), vlm.get(t)
        if l is None or v is None:
            continue
        row: dict = {"task": t, "llm": l, "vlm": v, "delta": round(v - l, 2)}
        if merged:
            for name, mscores in merged.items():
                m = mscores.get(t)
                if m is not None:
                    row[name] = m
                    row[f"{name}_trr"] = round(trr(m, v, l), 1)
        rows.append(row)

    if as_json:
        print(json.dumps(rows, indent=2))
        return

    cols = ["task", "llm", "vlm", "delta"]
    if merged:
        for name in merged:
            cols += [name, f"TRR({name})"]
    widths = [26, 7, 7, 7] + [9, 9] * (len(merged) if merged else 0)
    fmt_hdr = "  ".join(f"{{:<{w}}}" if i == 0 else f"{{:>{w}}}" for i, w in enumerate(widths))
    fmt_row = "  ".join(f"{{:<{w}}}" if i == 0 else f"{{:>{w}}}" for i, w in enumerate(widths))

    print(fmt_hdr.format(*cols))
    print("  " + "-" * (sum(widths) + 2 * (len(widths) - 1)))
    for row in rows:
        d = row["delta"]
        vals = [
            row["task"],
            f"{row['llm']:.2f}",
            f"{row['vlm']:.2f}",
            f"{d:+.2f}%",
        ]
        if merged:
            for name in merged:
                m = row.get(name)
                t_ = row.get(f"{name}_trr")
                vals += [f"{m:.2f}" if m is not None else "-",
                         f"{t_:.1f}%" if t_ is not None else "-"]
        print(fmt_row.format(*vals))

    deg = [r for r in rows if r["delta"] <= -1.5]
    print(f"\n  {len(deg)}/{len(rows)} tasks show degradation (Δ < -1.5 pt)")
    avg_delta = sum(r["delta"] for r in rows) / len(rows) if rows else 0
    print(f"  Average delta: {avg_delta:+.2f} pt")

    if merged:
        for name in merged:
            trrs = [r[f"{name}_trr"] for r in rows if f"{name}_trr" in r]
            if trrs:
                print(f"  Avg TRR [{name}]: {sum(trrs)/len(trrs):.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Parse lm-eval results and compare models")
    parser.add_argument("--llm",    required=True, nargs="+", help="LLM result directory (multiple OK)")
    parser.add_argument("--vlm",    required=True, nargs="+", help="VLM-LM result directory (multiple OK)")
    parser.add_argument("--merged", nargs="+", metavar="NAME:DIR",
                        help="Merged model results, e.g. method_a:eval_results/exp1")
    parser.add_argument("--json",   action="store_true", help="Output as JSON")
    args = parser.parse_args()

    llm_scores = load_results(args.llm)
    vlm_scores = load_results(args.vlm)

    if not llm_scores:
        print(f"ERROR: No results found in {args.llm}", file=sys.stderr); sys.exit(1)
    if not vlm_scores:
        print(f"ERROR: No results found in {args.vlm}", file=sys.stderr); sys.exit(1)

    merged_scores: dict[str, dict[str, float]] | None = None
    if args.merged:
        merged_scores = {}
        for spec in args.merged:
            name, path = spec.split(":", 1)
            merged_scores[name] = load_results(path)

    print_table(llm_scores, vlm_scores, merged_scores, as_json=args.json)


if __name__ == "__main__":
    main()
