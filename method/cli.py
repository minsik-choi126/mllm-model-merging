"""End-to-end E-Pull merge CLI.

Usage:
    python -m method.cli \
        --base /path/to/base_llm \
        --models llm:/path/to/llm vlm_lm:/path/to/extracted_vlm_lm \
        --calibration-text /path/to/calibration.jsonl \
        --output /path/to/merged
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from extraction.loader import (
    _from_pretrained_with_local_fallback,
)
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .covariance import collect_input_grams
from .merge import EpullConfig, epull_merge_state_dicts


def _load_calibration_texts(path: str | None, n: int) -> list[str]:
    if path is None:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train",
                          streaming=True)
        out = []
        for row in ds:
            text = row.get("text", "").strip()
            if len(text) >= 64:
                out.append(text)
            if len(out) >= n:
                break
        return out

    p = Path(path)
    if p.suffix in (".jsonl",):
        out = []
        with p.open() as f:
            for line in f:
                d = json.loads(line)
                t = d.get("text") or d.get("input") or d.get("prompt")
                if t and len(t.strip()) >= 32:
                    out.append(t)
                if len(out) >= n:
                    break
        return out
    return [p.read_text()]


def _collect_for_model(
    model_id: str,
    tokenizer_id: str | None,
    texts: list[str],
    *,
    device: str,
    max_seq_len: int,
):
    print(f"\n[calib] Loading {model_id}")
    model = _from_pretrained_with_local_fallback(
        AutoModelForCausalLM, model_id,
        dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    )
    tok = _from_pretrained_with_local_fallback(
        AutoTokenizer, tokenizer_id or model_id, trust_remote_code=True,
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print(f"[calib] Collecting Gram on {len(texts)} sequences (seqlen≤{max_seq_len})")
    art = collect_input_grams(
        model, tok, texts, device=device, max_seq_len=max_seq_len,
    )
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    art_cpu = art.clone_to_cpu()
    del model
    torch.cuda.empty_cache()
    return sd, art_cpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base model HF id or path")
    ap.add_argument(
        "--models", nargs="+", required=True,
        metavar="NAME:PATH",
        help="Fine-tuned models, e.g. llm:/p1 vlm_lm:/p2",
    )
    ap.add_argument("--alphas", nargs="*", type=float, default=None,
                    help="Modality weights, default = uniform")
    ap.add_argument("--output", required=True)
    ap.add_argument("--calibration-text", default=None,
                    help="JSONL of {'text': ...}; default: wikitext-103 stream")
    ap.add_argument("--n-samples", type=int, default=128)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-stats", default=None)
    ap.add_argument("--jacobi-sweeps", type=int, default=2,
                    help="CS-Jacobi sweep count for orthogonal FG joint diag (default 2)")
    args = ap.parse_args()

    pairs = []
    for spec in args.models:
        if ":" not in spec:
            raise ValueError(f"--models entry {spec!r} must be NAME:PATH")
        name, path = spec.split(":", 1)
        pairs.append((name, path))

    k = len(pairs)
    alphas = tuple(args.alphas) if args.alphas else tuple([1.0 / k] * k)
    if len(alphas) != k:
        raise ValueError(f"--alphas len ({len(alphas)}) != #models ({k})")

    texts = _load_calibration_texts(args.calibration_text, args.n_samples)
    print(f"[setup] base={args.base}  k={k}  alphas={alphas}")
    for n, p in pairs:
        print(f"[setup]   modality {n}: {p}")
    print(f"[setup] {len(texts)} calibration sequences")

    cfg = EpullConfig(
        alphas=alphas, device=args.device, jacobi_sweeps=args.jacobi_sweeps,
    )

    sds: list[dict] = []
    arts = []
    for name, path in pairs:
        sd, art = _collect_for_model(
            path, tokenizer_id=args.base, texts=texts,
            device=args.device, max_seq_len=args.max_seq_len,
        )
        sds.append(sd)
        arts.append(art)

    print("\n[base] Loading base model state dict")
    base_model = _from_pretrained_with_local_fallback(
        AutoModelForCausalLM, args.base,
        dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True,
    )
    base_sd = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}
    del base_model

    print("\n[merge] Running E-Pull merge")
    merged_sd, stats = epull_merge_state_dicts(base_sd, sds, arts, cfg)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[save] Writing merged model to {out_dir}")
    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in merged_sd.items()},
              str(out_dir / "model.safetensors"))

    cfg_obj = _from_pretrained_with_local_fallback(
        AutoConfig, args.base, trust_remote_code=True,
    )
    cfg_obj.save_pretrained(out_dir)
    tok = _from_pretrained_with_local_fallback(
        AutoTokenizer, pairs[0][1], trust_remote_code=True,
    )
    tok.save_pretrained(out_dir)

    if args.save_stats:
        rows = []
        for s in stats:
            rows.append({
                "name": s.name,
                "mode": s.mode,
                "in_dim": s.in_dim, "out_dim": s.out_dim,
                "off_diagonal_residual": s.off_diagonal_residual,
                "fg_cost": s.fg_cost,
                "commutator_residual": s.commutator_residual,
                "n_jacobi_sweeps": s.n_jacobi_sweeps,
                "avg_gate": s.avg_gate,
                "avg_entropy_norm": s.avg_entropy_norm,
                "owner_fraction_per_modality": s.owner_fraction_per_modality,
                "chosen_modality": s.chosen_modality,
                "diag_method": s.diag_method,
            })
        Path(args.save_stats).write_text(json.dumps(rows, indent=2))
        print(f"[save] stats written to {args.save_stats}")

    n_epull = sum(1 for s in stats if s.mode == "epull")
    n_owner = sum(1 for s in stats if s.mode == "owner_energy")
    epull = [s for s in stats if s.mode == "epull"]
    if epull:
        avg_gate = sum(s.avg_gate for s in epull) / len(epull)
        avg_off = sum(s.off_diagonal_residual for s in epull) / len(epull)
        avg_fg = sum(s.fg_cost for s in epull) / len(epull)
        avg_comm = sum(s.commutator_residual for s in epull) / len(epull)
        avg_sw = sum(s.n_jacobi_sweeps for s in epull) / len(epull)
        print(
            f"\n[summary] epull layers: {n_epull}  owner-energy layers: {n_owner}\n"
            f"          avg gate {avg_gate:.4f}  avg FG cost {avg_fg:.3e}  "
            f"avg off-diag {avg_off:.3e}  avg commutator {avg_comm:.3e}  "
            f"avg jacobi sweeps {avg_sw:.2f}"
        )
    else:
        print(f"\n[summary] no epull layers (all owner-energy: {n_owner})")


if __name__ == "__main__":
    main()
