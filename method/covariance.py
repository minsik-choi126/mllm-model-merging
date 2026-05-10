"""Activation Gram (input-side covariance) collection per linear layer.

For each `nn.Linear`, captures inputs X across calibration tokens, accumulates
G = sum X^T X, and returns C = G / N. Trace-normalized to C / tr(C) * d.

Down-projection layers (whose input dim is the intermediate size, e.g. 18944
for Qwen2.5-7B) are too large for dense covariance; only the running trace is
stored — used downstream to pick the dominant-energy modality, per the method
section.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn


DOWN_PROJ_TOKENS = ("down_proj", "fc2", "wo")  # MLP down projection by family


def is_down_projection(layer_name: str) -> bool:
    return any(tok in layer_name.split(".")[-1] for tok in DOWN_PROJ_TOKENS)


def linear_layers_for_merging(model: nn.Module):
    """Yield (qualified_name, module) for nn.Linear modules under `model.`.

    Skips lm_head and embedding-replacement linears unless they're inside
    the standard text-backbone path (model.layers.*).
    """
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if "lm_head" in name:
            continue
        if not name.startswith("model.layers."):
            continue
        yield name, module


@dataclass
class GramArtifacts:
    """Per-linear-layer activation statistics."""
    grams: dict[str, torch.Tensor]            # full Gram for active-merge layers
    traces: dict[str, float]                  # only for down_proj layers
    counts: dict[str, int]                    # token count seen at each layer
    in_dims: dict[str, int]                   # input dim per layer
    out_dims: dict[str, int]                  # output dim per layer

    def clone_to_cpu(self) -> "GramArtifacts":
        return GramArtifacts(
            grams={k: v.detach().to("cpu") for k, v in self.grams.items()},
            traces=dict(self.traces),
            counts=dict(self.counts),
            in_dims=dict(self.in_dims),
            out_dims=dict(self.out_dims),
        )


@torch.no_grad()
def collect_input_grams(
    model: nn.Module,
    tokenizer,
    texts: Iterable[str],
    *,
    device: str = "cuda",
    accum_dtype: torch.dtype = torch.float32,
    max_seq_len: int = 512,
    trace_normalize: bool = True,
) -> GramArtifacts:
    """Run a forward pass over `texts` and collect input Gram per linear layer."""
    model.eval()

    grams: dict[str, torch.Tensor] = {}
    traces: dict[str, float] = {}
    counts: dict[str, int] = {}
    in_dims: dict[str, int] = {}
    out_dims: dict[str, int] = {}

    handles = []
    target_layers = list(linear_layers_for_merging(model))

    for name, module in target_layers:
        in_dims[name] = module.in_features
        out_dims[name] = module.out_features

        if is_down_projection(name):
            def _hook_trace(_mod, inp, _out, ln=name):
                x = inp[0].detach().to(accum_dtype).reshape(-1, inp[0].shape[-1])
                traces[ln] = traces.get(ln, 0.0) + (x * x).sum().item()
                counts[ln] = counts.get(ln, 0) + x.shape[0]
            handles.append(module.register_forward_hook(_hook_trace))
        else:
            def _hook_full(_mod, inp, _out, ln=name):
                x = inp[0].detach().to(accum_dtype).reshape(-1, inp[0].shape[-1])
                g_inc = x.t() @ x  # (in, in)
                if ln not in grams:
                    grams[ln] = torch.zeros(
                        g_inc.shape, dtype=accum_dtype, device=g_inc.device
                    )
                    counts[ln] = 0
                grams[ln] += g_inc
                counts[ln] += x.shape[0]
            handles.append(module.register_forward_hook(_hook_full))

    n_seqs = 0
    for text in texts:
        if not text:
            continue
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_len)
        enc = {k: v.to(device) for k, v in enc.items()}
        model(**enc)
        n_seqs += 1

    for h in handles:
        h.remove()

    for name, G in grams.items():
        n = max(1, counts.get(name, 1))
        C = G / n
        if trace_normalize:
            d = C.shape[0]
            tr = torch.diagonal(C).sum().clamp_min(torch.finfo(C.dtype).tiny)
            C = C / tr * d
        grams[name] = C

    if not grams and not traces:
        raise RuntimeError("Collected no activation statistics — check that texts is non-empty")

    return GramArtifacts(
        grams=grams,
        traces=traces,
        counts=counts,
        in_dims=in_dims,
        out_dims=out_dims,
    )
