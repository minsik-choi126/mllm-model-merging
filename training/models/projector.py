"""Multimodal projector builders.

MERIT's training pipeline currently uses the LLaVA-standard two-layer GELU MLP
projector from the vision encoder hidden size to the LM hidden size. This file
exists as its own module so that the Stage 1 trainer can reach in and freeze /
tune just the projector parameters without importing the full model.
"""
from __future__ import annotations

import torch
from torch import nn


def build_mlp2x_gelu(in_dim: int, out_dim: int, *, dtype: torch.dtype = torch.float32) -> nn.Module:
    """Two-layer GELU MLP: Linear -> GELU -> Linear.

    Used to bridge vision encoder features (``in_dim``) to the LM hidden size
    (``out_dim``). Compatible with the ``mm_projector_type=mlp2x_gelu`` LLaVA
    convention so that existing Qwen2.5-VL checkpoints load cleanly.
    """
    mod = nn.Sequential(
        nn.Linear(in_dim, out_dim, dtype=dtype),
        nn.GELU(),
        nn.Linear(out_dim, out_dim, dtype=dtype),
    )
    return mod


def projector_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Collect projector parameters by name pattern.

    Qwen2.5-VL exposes them as ``model.visual.merger.*`` or
    ``multi_modal_projector.*`` depending on the transformers version; we scan
    both.
    """
    keys = []
    for name, param in model.named_parameters():
        lname = name.lower()
        if "multi_modal_projector" in lname or "mm_projector" in lname or "visual.merger" in lname:
            keys.append((name, param))
    return [p for _, p in keys]
