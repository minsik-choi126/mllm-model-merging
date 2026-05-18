"""Dataset registry: ``name -> (loader callable)``.

Usage:

    from training.data import get_dataset
    ds = get_dataset("vflan136", split="train", cfg=cfg)

Built-in names (registered by the individual modules on import):
  - ``vflan136`` : 136 Vision-FLAN tasks (3B controlled study)
  - ``mix176``   : 176-source 1.6 M multimodal mixture (7B scale)
  - ``flan66``   : 66-task text-only FLAN (text-only experiments)
"""
from __future__ import annotations

from typing import Any, Callable, Dict

_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_dataset(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        if name in _REGISTRY:
            raise ValueError(f"dataset already registered: {name}")
        _REGISTRY[name] = fn
        return fn

    return deco


def get_dataset(name: str, **kwargs: Any) -> Any:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown dataset '{name}'. registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](**kwargs)


def list_datasets() -> list[str]:
    return sorted(_REGISTRY)
