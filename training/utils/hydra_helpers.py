"""Lightweight YAML config loader with `defaults:` inheritance.

We avoid the full Hydra dependency but support:
  - `defaults: [foo.yaml, ../_base_/bar.yaml]` merging (later files win)
  - path resolution relative to the including file
  - deep-merge semantics (dicts merge, everything else is overwritten)
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and resolve `defaults: [...]` chains.

    Cycle detection is handled by tracking visited absolute paths.
    """
    return _load(Path(path).resolve(), visited=set())


def _load(path: Path, visited: set[Path]) -> dict[str, Any]:
    if path in visited:
        raise ValueError(f"cyclic config inheritance detected at {path}")
    visited = visited | {path}

    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    defaults = raw.pop("defaults", []) or []
    merged: dict[str, Any] = {}
    for rel in defaults:
        base_path = (path.parent / rel).resolve()
        merged = _deep_merge(merged, _load(base_path, visited))
    return _deep_merge(merged, raw)


def resolve_defaults(cfg: dict[str, Any], key: str, default: Any = None) -> Any:
    """dotted-key getter (e.g. `train.learning_rate`)."""
    cur: Any = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
