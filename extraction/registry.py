"""Known VLM↔LLM model pairs, loaded from `models.yaml`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelPair:
    """A VLM and its base LLM, with the text-backbone shape parameters."""
    key: str
    vlm: str
    llm: str
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    intermediate_size: int
    role: str = ""
    base: Optional[str] = None
    extracted_lm: Optional[str] = None
    notes: str = ""

    @property
    def uses_gqa(self) -> bool:
        return self.num_kv_heads < self.num_heads


def _default_config_path() -> Path:
    return Path(__file__).parent / "models.yaml"


def load_registry(config_path: Optional[str] = None) -> dict[str, ModelPair]:
    """Load all model pairs from the YAML config."""
    path = Path(config_path) if config_path else _default_config_path()
    with open(path) as f:
        cfg = yaml.safe_load(f)

    pairs: dict[str, ModelPair] = {}
    for key, info in cfg["model_pairs"].items():
        pairs[key] = ModelPair(
            key=key,
            base=info.get("base"),
            vlm=info["vlm"],
            llm=info["llm"],
            num_layers=info["num_layers"],
            num_heads=info["num_heads"],
            num_kv_heads=info["num_kv_heads"],
            head_dim=info["head_dim"],
            hidden_size=info["hidden_size"],
            intermediate_size=info["intermediate_size"],
            role=info.get("role", ""),
            extracted_lm=info.get("extracted_lm"),
            notes=info.get("notes", ""),
        )
    return pairs


def get_pair(key: str, config_path: Optional[str] = None) -> ModelPair:
    """Look up a single pair by key."""
    registry = load_registry(config_path)
    if key not in registry:
        raise KeyError(
            f"Unknown model pair '{key}'. Available: {sorted(registry.keys())}"
        )
    return registry[key]
