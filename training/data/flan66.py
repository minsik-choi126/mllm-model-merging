"""66-task text-only FLAN loader used by the text-only experiments.

No image column, no LMDB reader. Sample format::

    {
        "task_id":       str,
        "conversations": list[dict],
        "image_bytes":   None,
        "image_path":    None,
    }
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from training.data.registry import register_dataset
from training.utils.io import safe_json_load


@dataclass
class Flan66Config:
    root: str
    split: str = "train"
    max_samples: int | None = None
    allowed_task_ids: list[str] | None = None


class Flan66Dataset(Dataset):
    def __init__(self, cfg: Flan66Config):
        self.cfg = cfg
        root = Path(cfg.root)
        manifest = safe_json_load(root / "manifest.json")
        allowed = set(cfg.allowed_task_ids) if cfg.allowed_task_ids else None

        self._samples: list[tuple[str, dict]] = []
        for task_entry in manifest["tasks"]:
            tid = task_entry["task_id"]
            if allowed is not None and tid not in allowed:
                continue
            rows = safe_json_load(root / "tasks" / task_entry["json"])
            for row in rows:
                self._samples.append((tid, row))

        if cfg.max_samples is not None:
            self._samples = self._samples[: cfg.max_samples]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        tid, row = self._samples[idx]
        return {
            "task_id": tid,
            "conversations": row["conversations"],
            "image_bytes": None,
            "image_path": None,
        }

    @property
    def task_ids(self) -> list[str]:
        return sorted({tid for tid, _ in self._samples})


@register_dataset("flan66")
def _build(**kwargs) -> Flan66Dataset:
    cfg = Flan66Config(**kwargs)
    return Flan66Dataset(cfg)
