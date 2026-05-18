"""Vision-FLAN 136-task loader for the 3B controlled study.

Expected on-disk layout::

    <root>/
      manifest.json           # {"tasks": [{"task_id": "...", "json": "task_0.json", "image_subdir": "images/"}, ...]}
      images.lmdb/            # built with `merit.data.lmdb_builder`
      tasks/
        task_0.json           # LLaVA-style list of {"image": "...", "conversations": [...]}
        task_1.json
        ...

Each sample returned from ``__getitem__`` is a dict:

    {
        "task_id":       str,
        "conversations": list[dict],    # LLaVA/ShareGPT style
        "image_bytes":   bytes | None,  # raw JPEG/PNG
        "image_path":    str | None,
    }

The collator (``merit.data.collator.DataCollatorForMERIT``) turns this dict into
tokenized + imaged tensors using the tokenizer + image processor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from training.data.lmdb_dataset import LMDBReader
from training.data.registry import register_dataset
from training.utils.io import safe_json_load


@dataclass
class VFlan136Config:
    root: str                          # dataset root
    split: str = "train"               # currently only "train"
    max_samples: int | None = None     # for smoke tests
    allowed_task_ids: list[str] | None = None  # subset selection (e.g. a branch group)


class VFlan136Dataset(Dataset):
    def __init__(self, cfg: VFlan136Config):
        self.cfg = cfg
        root = Path(cfg.root)
        manifest = safe_json_load(root / "manifest.json")
        lmdb_path = root / "images.lmdb"
        self._reader = LMDBReader(lmdb_path) if lmdb_path.exists() else None
        self._root = root

        allowed = set(cfg.allowed_task_ids) if cfg.allowed_task_ids else None

        self._samples: list[tuple[str, dict]] = []  # (task_id, sample)
        for task_entry in manifest["tasks"]:
            tid = task_entry["task_id"]
            if allowed is not None and tid not in allowed:
                continue
            task_json = safe_json_load(root / "tasks" / task_entry["json"])
            for row in task_json:
                self._samples.append((tid, row))

        if cfg.max_samples is not None:
            self._samples = self._samples[: cfg.max_samples]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        tid, row = self._samples[idx]
        image_bytes: bytes | None = None
        image_path: str | None = row.get("image")
        if image_path is not None:
            if self._reader is not None:
                image_bytes = self._reader.get(image_path)
            else:
                # fallback: read from disk
                p = (self._root / image_path).resolve()
                image_bytes = p.read_bytes()

        return {
            "task_id": tid,
            "conversations": row["conversations"],
            "image_bytes": image_bytes,
            "image_path": image_path,
        }

    @property
    def task_ids(self) -> list[str]:
        return sorted({tid for tid, _ in self._samples})


@register_dataset("vflan136")
def _build(**kwargs) -> VFlan136Dataset:
    cfg = VFlan136Config(**kwargs)
    return VFlan136Dataset(cfg)
