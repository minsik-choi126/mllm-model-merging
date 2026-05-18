"""176-source 1.6 M multimodal mixture loader for 7B experiments.

Expected layout::

    <root>/
      manifest.json           # {"sources": [{"source_id": "cauldron_ocrvqa", "json": "cauldron_ocrvqa.json"}, ...]}
      images.lmdb/
      sources/
        cauldron_ocrvqa.json
        lvis_instruct4v.json
        ...
      token_counts.parquet    # per-source token counts for the merge step

Sample format matches :mod:`merit.data.vflan136`, but each item carries a
``source_id`` instead of ``task_id``. Both fields are unified as ``task_id`` at
the collator boundary so downstream code doesn't need to branch.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from training.data.lmdb_dataset import LMDBReader
from training.data.registry import register_dataset
from training.utils.io import safe_json_load


@dataclass
class Mix176Config:
    root: str
    split: str = "train"
    max_samples: int | None = None
    # Alias ``allowed_source_ids`` as ``allowed_task_ids`` so the registry's
    # unified kwarg from the branch trainer (which came from group assignment
    # ``task_ids``) works for both Vision-FLAN and Mix-176 datasets.
    allowed_task_ids: list[str] | None = None
    allowed_source_ids: list[str] | None = None


class Mix176Dataset(Dataset):
    def __init__(self, cfg: Mix176Config):
        self.cfg = cfg
        root = Path(cfg.root)
        self._root = root
        manifest = safe_json_load(root / "manifest.json")
        lmdb_path = root / "images.lmdb"
        self._reader = LMDBReader(lmdb_path) if lmdb_path.exists() else None

        allowed_ids = cfg.allowed_task_ids or cfg.allowed_source_ids
        allowed = set(allowed_ids) if allowed_ids else None
        self._samples: list[tuple[str, dict]] = []
        for source_entry in manifest["sources"]:
            sid = source_entry["source_id"]
            if allowed is not None and sid not in allowed:
                continue
            rows = safe_json_load(root / "sources" / source_entry["json"])
            for row in rows:
                self._samples.append((sid, row))

        if cfg.max_samples is not None:
            self._samples = self._samples[: cfg.max_samples]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sid, row = self._samples[idx]
        image_path: str | None = row.get("image")
        image_bytes: bytes | None = None
        if image_path is not None:
            if self._reader is not None:
                image_bytes = self._reader.get(image_path)
            else:
                image_bytes = (self._root / image_path).resolve().read_bytes()

        return {
            "task_id": sid,            # unified with vflan136 for downstream code
            "conversations": row["conversations"],
            "image_bytes": image_bytes,
            "image_path": image_path,
        }

    @property
    def task_ids(self) -> list[str]:
        return sorted({sid for sid, _ in self._samples})


@register_dataset("mix176")
def _build(**kwargs) -> Mix176Dataset:
    cfg = Mix176Config(**kwargs)
    return Mix176Dataset(cfg)
