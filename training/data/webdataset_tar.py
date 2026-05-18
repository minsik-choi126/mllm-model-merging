"""Generic TAR-shard ("webdataset"-style) multimodal dataset loader.

Each source is a directory of TAR shards. A TAR shard contains paired files
keyed by a common stem, e.g.::

    00000000.json
    00000000.image       (or .jpg / .png / .webp)
    00000001.json
    00000001.image
    ...

The ``.json`` file holds the conversation in LLaVA/ShareGPT shape::

    {
      "conversations": [
        {"from": "human"|"user",  "value": "<image>\\nQuestion?"},
        {"from": "gpt"|"assistant", "value": "Answer"}
      ]
    }

The image file is the raw image bytes, decoded by Pillow at training time.

This loader is the TAR-shard counterpart to :mod:`merit.data.jsonl`. The two
share the manifest schema (``sources: [...]``) and per-sample output dict
(``{task_id, conversations, image_bytes, image_path}``) so the collator and
chat template builder are agnostic to the storage backend.

Manifest schema::

    {
      "sources": [
        {
          "task_id": "my_task",
          "tar_dir": "/path/to/shards/",       # contains *.tar shards
          "max_samples": 1000                   # optional per-source cap
        },
        ...
      ]
    }

Random access in TAR shards is supported via per-shard member offset tables
built once at construction time. ``__getitem__`` then opens the relevant TAR
file (with a per-process LRU cache) and seeks to the precomputed offset.
"""
from __future__ import annotations

import io
import json
import os
import tarfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from torch.utils.data import Dataset

from training.data.registry import register_dataset
from training.utils.io import safe_json_load
from training.utils.logging import get_logger

logger = get_logger(__name__)

# Recognised image-file extensions inside a TAR shard. The first match wins.
_IMAGE_EXTS = ("image", "jpg", "jpeg", "png", "webp")


@dataclass
class WebdatasetTarConfig:
    # Manifest JSON path. Named ``root`` so it binds to the unified
    # ``DataArgs.root`` field that MERIT trainers pass through.
    root: str
    split: str = "train"
    max_samples: int | None = None
    allowed_task_ids: list[str] | None = None
    allowed_source_ids: list[str] | None = None


class _TarLRU:
    """Tiny LRU of open TarFile handles. Per-process; not thread-safe."""

    def __init__(self, capacity: int = 8) -> None:
        self.capacity = capacity
        self._cache: "OrderedDict[str, tarfile.TarFile]" = OrderedDict()

    def get(self, path: str) -> tarfile.TarFile:
        tf = self._cache.get(path)
        if tf is not None:
            self._cache.move_to_end(path)
            return tf
        tf = tarfile.open(path, "r:")
        self._cache[path] = tf
        if len(self._cache) > self.capacity:
            _, evicted = self._cache.popitem(last=False)
            try:
                evicted.close()
            except Exception:
                pass
        return tf

    def close_all(self) -> None:
        for tf in self._cache.values():
            try:
                tf.close()
            except Exception:
                pass
        self._cache.clear()


def _index_tar(tar_path: Path) -> dict[str, dict[str, tarfile.TarInfo]]:
    """Open a TAR once and return ``{base_key: {ext: TarInfo}}`` for every
    member. The TarInfo objects retain offset/size so subsequent extractions
    are O(1) seeks.
    """
    by_key: dict[str, dict[str, tarfile.TarInfo]] = {}
    with tarfile.open(tar_path, "r:") as tf:
        for info in tf:
            if not info.isfile():
                continue
            name = info.name
            stem, _, ext = name.rpartition(".")
            if not stem:
                continue
            by_key.setdefault(stem, {})[ext] = info
    return by_key


def _pick_image_info(by_ext: dict[str, tarfile.TarInfo]) -> tarfile.TarInfo | None:
    for ext in _IMAGE_EXTS:
        if ext in by_ext:
            return by_ext[ext]
    return None


def _read_member(tar: tarfile.TarFile, info: tarfile.TarInfo) -> bytes:
    f = tar.extractfile(info)
    if f is None:
        return b""
    try:
        return f.read()
    finally:
        f.close()


class WebdatasetTarDataset(Dataset):
    """Builds an index over all TAR shards listed in the manifest.

    The index is a flat list of
    ``[(task_id, tar_path, base_key, json_info, image_info_or_None), ...]``
    tuples. Each ``__getitem__`` opens the relevant TAR (LRU-cached), reads the
    JSON + image bytes, and emits the standard MERIT sample dict.
    """

    def __init__(self, cfg: WebdatasetTarConfig):
        self.cfg = cfg
        self._index: list[
            tuple[str, str, str, tarfile.TarInfo, tarfile.TarInfo | None]
        ] = []
        self._tar_cache = _TarLRU(capacity=8)

        manifest = safe_json_load(Path(cfg.root))
        allowed_ids = cfg.allowed_task_ids or cfg.allowed_source_ids
        allowed = set(allowed_ids) if allowed_ids else None

        for src in manifest["sources"]:
            tid = src["task_id"]
            if allowed is not None and tid not in allowed:
                continue
            tar_dir = Path(src["tar_dir"])
            limit = src.get("max_samples")

            shards = sorted(tar_dir.glob("*.tar"))
            if not shards:
                logger.warning(f"[webdataset_tar] no .tar shards under {tar_dir}")
                continue

            kept = 0
            for shard in shards:
                idx = _index_tar(shard)
                for stem, by_ext in idx.items():
                    if "json" not in by_ext:
                        continue
                    img_info = _pick_image_info(by_ext)
                    self._index.append(
                        (tid, str(shard), stem, by_ext["json"], img_info)
                    )
                    kept += 1
                    if limit is not None and kept >= limit:
                        break
                if limit is not None and kept >= limit:
                    break

        if cfg.max_samples is not None:
            self._index = self._index[: cfg.max_samples]

        logger.info(
            f"[webdataset_tar] built index over "
            f"{len(set(t for t, *_ in self._index))} sources, "
            f"{len(self._index)} samples"
        )

    def __len__(self) -> int:
        return len(self._index)

    def _load_json(self, raw: bytes) -> dict[str, Any]:
        return json.loads(raw.decode("utf-8"))

    def _materialise(
        self, raw_json: dict[str, Any], task_id: str
    ) -> dict[str, Any] | None:
        """Hook for subclasses that adapt non-LLaVA JSON schemas. Default
        behaviour expects the LLaVA/ShareGPT shape
        ``{conversations: [{from, value}], ...}`` and yields it unchanged.

        ``task_id`` is provided so subclasses can apply per-source policy
        (e.g. multi-turn sample counts) without re-parsing the manifest.
        """
        del task_id  # unused in the default implementation
        if not raw_json.get("conversations"):
            return None
        return {"conversations": raw_json["conversations"]}

    def __getitem__(self, idx: int) -> dict[str, Any]:
        tid, tar_path, stem, json_info, image_info = self._index[idx]
        tar = self._tar_cache.get(tar_path)

        json_bytes = _read_member(tar, json_info)
        record = self._load_json(json_bytes)
        materialised = self._materialise(record, tid)
        if materialised is None:
            # Defensive fallback — emit a dummy single-turn so the trainer doesn't
            # crash; the sample will be dropped at the loss-mask stage anyway.
            materialised = {"conversations": [
                {"from": "user", "value": ""},
                {"from": "gpt", "value": ""},
            ]}

        image_bytes: bytes | None = None
        if image_info is not None:
            image_bytes = _read_member(tar, image_info)
            if not image_bytes:
                image_bytes = None

        return {
            "task_id": tid,
            "conversations": materialised["conversations"],
            "image_bytes": image_bytes,
            "image_path": f"{os.path.basename(tar_path)}#{stem}",
        }

    @property
    def task_ids(self) -> list[str]:
        return sorted({tid for tid, *_ in self._index})

    def __del__(self) -> None:
        try:
            self._tar_cache.close_all()
        except Exception:
            pass


@register_dataset("webdataset_tar")
def _build(**kwargs) -> WebdatasetTarDataset:
    cfg = WebdatasetTarConfig(**kwargs)
    return WebdatasetTarDataset(cfg)
