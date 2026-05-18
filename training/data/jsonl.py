"""Generic JSONL multimodal dataset loader.

Reads a manifest of JSONL sources and yields per-sample dicts compatible with
:class:`merit.data.collator.DataCollatorForMERIT`.

Sample schema (per JSONL line)::

    {
      "conversations": [
        {"from": "human"|"user",  "value": "<image>\\nQuestion?"},
        {"from": "gpt"|"assistant", "value": "Answer"}
      ],
      "image": "path/to/image.jpg"     # optional; text-only when absent
    }

Image paths support two forms:

* **Plain file** (``foo.jpg``) — read directly from disk.
* **Zip-archived** (``bundle.zip#inner.jpg``) — useful when datasets pack many
  small images into per-record zip archives. The substring before ``#`` is the
  zip path on disk; the substring after ``#`` selects an entry inside the zip.

Manifest schema::

    {
      "sources": [
        {
          "task_id": "my_task",
          "jsonl":   "/path/to/my_task.jsonl",
          "image_root": "/path/to/my_task/images",   # optional; defaults to
                                                       # the JSONL's parent dir
          "max_samples": 1000                          # optional per-source cap
        },
        ...
      ]
    }

The collator-side schema (``task_id``, ``conversations``, ``image_bytes``,
``image_path``) is identical to :mod:`merit.data.mix176`, so the collator and
chat builder need no changes.
"""
from __future__ import annotations

import json
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from torch.utils.data import Dataset

from training.data.registry import register_dataset
from training.utils.io import safe_json_load


@dataclass
class JsonlConfig:
    # ``root`` is the manifest JSON path. Named ``root`` so it binds to the
    # unified ``DataArgs.root`` field that MERIT trainers pass through.
    root: str
    split: str = "train"
    max_samples: int | None = None
    allowed_task_ids: list[str] | None = None
    allowed_source_ids: list[str] | None = None


def _resolve_image_path(image_root: Path, raw: str) -> tuple[Path, str | None]:
    """Resolve an image path. Returns (file_path, inner_path_in_zip_or_None)."""
    if "#" in raw:
        outer, inner = raw.split("#", 1)
        return image_root / outer, inner
    return image_root / raw, None


def _read_image_bytes(image_root: Path, raw: str) -> bytes:
    p, inner = _resolve_image_path(image_root, raw)
    if inner is None:
        return p.read_bytes()
    with zipfile.ZipFile(p, "r") as zf:
        return zf.read(inner)


def iter_jsonl_records(jsonl_path: Path) -> Iterable[dict[str, Any]]:
    """Yield JSON records from a file. Two formats are auto-detected:

    * **JSONL** (one JSON object per line) — streaming, no full file buffered.
    * **JSON array** (single ``[ {...}, {...}, ... ]`` blob, e.g. LLaVA-1.5's
      ``llava_v1_5_mix665k.json``) — loaded once into memory.

    Blank lines and individually unparseable lines are silently skipped in the
    JSONL path; in the JSON-array path the whole file must parse.
    """
    with jsonl_path.open("rb") as f:
        head = f.read(2048)
    stripped = head.lstrip()
    is_json_array = stripped.startswith(b"[")

    if is_json_array:
        with jsonl_path.open("r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{jsonl_path}: top-level JSON is not a list")
        for record in data:
            if isinstance(record, dict):
                yield record
        return

    with jsonl_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


class JsonlDataset(Dataset):
    """Iterates over the manifest's JSONL files at construction time and
    materialises a flat ``[(task_id, image_root, sample_dict), ...]`` index.

    ``sample_dict`` is the raw JSONL row — the dataset preserves whatever
    fields are present beyond ``conversations`` / ``image`` so subclasses or
    callers that add side-channels (e.g. token counts) are not lossy.
    """

    def __init__(self, cfg: JsonlConfig):
        self.cfg = cfg
        self._samples: list[tuple[str, Path, dict]] = []

        manifest = safe_json_load(Path(cfg.root))
        allowed_ids = cfg.allowed_task_ids or cfg.allowed_source_ids
        allowed = set(allowed_ids) if allowed_ids else None

        for src in manifest["sources"]:
            tid = src["task_id"]
            if allowed is not None and tid not in allowed:
                continue
            jsonl_path = Path(src["jsonl"])
            image_root = Path(src.get("image_root") or jsonl_path.parent)
            limit = src.get("max_samples")

            kept = 0
            for row in self._iter_source(jsonl_path):
                self._samples.append((tid, image_root, row))
                kept += 1
                if limit is not None and kept >= limit:
                    break

        if cfg.max_samples is not None:
            self._samples = self._samples[: cfg.max_samples]

    # Hook for subclasses that need to transform records (e.g. site-local
    # JSONL formats that need to be normalised to {conversations, image}).
    def _iter_source(self, jsonl_path: Path) -> Iterable[dict[str, Any]]:
        for row in iter_jsonl_records(jsonl_path):
            if self._is_valid_sample(row):
                yield row

    @staticmethod
    def _is_valid_sample(row: dict[str, Any]) -> bool:
        convos = row.get("conversations")
        if not convos:
            return False
        return any(
            (m.get("from") in ("gpt", "assistant", "bot")
             or m.get("role") in ("assistant",))
            for m in convos
        )

    def __len__(self) -> int:
        return len(self._samples)

    def _build_sample(self, idx: int) -> dict[str, Any]:
        """Materialise one sample. Raises on any decoding failure so the
        outer ``__getitem__`` retry loop can pick a different sample."""
        tid, image_root, row = self._samples[idx]
        image_path = row.get("image")
        image_bytes: bytes | None = None
        if image_path is not None:
            image_bytes = _read_image_bytes(image_root, image_path)
            # Verify the image is decodable end-to-end. If a JPEG is
            # truncated or a PNG is malformed, this raises here rather than
            # downstream in the model forward, where it could desync NCCL.
            from PIL import Image
            import io as _io
            Image.open(_io.BytesIO(image_bytes)).load()

        return {
            "task_id": tid,
            "conversations": row["conversations"],
            "image_bytes": image_bytes,
            "image_path": image_path,
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Robust fetch: if a particular sample fails to load (corrupt image,
        # missing file, malformed JSON record), fall back to a random other
        # sample from the same dataset.  Prevents a single bad sample from
        # desyncing NCCL in distributed training (one rank stuck retrying
        # while the other ranks march on into their next collective op).
        n = len(self._samples)
        if n == 0:
            raise IndexError("empty dataset")
        last_err: Exception | None = None
        rng = random.Random(idx)
        for attempt in range(8):
            cur_idx = idx if attempt == 0 else rng.randint(0, n - 1)
            try:
                return self._build_sample(cur_idx)
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue
        raise RuntimeError(
            f"dataset fallback exhausted at idx={idx} after 8 attempts: {last_err}"
        )

    @property
    def task_ids(self) -> list[str]:
        return sorted({tid for tid, _, _ in self._samples})


@register_dataset("jsonl")
def _build(**kwargs) -> JsonlDataset:
    cfg = JsonlConfig(**kwargs)
    return JsonlDataset(cfg)
