"""TAR-shard adapter for QA-pair multimodal datasets.

A common multimodal pre-training format ships TAR shards whose per-sample
JSON looks like::

    {
      "info": {...},                                    # arbitrary, ignored
      "qa_pairs_en": [["<image>\\nQuestion?", "Answer"], ["...", "..."]],
      "qa_pairs":    [["<image>\\nQuestion?", "Answer"], ...],
      "captions_en": ["..."],                           # ignored
      "text": "...", "words": [...], "lines": [...]     # ignored
    }

This adapter transforms each row into the LLaVA/ShareGPT
``{conversations: [{from, value}, ...]}`` shape expected by
:class:`merit.data.webdataset_tar.WebdatasetTarDataset`, then delegates the
rest (TAR indexing, image extraction, LRU caching, manifest scanning) to
the base class.

The English QA pairs (``qa_pairs_en``) are preferred when present; we fall
back to the language-agnostic ``qa_pairs``.

Manifest extension: each source may carry ``multiturn_n_samples`` (int) to
control how many of the available QA pairs are concatenated into a single
multi-turn conversation::

    {
      "sources": [
        {
          "task_id": "my_qa_shards",
          "tar_dir": "/path/to/shards/",
          "multiturn_n_samples": 4    # randomly pick 4 of the QA pairs
        },
        ...
      ]
    }

``0`` or ``1`` means single-turn (only the first QA pair is used).

Register name: ``qa_pairs_webdataset_tar``.
"""
from __future__ import annotations

import random
from typing import Any

from training.data.registry import register_dataset
from training.data.webdataset_tar import (
    WebdatasetTarConfig,
    WebdatasetTarDataset,
)
from training.utils.io import safe_json_load


def _pick_qa_pairs(record: dict[str, Any]) -> list[list[str]]:
    """Return the best-available QA pair list. English preferred; fall back to
    the language-agnostic ``qa_pairs`` field."""
    for key in ("qa_pairs_en", "qa_pairs"):
        v = record.get(key)
        if v:
            return v
    return []


def _qa_to_conversations(
    record: dict[str, Any],
    *,
    multiturn_n_samples: int = 0,
    rng: random.Random | None = None,
) -> list[dict[str, str]]:
    """Convert a QA-pair record into a LLaVA/ShareGPT conversation list.

    The first user turn keeps the ``<image>`` placeholder; subsequent turns
    get it stripped (the collator only attaches one image to the first user
    turn).
    """
    pairs = _pick_qa_pairs(record)
    if not pairs:
        return []

    if multiturn_n_samples and multiturn_n_samples >= 2 and len(pairs) > multiturn_n_samples:
        rng = rng or random
        # Sample without replacement, preserve order.
        idxs = sorted(rng.sample(range(len(pairs)), multiturn_n_samples))
        pairs = [pairs[i] for i in idxs]

    convos: list[dict[str, str]] = []
    image_placed = False
    for q, a in pairs:
        q_text = q
        if "<image>" in q_text:
            if image_placed:
                q_text = q_text.replace("<image>", "").strip()
            else:
                image_placed = True
        convos.append({"from": "human", "value": q_text})
        convos.append({"from": "gpt", "value": a})
    return convos


class QAPairsWebdatasetDataset(WebdatasetTarDataset):
    """Same as :class:`WebdatasetTarDataset` but transforms QA-pair JSON
    schemas and supports a per-source ``multiturn_n_samples`` override."""

    def __init__(self, cfg: WebdatasetTarConfig):
        # Re-load the manifest to capture the multiturn knob per source. The
        # base class already parses the same manifest for indexing — we only
        # need the extra ``multiturn_n_samples`` field, so a second cheap pass
        # is fine.
        manifest = safe_json_load(cfg.root)
        self._multiturn_by_task: dict[str, int] = {
            src["task_id"]: int(src.get("multiturn_n_samples") or 0)
            for src in manifest["sources"]
        }
        self._rng = random.Random(0)
        super().__init__(cfg)

    def _materialise(
        self, raw_json: dict[str, Any], task_id: str
    ) -> dict[str, Any] | None:
        n = self._multiturn_by_task.get(task_id, 0)
        convos = _qa_to_conversations(raw_json, multiturn_n_samples=n, rng=self._rng)
        if not convos:
            return None
        return {"conversations": convos}


@register_dataset("qa_pairs_webdataset_tar")
def _build(**kwargs) -> QAPairsWebdatasetDataset:
    cfg = WebdatasetTarConfig(**kwargs)
    return QAPairsWebdatasetDataset(cfg)
