"""JSONL adapter for OpenAI Chat Completions-style records.

Many in-house and public chat datasets use the OpenAI Chat Completions
schema, where each line is a single example with a ``messages`` list whose
entries carry ``role`` (``system``/``user``/``assistant``) and ``content``,
plus an optional ``image_urls`` list for multimodal turns. This adapter
transforms that shape into the LLaVA/ShareGPT
``{conversations, image}`` shape that :class:`merit.data.jsonl.JsonlDataset`
consumes, then delegates the rest (manifest scanning, ``zip#inner.jpg``
resolution, image-byte loading) to the public loader.

Expected per-line record::

    {
      "messages": [
        {"role": "system",    "content": "You are a helpful ..."},
        {"role": "user",      "content": "<|image|>What's in this image?",
                              "image_urls": ["images/foo.zip#inner.jpg"]},
        {"role": "assistant", "content": "..."}
      ],
      "meta": {...},                 # any additional fields are ignored
      "oid": "..."
    }

Only single-image, non-video samples are kept; multi-image and video rows
are dropped (return ``None``). The first ``<|image|>`` placeholder in the
first user turn is rewritten to ``<image>``; trailing image / video tag
sentinels in subsequent turns are stripped.

Register name: ``openai_chat_jsonl``. Use it in a manifest like::

    {
      "data": {
        "name": "openai_chat_jsonl",
        "root": "configs/data/your_chat_jsonl.json"
      }
    }
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from training.data.jsonl import (
    JsonlConfig,
    JsonlDataset,
    iter_jsonl_records,
)
from training.data.registry import register_dataset

_IMAGE_TAG_IN = "<|image|>"
_VIDEO_TAG_IN = "<|video|>"
_IMAGE_TAG_OUT = "<image>"


def _to_generic(row: dict[str, Any]) -> dict[str, Any] | None:
    """Convert one OpenAI-chat record to ``{conversations, image}``.

    Drops samples that are video, multi-image, or missing an assistant turn.
    """
    convos: list[dict[str, str]] = []
    image_path: str | None = None
    seen_images = 0

    for m in row.get("messages") or []:
        role = m.get("role")
        if role == "system":
            content = (m.get("content") or "").strip()
            if not content:
                continue
            convos.append({"from": "system", "value": content})
            continue

        if role not in ("user", "assistant"):
            continue

        text = m.get("content") or ""
        urls = m.get("image_urls") or []
        videos = m.get("video_urls") or []

        if videos:
            return None
        if len(urls) > 1:
            return None
        if urls:
            seen_images += 1
            if seen_images > 1:
                return None
            image_path = urls[0]
            text = text.replace(_IMAGE_TAG_IN, _IMAGE_TAG_OUT)
        else:
            text = text.replace(_IMAGE_TAG_IN, "").replace(_VIDEO_TAG_IN, "").strip()

        convos.append(
            {"from": "human" if role == "user" else "gpt", "value": text}
        )

    if not any(c.get("from") in ("gpt", "assistant") for c in convos):
        return None

    out: dict[str, Any] = {"conversations": convos}
    if image_path is not None:
        out["image"] = image_path
    return out


class OpenAIChatJsonlDataset(JsonlDataset):
    """Same as :class:`JsonlDataset` but transforms OpenAI-chat rows on the fly."""

    def _iter_source(self, jsonl_path: Path) -> Iterable[dict[str, Any]]:
        for row in iter_jsonl_records(jsonl_path):
            generic = _to_generic(row)
            if generic is not None:
                yield generic


@register_dataset("openai_chat_jsonl")
def _build(**kwargs) -> OpenAIChatJsonlDataset:
    cfg = JsonlConfig(**kwargs)
    return OpenAIChatJsonlDataset(cfg)
