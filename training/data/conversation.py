"""Qwen2.5 / Qwen2.5-VL chat-template builder.

We produce a single tokenized sample with assistant-only supervision — loss is
taken only on assistant response tokens, not on system/user prompts.

Two paths:
  1. **Text-only**: ``tokenizer.apply_chat_template`` (works identically across
     Qwen text and VL tokenizers).
  2. **Image**: Qwen2.5-VL's processor expands a ``{"type": "image"}`` content
     item into the correct number of ``<|image_pad|>`` tokens based on the
     image's grid. We therefore go through the full processor for image
     samples, then locate assistant spans via the ``<|im_start|>assistant`` /
     ``<|im_end|>`` markers in the resulting token ids.

The collator (``merit.data.collator.DataCollatorForMERIT``) is the only direct
caller; downstream code should construct its samples via the collator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

IMAGE_TOKEN = "<image>"
IGNORE_INDEX = -100

_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"


@dataclass
class ChatSample:
    input_ids: torch.LongTensor
    labels: torch.LongTensor
    has_image: bool
    pixel_values: torch.Tensor | None = None
    image_grid_thw: torch.Tensor | None = None


def _normalize_role(role_raw: str) -> str:
    if role_raw in ("human", "user"):
        return "user"
    if role_raw in ("gpt", "assistant", "bot"):
        return "assistant"
    if role_raw in ("system",):
        return "system"
    raise ValueError(f"unknown role: {role_raw}")


def _strip_image_placeholder(text: str) -> str:
    """LLaVA-style ``<image>`` placeholders are absorbed by the structured
    ``{"type": "image"}`` content entry — remove them from the text so they
    don't appear as literal user tokens."""
    if not text:
        return text
    return text.replace(IMAGE_TOKEN, "").strip()


def _messages_to_qwen(
    messages: list[dict], *, has_image: bool
) -> list[dict[str, Any]]:
    """Convert LLaVA/ShareGPT messages to Qwen processor-friendly format.

    Attaches a single ``{"type": "image"}`` entry to the first user turn when
    ``has_image`` is True. Text-only messages use a plain string content so
    that the text-only tokenizer path works uniformly.
    """
    out: list[dict[str, Any]] = []
    image_attached = False
    for m in messages:
        role_raw = m.get("from") or m.get("role") or ""
        role = _normalize_role(role_raw)
        text = _strip_image_placeholder(m.get("value") or m.get("content") or "")
        if has_image and role == "user" and not image_attached:
            out.append(
                {
                    "role": role,
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": text},
                    ],
                }
            )
            image_attached = True
        else:
            out.append({"role": role, "content": text})
    return out


def _apply_chat_ids(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    """Return token ids from ``apply_chat_template`` across transformers versions.

    transformers 4.x returned a bare ``list[int]``; transformers 5.x returns a
    ``BatchEncoding`` whose subscript access returns nested encodings instead of
    the flat id list — hence the ``return_dict=False`` shim below.
    """
    if not messages:
        return []
    kwargs: dict = dict(
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_tensors=None,
    )
    try:
        out = tokenizer.apply_chat_template(messages, return_dict=False, **kwargs)
    except TypeError:
        out = tokenizer.apply_chat_template(messages, **kwargs)
    if isinstance(out, dict):
        out = out.get("input_ids", out)
    if out and isinstance(out[0], (list, tuple)):
        out = out[0]
    return list(out)


def _mask_assistant_only(
    input_ids: torch.Tensor, tokenizer: PreTrainedTokenizerBase
) -> torch.Tensor:
    """Build a label tensor that supervises only ``<|im_start|>assistant\n ... <|im_end|>`` spans."""
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    assistant_header = tokenizer.encode(
        f"{_IM_START}assistant\n", add_special_tokens=False
    )
    im_end_id = tokenizer.convert_tokens_to_ids(_IM_END)
    n = input_ids.numel()
    h = len(assistant_header)
    if h == 0 or im_end_id is None:
        return labels

    header_t = torch.tensor(assistant_header, dtype=input_ids.dtype)
    i = 0
    while i + h <= n:
        if torch.equal(input_ids[i : i + h], header_t):
            start = i + h
            j = start
            while j < n and input_ids[j].item() != im_end_id:
                j += 1
            if j < n:
                j += 1  # include closing <|im_end|>
            labels[start:j] = input_ids[start:j]
            i = j
        else:
            i += 1
    return labels


def build_qwen_chat(
    messages: list[dict],
    tokenizer: PreTrainedTokenizerBase,
    *,
    processor: Any | None = None,
    image: Any | None = None,
    max_length: int = 4096,
) -> ChatSample:
    """Build a single chat sample (input_ids + labels [+ pixel_values]).

    If ``image`` is provided, ``processor`` must be the full Qwen2.5-VL
    AutoProcessor (not just the image processor) so that ``<|image_pad|>``
    tokens expand correctly. Otherwise the tokenizer path is used.
    """
    has_image = image is not None
    qwen_msgs = _messages_to_qwen(messages, has_image=has_image)

    pixel_values: torch.Tensor | None = None
    image_grid_thw: torch.Tensor | None = None

    if has_image:
        if processor is None:
            raise ValueError("image sample requires a Qwen processor")
        text = processor.apply_chat_template(
            qwen_msgs, tokenize=False, add_generation_prompt=False
        )
        enc = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=False,
        )
        input_ids = enc["input_ids"][0].to(torch.long)
        pixel_values = enc["pixel_values"]
        image_grid_thw = enc["image_grid_thw"]
        labels = _mask_assistant_only(input_ids, tokenizer)
    else:
        full = _apply_chat_ids(tokenizer, qwen_msgs, add_generation_prompt=False)
        input_ids = torch.tensor(full, dtype=torch.long)
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        for i in range(len(qwen_msgs)):
            prefix_before = _apply_chat_ids(
                tokenizer,
                qwen_msgs[:i],
                add_generation_prompt=(qwen_msgs[i]["role"] == "assistant"),
            )
            prefix_after = _apply_chat_ids(
                tokenizer,
                qwen_msgs[: i + 1],
                add_generation_prompt=False,
            )
            if qwen_msgs[i]["role"] == "assistant":
                start = len(prefix_before)
                end = len(prefix_after)
                labels[start:end] = input_ids[start:end]

    if input_ids.numel() > max_length:
        # Refuse to truncate. Both options corrupt the sample:
        #   - left-trunc cuts the leading ``<|image_pad|>`` tokens, so the
        #     number-of-image-tokens vs image-feature count check inside
        #     ``Qwen2_5_VLForConditionalGeneration.forward`` fails and one
        #     rank exits with ValueError, NCCL-timing-out the whole job.
        #   - right-trunc can drop the assistant turn entirely, leaving the
        #     sample with all-IGNORE_INDEX labels → no gradient → NaN loss.
        # Instead raise and let the collator substitute a different sample.
        raise ValueError(
            f"sample length {input_ids.numel()} exceeds max_length {max_length}; "
            "skip and substitute"
        )

    return ChatSample(
        input_ids=input_ids,
        labels=labels,
        has_image=has_image,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )
