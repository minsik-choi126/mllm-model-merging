"""Data collator for MERIT training.

Responsibilities:
  1. Build Qwen chat-template inputs via :func:`merit.data.conversation.build_qwen_chat`.
     For image samples, the full Qwen processor expands ``<|image_pad|>`` tokens.
  2. Pad ``input_ids`` / ``labels`` to the batch max length; labels pad with -100.
  3. Concatenate per-image ``pixel_values`` / ``image_grid_thw`` across the batch
     (Qwen2.5-VL packs all patches into a single flat tensor and relies on
     ``image_grid_thw`` to split them back out).
  4. Propagate ``task_ids`` through to the trainer for logging / grouping.

Handles mixed batches (some samples text-only, some multimodal).
"""
from __future__ import annotations

import io
import random
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from training.data.conversation import IGNORE_INDEX, build_qwen_chat
from training.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DataCollatorForMERIT:
    tokenizer: Any                     # PreTrainedTokenizerBase
    processor: Any | None = None       # full AutoProcessor (Qwen2.5-VL); None for text-only
    max_length: int = 4096
    # Back-compat alias: some callers still pass image_processor=...
    image_processor: Any | None = None

    def __post_init__(self) -> None:
        # Accept either ``processor=`` or ``image_processor=`` for back-compat.
        # If a bare image-processor is passed (no chat template), we fall back
        # to the text-only path and drop image bytes downstream.
        if self.processor is None and self.image_processor is not None:
            if hasattr(self.image_processor, "apply_chat_template") and hasattr(
                self.image_processor, "image_processor"
            ):
                self.processor = self.image_processor

    def __call__(self, batch: list[dict]) -> dict[str, Any]:
        input_ids_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []
        pixel_values_list: list[torch.Tensor] = []
        grid_thw_list: list[torch.Tensor] = []
        task_ids: list[str] = []

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        # First pass: build a chat for each sample. If build_qwen_chat raises
        # (over-length sample, malformed conversation, image-decode failure,
        # …), substitute with a *successfully built* sibling from the same
        # batch. Two reasons for this design over a 1-token dummy:
        #   1. batch shape (input_ids, pixel_values) stays consistent across
        #      ranks, avoiding the NumelIn-1 vs NumelIn-N NCCL desync seen
        #      when one rank gets a near-empty fast-path sample.
        #   2. forward+backward wall-clock per rank stays comparable, so
        #      ranks finish their step in approximate lockstep instead of
        #      drifting into the next collective op early.
        # If no sibling has been built yet (e.g. the very first sample
        # raises) we use a 1-token IGNORE pad as last resort.
        successful_chats: list[Any] = []
        successful_image_bytes: list[bytes | None] = []
        last_resort = type(
            "ChatSample",
            (),
            dict(
                input_ids=torch.tensor([pad_id], dtype=torch.long),
                labels=torch.full((1,), IGNORE_INDEX, dtype=torch.long),
                has_image=False,
                pixel_values=None,
                image_grid_thw=None,
            ),
        )()

        for sample in batch:
            convos = sample["conversations"]
            img_bytes = sample.get("image_bytes")
            img = None
            if img_bytes is not None and self.processor is not None:
                try:
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                except Exception:
                    img = None

            chat = None
            try:
                chat = build_qwen_chat(
                    convos,
                    self.tokenizer,
                    processor=self.processor,
                    image=img,
                    max_length=self.max_length,
                )
                successful_chats.append(chat)
                successful_image_bytes.append(img_bytes)
            except Exception as e:
                if successful_chats:
                    # Substitute with a random sample we've already built —
                    # same batch, so per-rank workload is comparable.
                    chat = random.choice(successful_chats)
                    logger.warning(
                        f"[collator] build_qwen_chat failed ({e}); "
                        f"substituting an earlier sample from the same batch."
                    )
                else:
                    chat = last_resort
                    logger.warning(
                        f"[collator] build_qwen_chat failed ({e}); "
                        f"no prior successful sample in batch — using 1-token pad."
                    )

            input_ids_list.append(chat.input_ids)
            labels_list.append(chat.labels)
            task_ids.append(sample["task_id"])
            if chat.pixel_values is not None:
                pixel_values_list.append(chat.pixel_values)
            if chat.image_grid_thw is not None:
                grid_thw_list.append(chat.image_grid_thw)

        # If the entire batch is text-only (no sample produced pixel_values),
        # inject a single small dummy image so the vision tower + projector
        # are still exercised on this rank. Without this, a DDP rank whose
        # micro-batch happens to draw only ShareGPT-style text-only samples
        # would skip the projector's collective ops while the other ranks
        # call them — producing the NumelIn=1 vs NumelIn=544M SeqNum=275407
        # desync we hit at step 3923 across takes 5/7/11. The dummy image's
        # features are absorbed by the model-side DDP-safety patch (see
        # ``merit.models.qwen25vl._patch_qwen25vl_forward_for_ddp_safety``)
        # which runs the vision tower then drops the features with a
        # zero-contribution graph link, so the LM is unaffected.
        if not pixel_values_list and self.processor is not None:
            from PIL import Image as _PILImage
            dummy = _PILImage.new("RGB", (56, 56), (0, 0, 0))
            enc = self.processor.image_processor(dummy, return_tensors="pt")
            pixel_values_list.append(enc["pixel_values"])
            grid_thw_list.append(enc["image_grid_thw"])

        max_len = max(x.numel() for x in input_ids_list)
        padded_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        padded_labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, (ids, lbls) in enumerate(zip(input_ids_list, labels_list)):
            n = ids.numel()
            padded_ids[i, :n] = ids
            padded_labels[i, :n] = lbls
            attention_mask[i, :n] = 1

        out: dict[str, Any] = {
            "input_ids": padded_ids,
            "labels": padded_labels,
            "attention_mask": attention_mask,
            "task_ids": task_ids,
        }
        if pixel_values_list:
            # Qwen2.5-VL flattens all patches across the batch into dim 0.
            out["pixel_values"] = torch.cat(pixel_values_list, dim=0)
            out["image_grid_thw"] = torch.cat(grid_thw_list, dim=0)
        return out
