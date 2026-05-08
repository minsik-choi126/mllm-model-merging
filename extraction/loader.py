"""Load VLM/LLM weights and normalize them onto a shared text-backbone schema.

The key job is `normalize_text_backbone_state_dict`: VLM checkpoints across
families (Qwen-VL, LLaVA-Llama, InternVL, Phi-Vision) embed the language model
under different prefixes (`model.language_model.*`, `language_model.model.*`,
`language_model.output.weight`, …). Normalizing them all to the standard
HuggingFace causal-LM schema (`model.embed_tokens.*`, `model.layers.*`,
`model.norm.*`, `lm_head.*`) lets downstream code treat any VLM's text path
the same way as a plain LLM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import save_file
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)


TEXT_KEY_PREFIXES = (
    "model.embed_tokens.",
    "model.layers.",
    "model.norm.",
    "model.rotary_emb.",
    "lm_head.",
)


def is_multimodal_model(model_id: str) -> bool:
    """Best-effort multimodal detection from the HuggingFace model ID."""
    model_id_lower = model_id.lower()
    return any(
        token in model_id_lower
        for token in ("-vl", "vision", "mllama", "internvl", "llava")
    )


def is_normalized_text_key(key: str) -> bool:
    """Whether a key already lives under the normalized text-backbone schema."""
    return key.startswith(TEXT_KEY_PREFIXES)


def resolve_local_pretrained_path(model_id: str) -> str:
    """Resolve an HF model ID to a local cache snapshot when one exists."""
    path = Path(model_id).expanduser()
    if path.exists():
        return str(path.resolve())

    hub_root = Path.home() / ".cache" / "huggingface" / "hub"
    model_cache_dir = hub_root / f"models--{model_id.replace('/', '--')}"
    if not model_cache_dir.exists():
        return model_id

    refs_main = model_cache_dir / "refs" / "main"
    if refs_main.exists():
        revision = refs_main.read_text().strip()
        snapshot_dir = model_cache_dir / "snapshots" / revision
        if snapshot_dir.exists():
            return str(snapshot_dir.resolve())

    snapshots_dir = model_cache_dir / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted(p for p in snapshots_dir.iterdir() if p.is_dir())
        if snapshots:
            return str(snapshots[-1].resolve())

    return model_id


def _from_pretrained_with_local_fallback(factory, model_id: str, **kwargs):
    """Retry from the local HF cache when network lookups fail."""
    resolved_model_id = resolve_local_pretrained_path(model_id)
    if resolved_model_id != model_id:
        kwargs = dict(kwargs)
        kwargs["local_files_only"] = True
        model_id = resolved_model_id

    try:
        return factory.from_pretrained(model_id, **kwargs)
    except Exception:
        if kwargs.get("local_files_only"):
            raise
        fallback_kwargs = dict(kwargs)
        fallback_kwargs["local_files_only"] = True
        return factory.from_pretrained(model_id, **fallback_kwargs)


def _load_transformers_model(
    model_id: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
):
    """Load a model with the right Auto class for its modality."""
    if is_multimodal_model(model_id):
        try:
            return _from_pretrained_with_local_fallback(
                AutoModelForImageTextToText,
                model_id,
                dtype=dtype,
                device_map=device,
                trust_remote_code=True,
            )
        except (ValueError, KeyError):
            from transformers import AutoModel
            return _from_pretrained_with_local_fallback(
                AutoModel,
                model_id,
                dtype=dtype,
                device_map=device,
                trust_remote_code=True,
            )

    try:
        return _from_pretrained_with_local_fallback(
            AutoModelForCausalLM,
            model_id,
            dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
    except (ValueError, TypeError):
        return _from_pretrained_with_local_fallback(
            AutoModelForImageTextToText,
            model_id,
            dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )


def normalize_text_backbone_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Normalize VLM/LLM weights to the shared text-backbone schema.

    Output keys are restricted to the standard causal-LM schema:
        model.embed_tokens.*
        model.layers.*
        model.norm.*
        lm_head.*
    """
    normalized: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        mapped_key: Optional[str] = None

        if key.startswith("model.language_model."):
            mapped_key = "model." + key[len("model.language_model."):]
        elif key.startswith("language_model.model."):
            mapped_key = "model." + key[len("language_model.model."):]
        elif key.startswith("language_model.lm_head."):
            mapped_key = "lm_head." + key[len("language_model.lm_head."):]
        elif key == "language_model.output.weight":
            mapped_key = "lm_head.weight"  # InternLM2 style
        elif is_normalized_text_key(key):
            mapped_key = key

        if mapped_key is not None:
            normalized[mapped_key] = value

    if not normalized:
        raise ValueError(
            "Could not normalize any text-backbone keys from the provided state dict."
        )
    return normalized


def load_state_dict(
    model_id: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
    normalize_text_backbone: bool = False,
) -> dict[str, torch.Tensor]:
    """Load a model state dict from HuggingFace.

    Args:
        model_id: HuggingFace model ID or local path.
        dtype: Weight dtype.
        device: Device map target (default cpu).
        normalize_text_backbone: If True, return only the text backbone in the
            normalized key schema shared by VLMs and base LLMs.
    """
    model = _load_transformers_model(model_id, dtype=dtype, device=device)
    sd = model.state_dict()
    del model
    torch.cuda.empty_cache()

    if normalize_text_backbone:
        return normalize_text_backbone_state_dict(sd)
    return sd


def _sanitize_metadata(metadata: Optional[dict[str, str]]) -> Optional[dict[str, str]]:
    """Safetensors metadata must be string-only."""
    if metadata is None:
        return None
    return {str(k): str(v) for k, v in metadata.items()}


def save_text_model(
    state_dict: dict[str, torch.Tensor],
    template_model_id: str,
    output_dir: str,
    metadata: Optional[dict[str, str]] = None,
    tokenizer_model_id: Optional[str] = None,
) -> Path:
    """Save a text-only model in HuggingFace format.

    Writes `model.safetensors` plus the config and tokenizer copied from
    `template_model_id` (or `tokenizer_model_id` if specified for the latter).
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    output_path = output_dir_path / "model.safetensors"
    save_file(state_dict, str(output_path), metadata=_sanitize_metadata(metadata))

    config = _from_pretrained_with_local_fallback(
        AutoConfig,
        template_model_id,
        trust_remote_code=True,
    )
    config.save_pretrained(output_dir_path)

    tokenizer = _from_pretrained_with_local_fallback(
        AutoTokenizer,
        tokenizer_model_id or template_model_id,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(output_dir_path)

    return output_path
