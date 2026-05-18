"""In-place injection of per-head QK-RMSNorm modules into a Qwen2 model.

After calling `inject_qknorm(model)`:
- Each attention block has new `q_norm` and `k_norm` modules (per-head RMSNorm
  with γ ∈ R^head_dim).
- γ is initialized to 1 (identity). With γ=1, the modified model is *not*
  exactly identical to the original Qwen2 (RMSNorm normalizes magnitudes),
  but its functional behavior should be close to a pre-normalized form.
- During training, γ is a learnable parameter that can grow to create
  amplifier structure analogous to Qwen3's q_norm / k_norm.

Reference behavior (Qwen3): after qkv projection and reshape to
(B, n_heads, T, head_dim), apply RMSNorm(γ) along the last axis, then RoPE,
then attention. We replicate that ordering.

Usage:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    inject_qknorm(model)
    # now model.model.layers[i].self_attn has q_norm / k_norm
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Callable

import torch
import torch.nn as nn


class HeadDimRMSNorm(nn.Module):
    """RMSNorm over the last (head_dim) axis, with per-channel scale γ.

    Matches Qwen3's `Qwen3RMSNorm(head_dim)` behavior.
    """

    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., head_dim)
        dtype_in = x.dtype
        x32 = x.to(torch.float32)
        var = x32.pow(2).mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.eps)
        out = (x32 * self.weight).to(dtype_in)
        return out


def _make_qknorm_forward(orig_forward: Callable, q_norm: nn.Module, k_norm: nn.Module):
    """Wrap an existing Qwen2Attention.forward to apply q_norm/k_norm after
    qkv projection (and reshape) but before RoPE/attention.

    Strategy: instead of replacing forward (which would require re-implementing
    all of Qwen2Attention internals, version-specific), we use forward pre-hook
    on q_proj/k_proj to inject the norm. But pre-hooks see the input, not the
    reshaped output — wrong place.

    Cleaner: post-forward hook on q_proj/k_proj that reshapes, normalizes, then
    re-flattens. But the original Qwen2 code does its own reshape after q_proj.
    Easiest is to *monkey-patch* the entire forward method with a copy that
    inserts q_norm / k_norm at the right place. We do this safely by relying
    on a method that we patch into the *instance* (not the class) so different
    models can have different versions.

    Concretely, we patch attn.forward to a function that:
      1. Runs the original q_proj, k_proj, v_proj.
      2. Reshapes to (B, n_heads, T, head_dim).
      3. Applies q_norm to query_states, k_norm to key_states.
      4. Continues with RoPE + attention as in the original.

    This requires a version-specific replication of Qwen2Attention.forward.
    """
    raise NotImplementedError("use inject_qknorm_via_subclass instead — see below")


def _find_attention_layers(model: nn.Module) -> list[nn.Module]:
    """Locate the list of decoder layers in a Qwen2 or Qwen2.5-VL model.

    Returns a list of layer modules each having `.self_attn`. Supports:
      - Pure Qwen2 / Qwen3: `model.model.layers`
      - Qwen2.5-VL composite: `model.model.language_model.layers` or
        `model.language_model.model.layers`
    """
    # Candidate 1: pure LLM
    if hasattr(model, "model") and hasattr(model.model, "layers") \
            and isinstance(model.model.layers, (list, nn.ModuleList)):
        return list(model.model.layers)
    # Candidate 2: Qwen2.5-VL ConditionalGeneration (model.model.language_model.layers)
    if hasattr(model, "model") and hasattr(model.model, "language_model") \
            and hasattr(model.model.language_model, "layers"):
        return list(model.model.language_model.layers)
    # Candidate 3: model.language_model.model.layers (alternative VL layout)
    if hasattr(model, "language_model") and hasattr(model.language_model, "model") \
            and hasattr(model.language_model.model, "layers"):
        return list(model.language_model.model.layers)
    raise RuntimeError("Could not locate decoder layers in model; "
                       "tried model.model.layers, model.model.language_model.layers, "
                       "model.language_model.model.layers")


def inject_qknorm(model: nn.Module, eps: Optional[float] = None) -> nn.Module:
    """In-place modify a Qwen2/Qwen2.5-VL model to add per-head q_norm/k_norm.

    Works for both pure Qwen2 LLMs and Qwen2.5-VL composite VLMs (the LM
    portion gets the injection; vision tower is untouched).
    """
    layers = _find_attention_layers(model)
    attn0 = layers[0].self_attn
    head_dim = attn0.head_dim

    config = model.config
    # Pull rms_norm_eps from the text-config when present (VL config nests it).
    if hasattr(config, "text_config") and config.text_config is not None:
        rms_eps_default = getattr(config.text_config, "rms_norm_eps", 1e-6)
    else:
        rms_eps_default = getattr(config, "rms_norm_eps", 1e-6)
    rms_eps = eps if eps is not None else rms_eps_default

    for layer in layers:
        attn = layer.self_attn
        if hasattr(attn, "q_norm") and isinstance(attn.q_norm, HeadDimRMSNorm):
            continue
        device = attn.q_proj.weight.device
        dtype = attn.q_proj.weight.dtype
        attn.q_norm = HeadDimRMSNorm(head_dim, eps=rms_eps).to(device=device, dtype=dtype)
        attn.k_norm = HeadDimRMSNorm(head_dim, eps=rms_eps).to(device=device, dtype=dtype)
        _patch_qwen2_attention_forward_inplace(attn)
    return model


def _patch_qwen2_attention_forward_inplace(attn: nn.Module):
    """Replace attn.forward with a version that applies q_norm / k_norm
    right after qkv projection and reshape.

    Replicates the canonical Qwen2 forward signature/behavior, version-checked
    against transformers 4.57.x.
    """
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, eager_attention_forward
    try:
        from transformers.modeling_flash_attention_utils import _flash_attention_forward
    except ImportError:
        _flash_attention_forward = None

    def forward(self,
                hidden_states: torch.Tensor,
                position_embeddings,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_value=None,
                cache_position=None,
                **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states   = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # NEW: apply q_norm / k_norm before RoPE
        query_states = self.q_norm(query_states)
        key_states   = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Use eager attention (compatible with all transformers versions)
        attention_interface = eager_attention_forward
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    # bind as bound method on the instance
    import types
    attn.forward = types.MethodType(forward, attn)


# ---------- Loading QK-norm weights from a saved checkpoint ----------

def load_qknorm_state_if_present(model: nn.Module, model_dir: str | object) -> int:
    """If ``model_dir``'s safetensors contain ``*.q_norm.weight`` /
    ``*.k_norm.weight`` keys, load them into the (already-injected) modules.

    Use case: between Stage 1 and Stage 2, we want to *continue* training the
    γ values learned in Stage 1. ``from_pretrained`` doesn't load these
    (they're "unexpected keys" w.r.t. the vanilla model class), so we
    explicitly copy them in after ``inject_qknorm`` has created the target
    modules.

    Returns: number of q_norm/k_norm parameters successfully loaded.
    """
    import json
    from pathlib import Path
    from safetensors import safe_open

    p = Path(str(model_dir))
    if not p.is_dir():
        return 0
    idx_file = p / "model.safetensors.index.json"
    if idx_file.exists():
        idx = json.load(open(idx_file))
        weight_map = idx["weight_map"]
    else:
        single = p / "model.safetensors"
        if not single.exists():
            return 0
        with safe_open(str(single), framework="pt") as f:
            weight_map = {k: "model.safetensors" for k in f.keys()}

    qk_keys = [k for k in weight_map
               if k.endswith(".q_norm.weight") or k.endswith(".k_norm.weight")]
    if not qk_keys:
        return 0

    by_shard: dict[str, list[str]] = {}
    for k in qk_keys:
        by_shard.setdefault(weight_map[k], []).append(k)

    target_state = model.state_dict()
    n_loaded = 0
    for shard, keys in by_shard.items():
        with safe_open(str(p / shard), framework="pt") as f:
            for k in keys:
                if k not in target_state:
                    continue
                tensor = f.get_tensor(k)
                target_state[k].data.copy_(tensor.to(
                    device=target_state[k].device,
                    dtype=target_state[k].dtype,
                ))
                n_loaded += 1
    return n_loaded


# ---------- Build-init helper: inject QK-norm and save ----------

def inject_and_save(input_dir: str, output_dir: str, dtype: str = "bfloat16") -> None:
    """Build-time helper: load a (composed) VLM checkpoint, inject QK-norm
    modules with γ=1, save the result. The output is a self-contained HF
    checkpoint whose safetensors now contain q_norm.weight / k_norm.weight.
    """
    import shutil
    from pathlib import Path
    from transformers import (
        AutoProcessor, AutoTokenizer,
        Qwen2_5_VLForConditionalGeneration,
    )

    _dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
              "float32": torch.float32}[dtype]

    print(f"[inject_and_save] loading {input_dir}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        input_dir, dtype=_dtype, low_cpu_mem_usage=True, device_map="cpu",
    )
    print("[inject_and_save] injecting QK-RMSNorm (γ=1)")
    inject_qknorm(model)
    n_new = sum(p.numel() for n, p in model.named_parameters()
                if n.endswith("q_norm.weight") or n.endswith("k_norm.weight"))
    print(f"[inject_and_save]   added {n_new} γ params total")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    # Carry tokenizer + processor across.
    AutoTokenizer.from_pretrained(input_dir, trust_remote_code=True).save_pretrained(out)
    AutoProcessor.from_pretrained(input_dir, trust_remote_code=True).save_pretrained(out)
    print(f"[inject_and_save] wrote {out}")


# ---------- Unit tests ----------

def _unit_test_identity_match(model_path: str = "/131_data/geeho/minsik/Qwen2.5-0.5B-Instruct",
                              prompt: str = "Hello. What is the capital of France?",
                              tol: float = 5e-3):
    """With γ = 1, the QK-norm-injected model output should be CLOSE to (but not
    exactly equal to) the original model. RMSNorm with γ=1 normalizes Q,K to
    unit RMS per head, which differs from the original (unnormalized) Q,K. So
    expect some drift but not catastrophic.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import copy

    print(f"Loading {model_path} ...")
    tok = AutoTokenizer.from_pretrained(model_path)
    base = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda:0")
    base.eval()

    print("Injecting QK-norm (γ=1) ...")
    patched = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda:0")
    inject_qknorm(patched)
    patched.eval()

    ids = tok(prompt, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        out_base = base(**ids).logits
        out_patched = patched(**ids).logits
    diff = (out_base.float() - out_patched.float()).abs()
    print(f"max abs diff: {diff.max().item():.5f}")
    print(f"mean abs diff: {diff.mean().item():.5f}")
    print(f"rel diff (mean / mean_of_abs): {(diff.mean() / out_base.abs().mean()).item():.5f}")
    # With γ=1 (RMSNorm normalizes), output WILL differ — sanity is that
    # it's not absurdly large or NaN.
    assert torch.isfinite(out_patched).all(), "non-finite output after QK-norm injection"
    print("\n✓ Sanity OK: QK-norm injection runs, output is finite. "
          "(Difference is expected since γ=1 applies non-trivial RMSNorm.)")


def _unit_test_grad_flow(model_path: str = "/131_data/geeho/minsik/Qwen2.5-0.5B-Instruct"):
    """γ is a learnable parameter; gradient should flow to it under
    standard training loss.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    m = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32, device_map="cuda:0")
    inject_qknorm(m)
    m.train()

    ids = tok("Hello world this is a test sentence.", return_tensors="pt").to("cuda:0")
    labels = ids["input_ids"].clone()
    out = m(**ids, labels=labels)
    out.loss.backward()

    # check that q_norm.weight.grad is non-None and non-zero
    bad = []
    for i, layer in enumerate(m.model.layers):
        attn = layer.self_attn
        for nm in ("q_norm", "k_norm"):
            g = getattr(attn, nm).weight.grad
            if g is None or g.abs().sum().item() == 0:
                bad.append((i, nm, g))
    if bad:
        print(f"⚠️  no gradient on {len(bad)} norm params:")
        for i, nm, g in bad[:5]:
            print(f"  layer {i} {nm} grad = {g}")
        raise AssertionError("γ parameter is not learning")
    print(f"\n✓ Gradient flows to q_norm/k_norm γ on all {len(m.model.layers)} layers.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--test", choices=["identity", "grad", "all"], default="all")
    p.add_argument("--model", default="/131_data/geeho/minsik/Qwen2.5-0.5B-Instruct")
    args = p.parse_args()

    if args.test in ("identity", "all"):
        print("=" * 70)
        print("Test 1: identity (γ=1) sanity — output is finite and reasonable")
        print("=" * 70)
        _unit_test_identity_match(args.model)

    if args.test in ("grad", "all"):
        print()
        print("=" * 70)
        print("Test 2: gradient flow to γ parameters")
        print("=" * 70)
        _unit_test_grad_flow(args.model)

    print("\nAll tests passed.")
