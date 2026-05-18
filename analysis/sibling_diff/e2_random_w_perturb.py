"""E2 — Direct test of the W-perturbation mechanism:

Add random Gaussian perturbation to W matrices of Qwen2.5-7B-Instruct (with
γ left untouched) such that ‖ΔW‖_F matches the empirically measured VL
adaptation magnitude. Then measure IFEval.

If IFEval drops by ~9 pt (matching the actual Qwen2.5-VL-LM observed drop),
then the *W-only perturbation mode* is sufficient to cause the IFEval damage,
closing the logical gap between C1 (γ ablation) and natural VL adaptation.

Per-sub-module relative perturbation magnitudes used (matching measured
diff_qwen25.csv mean rel_diff):

    self_attn.q_proj      : 0.682
    self_attn.k_proj      : 0.540
    self_attn.v_proj      : 0.420
    self_attn.o_proj      : 0.523
    mlp.gate_proj         : 0.608
    mlp.up_proj           : 0.600
    mlp.down_proj         : 0.572
    (norms left alone — γ frozen by construction)

Output: /131_data/geeho/minsik/Qwen2.5-7B-Instruct-Wperturb-seed{N}/
"""

from __future__ import annotations
import argparse, json, shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


SRC = Path("/131_data/geeho/minsik/Qwen2.5-7B-Instruct")
SUB_MAGNITUDES = {
    "self_attn.q_proj.weight": 0.682,
    "self_attn.k_proj.weight": 0.540,
    "self_attn.v_proj.weight": 0.420,
    "self_attn.o_proj.weight": 0.523,
    "mlp.gate_proj.weight":    0.608,
    "mlp.up_proj.weight":      0.600,
    "mlp.down_proj.weight":    0.572,
}
# Also perturb bias terms slightly (Qwen2.5 has q/k/v_proj.bias)
SUB_MAGNITUDES_BIAS = {
    "self_attn.q_proj.bias": 0.009,
    "self_attn.k_proj.bias": 0.075,
    "self_attn.v_proj.bias": 0.036,
}
NUM_LAYERS = 28


def load_all(src: Path) -> dict[str, torch.Tensor]:
    idx = json.load(open(src / "model.safetensors.index.json"))
    sd, shards = {}, {}
    for k, s in idx["weight_map"].items():
        shards.setdefault(s, []).append(k)
    for s, keys in shards.items():
        with safe_open(str(src / s), framework="pt") as f:
            for k in keys:
                sd[k] = f.get_tensor(k)
    return sd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rel-scale", type=float, default=1.0,
                   help="scale the per-sub magnitudes (1.0 = match Qwen2.5-VL mean rel_diff)")
    p.add_argument("--out", default=None,
                   help="output dir; default ./Qwen2.5-7B-Instruct-Wperturb-s{seed}-r{rel}")
    p.add_argument("--rank-match", action="store_true",
                   help="reduce perturbation effective rank to ~100 (match measured ΔW stable rank)")
    args = p.parse_args()

    if args.out is None:
        args.out = f"/131_data/geeho/minsik/Qwen2.5-7B-Instruct-Wperturb-s{args.seed}-r{args.rel_scale}"
    DST = Path(args.out)
    DST.mkdir(parents=True, exist_ok=True)

    print(f"loading source: {SRC}")
    sd = load_all(SRC)
    print(f"  {len(sd)} tensors loaded\n")

    torch.manual_seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    total_fro_added = 0.0
    n_patched = 0
    for li in range(NUM_LAYERS):
        for sub, rel_mag in SUB_MAGNITUDES.items():
            key = f"model.layers.{li}.{sub}"
            if key not in sd:
                continue
            W = sd[key].to(torch.float32)
            target_fro = rel_mag * args.rel_scale * W.norm().item()
            if args.rank_match:
                # Approximate stable rank ≈ 100 via low-rank random sketch
                m, n = W.shape
                r = 100
                U = torch.randn(m, r, generator=g) / (m ** 0.5)
                V = torch.randn(r, n, generator=g) / (r ** 0.5)
                N = U @ V
            else:
                N = torch.randn(W.shape, generator=g)
            # rescale to target Frobenius
            cur_fro = N.norm().item()
            if cur_fro < 1e-12:
                continue
            N = N * (target_fro / cur_fro)
            sd[key] = (W + N).to(sd[key].dtype)
            total_fro_added += target_fro
            n_patched += 1
        # also handle biases
        for sub, rel_mag in SUB_MAGNITUDES_BIAS.items():
            key = f"model.layers.{li}.{sub}"
            if key not in sd:
                continue
            W = sd[key].to(torch.float32)
            target_fro = rel_mag * args.rel_scale * max(W.norm().item(), 1e-6)
            N = torch.randn(W.shape, generator=g)
            N = N * (target_fro / max(N.norm().item(), 1e-12))
            sd[key] = (W + N).to(sd[key].dtype)
            n_patched += 1

    print(f"perturbed {n_patched} tensors  total Δ-Frobenius mass: {total_fro_added:.2f}")
    print(f"seed: {args.seed}  rel_scale: {args.rel_scale}  rank_match: {args.rank_match}")

    # save
    save_file(sd, str(DST / "model.safetensors"), metadata={"format": "pt"})
    print(f"wrote {DST/'model.safetensors'}")

    # copy config + tokenizer
    for name in ("config.json", "generation_config.json", "tokenizer.json",
                 "tokenizer_config.json", "vocab.json", "merges.txt",
                 "special_tokens_map.json", "added_tokens.json",
                 "chat_template.jinja"):
        sp = SRC / name
        if sp.exists() and sp.is_file():
            shutil.copy2(sp, DST / name)
    print(f"\noverlay ready: {DST}")


if __name__ == "__main__":
    main()
