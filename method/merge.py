"""E-Pull merge: per-direction constrained-Pareto closed form.

Per Theorem (Closed form) and Algorithm 1 of the method section:

    m_j^**  = (1 - g_j) * m_j^*  +  g_j * u_{r*(j)}^j
    g_j     = 1 - H_j / log(k)
    H_j     = -Σ_i π_{j,i} log π_{j,i}
    π_{j,i} = α_i λ_{i,j} / Σ_l α_l λ_{l,j}
    m_j^*   = Σ_i π_{j,i} ΔW_i v_j      (RegMean per-direction)
    u_i^j   = ΔW_i v_j

Reassembly: with V the **orthogonal** FG joint diagonalizer (paper's CPC
basis assumption) and M the matrix whose j-th column is m_j^**,

    W^** = W_base + M V^T

V is produced by `joint_diag.joint_diagonalize` (gen-eig + polar warm start
followed by Cardoso-Souloumiac Jacobi sweeps to minimize the FG cost). The
`V_inv` attribute on `JointDiagResult` is provided generically for the
reassembly call site; for an orthogonal V it is just `V^T`.

Down-projection layers (input dim = intermediate size) are excluded from
active merging and use the dominant-energy modality, matching the paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import torch

from .covariance import GramArtifacts, is_down_projection
from .joint_diag import joint_diagonalize


@dataclass
class EpullConfig:
    alphas: tuple[float, ...] = (0.5, 0.5)
    eps: float = 1e-6
    # fp32 is the practical default at d≈3584: SVD/eigh ~10× faster than fp64,
    # and fp32 residual error (~1e-6) is well below the calibration-sample noise.
    # Tests in _self_test.py explicitly pass fp64 to verify identities at machine precision.
    diag_dtype: torch.dtype = torch.float32
    output_dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"
    consume_inputs: bool = False
    jacobi_sweeps: int = 2                     # match joint_diag default; 0 disables Jacobi refinement

    def __post_init__(self):
        if len(self.alphas) < 2:
            raise ValueError(
                f"E-Pull requires at least 2 modalities; got alphas={self.alphas}"
            )
        if any(a <= 0 for a in self.alphas):
            raise ValueError(
                f"alphas must be strictly positive (got {self.alphas}); "
                "negative or zero α breaks the routing-probability semantics."
            )
        if self.eps <= 0:
            raise ValueError(f"eps must be > 0 (got {self.eps})")
        if self.jacobi_sweeps < 0:
            raise ValueError(
                f"jacobi_sweeps must be >= 0 (got {self.jacobi_sweeps})"
            )
        s = sum(self.alphas)
        self.alphas = tuple(a / s for a in self.alphas)


@dataclass
class LayerMergeStats:
    name: str
    in_dim: int
    out_dim: int
    mode: str                         # "epull" | "owner_energy"
    off_diagonal_residual: float = 0.0   # mean ||off-diag(V^T C_i V)||_F / ||V^T C_i V||_F
    fg_cost: float = 0.0                 # normalized FG objective Σ α_i ||off||² / Σ α_i ||M||²
    commutator_residual: float = 0.0     # ||C_1 C_2 - C_2 C_1||_F / (||C_1||·||C_2||) — k=2 only
    n_jacobi_sweeps: int = 0
    avg_gate: float = 0.0
    avg_entropy_norm: float = 0.0     # mean of H_j / log k
    owner_fraction_per_modality: dict[int, float] = field(default_factory=dict)
    chosen_modality: int = -1         # for owner_energy mode
    diag_method: str = ""             # joint-diag backend used

    @property
    def fg_residual(self) -> float:
        """Backwards-compat alias; semantically == off_diagonal_residual."""
        return self.off_diagonal_residual


def _epull_direction_combine(
    delta_Ws: Sequence[torch.Tensor],   # k tensors of (out, in)
    V: torch.Tensor,                    # (in, in) joint diagonalizer (need not be orthogonal)
    lam: torch.Tensor,                  # (k, in) per-direction eigvals (>0)
    alphas: torch.Tensor,               # (k,) summing to 1
    *,
    V_inv: torch.Tensor | None = None,  # (in, in) V^{-1}; defaults to V^T (only valid if V is orthogonal)
) -> tuple[torch.Tensor, dict]:
    """Compute the merged ΔW = W^** - W_base for one linear layer."""
    k, d_in = lam.shape
    if k < 2:
        raise ValueError(f"E-Pull undefined for k={k}")
    log_k = math.log(k)

    Z = (alphas[:, None] * lam).sum(0)
    Z = Z.clamp_min(torch.finfo(lam.dtype).tiny)
    pi = (alphas[:, None] * lam) / Z
    pi = pi.clamp_min(1e-12)
    pi = pi / pi.sum(0, keepdim=True)

    H = -(pi * pi.log()).sum(0)
    H_norm = (H / log_k).clamp(0.0, 1.0)
    g = (1.0 - H_norm).clamp(0.0, 1.0)

    r_star = pi.argmax(0)

    Us = torch.stack([dW @ V for dW in delta_Ws], dim=0)         # (k, out, in)
    M_star = (pi.unsqueeze(1) * Us).sum(0)                       # (out, in)
    Us_pio = Us.permute(2, 0, 1)                                 # (in, k, out)
    cols = torch.arange(d_in, device=Us.device)
    M_owner = Us_pio[cols, r_star, :].t()                        # (out, in)

    M_star_star = (1.0 - g).unsqueeze(0) * M_star + g.unsqueeze(0) * M_owner

    if V_inv is None:
        V_inv = V.t()  # only correct when V is orthonormal
    delta_W_merged = M_star_star @ V_inv

    owner_fraction = {
        int(i): float((r_star == i).float().mean().item())
        for i in range(k)
    }
    return delta_W_merged, {
        "avg_gate": float(g.mean().item()),
        "avg_entropy_norm": float(H_norm.mean().item()),
        "owner_fraction_per_modality": owner_fraction,
    }


@torch.no_grad()
def merge_one_layer(
    name: str,
    base_W: torch.Tensor,
    Ws: Sequence[torch.Tensor],
    grams: Sequence[GramArtifacts],
    cfg: EpullConfig,
) -> tuple[torch.Tensor, LayerMergeStats]:
    in_dim = base_W.shape[1]
    out_dim = base_W.shape[0]

    if is_down_projection(name):
        traces = []
        for g in grams:
            if name in g.traces and g.counts.get(name, 0) > 0:
                traces.append(g.traces[name] / g.counts[name])
            else:
                traces.append(0.0)
        scored = [a * t for a, t in zip(cfg.alphas, traces)]
        chosen = int(max(range(len(Ws)), key=lambda i: scored[i]))
        return Ws[chosen].to(cfg.output_dtype), LayerMergeStats(
            name=name, in_dim=in_dim, out_dim=out_dim,
            mode="owner_energy", chosen_modality=chosen,
        )

    Cs = []
    for g in grams:
        if name not in g.grams:
            raise KeyError(f"Layer {name!r} missing from a calibration set")
        Cs.append(g.grams[name].to(cfg.diag_dtype).to(cfg.device))

    jd = joint_diagonalize(
        Cs,
        cfg.alphas,
        eps=cfg.eps,
        dtype=cfg.diag_dtype,
        jacobi_sweeps=cfg.jacobi_sweeps,
    )
    V = jd.V
    V_inv = jd.V_inv
    lam = jd.eigvals_per_modality
    alphas = torch.tensor(cfg.alphas, dtype=cfg.diag_dtype, device=V.device)

    base_dev = base_W.to(cfg.device, cfg.diag_dtype)
    delta_Ws = [W.to(cfg.device, cfg.diag_dtype) - base_dev for W in Ws]

    delta_merged, stats_extra = _epull_direction_combine(
        delta_Ws, V, lam, alphas, V_inv=V_inv,
    )
    W_merged = (base_dev + delta_merged).to(cfg.output_dtype).cpu()

    return W_merged, LayerMergeStats(
        name=name, in_dim=in_dim, out_dim=out_dim,
        mode="epull",
        off_diagonal_residual=jd.off_diagonal_residual,
        fg_cost=jd.fg_cost,
        commutator_residual=jd.commutator_residual,
        n_jacobi_sweeps=jd.n_jacobi_sweeps,
        diag_method=jd.method,
        **stats_extra,
    )


@torch.no_grad()
def epull_merge_state_dicts(
    base_sd: dict[str, torch.Tensor],
    fine_tuned_sds: Sequence[dict[str, torch.Tensor]],
    grams: Sequence[GramArtifacts],
    cfg: EpullConfig,
    *,
    progress: bool = True,
) -> tuple[dict[str, torch.Tensor], list[LayerMergeStats]]:
    """Apply E-Pull layer-by-layer to all linear weights covered by Gram artifacts.

    Non-merged keys (norm scales, embeddings, lm_head, etc.) are copied from
    `base_sd`. Use `extraction.normalize_text_backbone_state_dict` first if
    the fine-tuned SDs are VLM-style.

    The fine-tuned state dicts are read but not mutated unless
    `cfg.consume_inputs` is True (in which case per-layer weights are dropped
    after use to save memory; caller's SDs become invalid post-call).
    """
    if len(fine_tuned_sds) != len(grams):
        raise ValueError("Need one Gram artifact per fine-tuned model")
    if len(fine_tuned_sds) != len(cfg.alphas):
        raise ValueError(
            f"alphas length {len(cfg.alphas)} != #models {len(fine_tuned_sds)}"
        )

    out_sd: dict[str, torch.Tensor] = {}
    for k, v in base_sd.items():
        out_sd[k] = v.to(cfg.output_dtype).cpu().clone()

    layer_names = sorted(set(grams[0].in_dims.keys()))
    stats: list[LayerMergeStats] = []

    for i, name in enumerate(layer_names):
        weight_key = f"{name}.weight"
        if weight_key not in base_sd:
            print(f"  [skip] {weight_key} missing from base SD")
            continue
        Ws = []
        ok = True
        for sd in fine_tuned_sds:
            if weight_key not in sd:
                print(f"  [skip] {weight_key} missing from a fine-tuned SD")
                ok = False
                break
            Ws.append(sd[weight_key])
        if not ok:
            continue

        merged, st = merge_one_layer(
            name=name,
            base_W=base_sd[weight_key],
            Ws=Ws,
            grams=grams,
            cfg=cfg,
        )
        out_sd[weight_key] = merged
        stats.append(st)

        if progress:
            if st.mode == "owner_energy":
                tag = f"owner mod={st.chosen_modality}"
            else:
                tag = (
                    f"g={st.avg_gate:.3f} fg={st.fg_cost:.2e} "
                    f"off={st.off_diagonal_residual:.2e} "
                    f"comm={st.commutator_residual:.2e} "
                    f"j={st.n_jacobi_sweeps}"
                )
            print(f"  [{i+1}/{len(layer_names)}] {name}  ({tag})")

        if cfg.consume_inputs:
            for sd in fine_tuned_sds:
                del sd[weight_key]
        torch.cuda.empty_cache()

    return out_sd, stats
