"""Symmetric orthogonal joint diagonalization {C_i = V Λ_i V^T}, paper-faithful.

The paper's setup (and the constrained-Pareto theorems) rely on an
**orthogonal** common basis `V^T V = I` and define the FG estimator as

    V_FG = argmin_{V^T V = I}  Σ_i α_i || off(V^T C_i V) ||_F^2.            (FG)

Construction
------------
The estimator combines a closed-form **warm start** with a small number of
**parallel Cardoso-Souloumiac (CS) Jacobi sweeps** that drive the FG cost
toward its (local) minimum:

  1. Generalized-eigvec joint diagonalization (Cholesky whitening) of (C_1, C_2)
     produces ``V_gen`` that exactly diagonalizes both, but is non-orthogonal.
  2. Polar projection onto the orthogonal manifold gives ``V_warm = polar(V_gen)``.
     Under exact CPC with shared orthogonal ``V_true``, ``V_gen = V_true Λ_2^{-1/2}``,
     so ``polar(V_gen) = V_true`` recovers the true CPC basis exactly. (Verified.)
  3. CS Jacobi sweeps refine ``V_warm`` toward the FG optimum. CS minimizes
     exactly the FG cost on the orthogonal Stiefel manifold — symmetric in
     modality order, with quadratic local convergence near the optimum. The
     warm start places us in that quadratic regime under (near-)CPC.

Sweeps are scheduled as round-robin tournaments: each of ``d−1`` rounds in a
sweep contains ``d/2`` disjoint pairs, all rotated in parallel via batched
``torch`` ops. The default ``jacobi_sweeps = 2`` is conservative; convergence
is reported via the FG cost residual after the final sweep.

Why not just sum-PCA?  Eigvecs of ``Σ_i α_i C_i`` fail when the weighted
average is degenerate, returning an arbitrary rotation in the degenerate
subspace even under exact CPC (see ``_self_test.test_degenerate_cpc_avg``).

Diagnostics
-----------
Three residuals are exposed:

  * ``off_diagonal_residual`` — ``mean_i ||off-diag(V^T C_i V)||_F /
    ||V^T C_i V||_F`` on the final orthogonal V; the FG-cost-based proxy.
  * ``fg_cost`` — ``Σ_i α_i ||off(V^T C_i V)||_F^2 / Σ_i α_i ||V^T C_i V||_F^2``,
    the actual FG objective normalized.
  * ``commutator_residual`` — ``||C_1 C_2 - C_2 C_1||_F /
    (||C_1||_F · ||C_2||_F)``. Estimator-independent CPC quality:
    commuting SPD matrices share an orthogonal eigenbasis (CPC) and have zero
    commutator. Diagnoses the CPC ansatz on the data, not the estimator.

Limitation
----------
``k > 2`` raises ``NotImplementedError`` (the warm start is currently k=2-only;
the CS-Jacobi sweep already supports general ``k``). Lifting requires a k≥3
warm start (e.g. iterative whitening of one matrix, or initializing V at the
identity and relying entirely on Jacobi). The current LLM+VLM-LM application
is ``k = 2``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class JointDiagResult:
    V: torch.Tensor                      # (d, d) orthogonal
    V_inv: torch.Tensor                  # (d, d) = V^T  (orthogonal)
    eigvals_per_modality: torch.Tensor   # (k, d)  λ_{i,j} = (V^T C_i V)_{j,j}.clamp_min(0) + ε
    off_diagonal_residual: float         # mean_i ||off-diag(V^T C_i V)||_F / ||V^T C_i V||_F
    fg_cost: float                       # Σ α_i ||off(V^T C_i V)||²_F  /  Σ α_i ||V^T C_i V||²_F
    commutator_residual: float           # ||C_1 C_2 - C_2 C_1||_F / (||C_1||_F ||C_2||_F)  (k=2 only)
    method: str
    n_jacobi_sweeps: int

    # Backwards-compat alias for older callers; semantically the off-diagonal residual.
    @property
    def fg_residual(self) -> float:
        return self.off_diagonal_residual


def _generalized_eig_k2(
    C1: torch.Tensor,
    C2: torch.Tensor,
    *,
    eps: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """V_gen with V_gen^T C_2 V_gen ≈ I, V_gen^T C_1 V_gen diagonal. Non-orthogonal."""
    d = C1.shape[0]
    eye = eps * torch.eye(d, dtype=dtype, device=C1.device)
    L = torch.linalg.cholesky(C2.to(dtype) + eye)
    Linv_C1 = torch.linalg.solve_triangular(L, C1.to(dtype), upper=False)
    M = torch.linalg.solve_triangular(L, Linv_C1.t(), upper=False).t()
    M = 0.5 * (M + M.t())
    eigvals_M, U = torch.linalg.eigh(M)
    order = torch.argsort(eigvals_M, descending=True)
    U = U[:, order]
    V_gen = torch.linalg.solve_triangular(L.t(), U, upper=True)
    return V_gen


def _polar_orthogonal(M: torch.Tensor) -> torch.Tensor:
    """Polar projection: closest orthogonal matrix to M (Frobenius)."""
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    return U @ Vh


def _round_robin_pairs(d: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Round-robin schedule: d-1 rounds × d/2 disjoint pairs each.

    For even d the schedule covers all C(d, 2) pairs exactly once. For odd d
    we run the schedule on d+1 with a dummy index and drop pairs touching it.
    """
    if d <= 1:
        return []
    pad = (d % 2 == 1)
    n = d + 1 if pad else d
    others = list(range(1, n))
    rounds: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(n - 1):
        ps_list = [0]
        qs_list = [others[0]]
        for i in range(1, n // 2):
            ps_list.append(others[i])
            qs_list.append(others[n - 1 - i])
        # Filter dummy if odd d
        if pad:
            keep = [(p < d) and (q < d) for p, q in zip(ps_list, qs_list)]
            ps_list = [p for p, k in zip(ps_list, keep) if k]
            qs_list = [q for q, k in zip(qs_list, keep) if k]
        if ps_list:
            rounds.append(
                (
                    torch.tensor(ps_list, dtype=torch.long),
                    torch.tensor(qs_list, dtype=torch.long),
                )
            )
        others = [others[-1]] + others[:-1]
    return rounds


@torch.no_grad()
def _cs_jacobi_sweep(
    Ms: torch.Tensor,                 # (k, d, d) — V^T C_i V, in-place updated
    V: torch.Tensor,                  # (d, d)    — orthogonal, in-place updated
    alphas: torch.Tensor,             # (k,)      — strictly positive
    *,
    schedule: list[tuple[torch.Tensor, torch.Tensor]],
    skip_tol: float,
) -> None:
    """One sweep of parallel Cardoso-Souloumiac Jacobi: minimize Σ α_i ||off(M_i)||²_F."""
    k, d, _ = Ms.shape
    for ps_cpu, qs_cpu in schedule:
        ps = ps_cpu.to(V.device)
        qs = qs_cpu.to(V.device)

        diag_M = torch.diagonal(Ms, dim1=-2, dim2=-1)             # (k, d)
        a = diag_M[:, ps] - diag_M[:, qs]                         # (k, P)
        b = 2.0 * Ms[:, ps, qs]                                   # (k, P)
        alpha_col = alphas.view(k, 1)
        Sxx = (alpha_col * a * a).sum(0)                          # (P,)
        Syy = (alpha_col * b * b).sum(0)
        Sxy = (alpha_col * a * b).sum(0)
        diff = Sxx - Syy

        active = (Sxy.abs() >= skip_tol) | (diff.abs() >= skip_tol)
        if not bool(active.any()):
            continue
        # We maximize  Σ_i α_i (a_i u + b_i v)²  with  (u, v) = (cos 2θ, sin 2θ).
        # The optimum (u, v) is the largest eigvec of [[Sxx, Sxy], [Sxy, Syy]], at
        # eigvec angle ω where  2ω = atan2(2·Sxy, Sxx - Syy)  (standard 2×2-symmetric
        # diagonalization formula). Setting (u, v) = (cos ω, sin ω) gives 2θ = ω, so
        #     θ = ω / 2 = 0.25 · atan2(2·Sxy, Sxx - Syy).
        # (Coordinate-space Givens by θ corresponds to (a, b)-space rotation by 2θ — the
        # double-angle relation — hence the 0.25 factor instead of 0.5.)
        theta = 0.25 * torch.atan2(2.0 * Sxy, diff)
        # Zero-out rotations for already-converged pairs so they genuinely no-op.
        theta = torch.where(active, theta, torch.zeros_like(theta))
        c = torch.cos(theta)                                       # (P,)
        s = torch.sin(theta)

        Vp = V[:, ps]                                              # (d, P)
        Vq = V[:, qs]
        new_Vp = c * Vp + s * Vq
        new_Vq = -s * Vp + c * Vq
        V[:, ps] = new_Vp
        V[:, qs] = new_Vq

        c_row = c.view(1, -1, 1)
        s_row = s.view(1, -1, 1)
        Mp_row = Ms[:, ps, :]                                      # (k, P, d)
        Mq_row = Ms[:, qs, :]
        new_Mp_row = c_row * Mp_row + s_row * Mq_row
        new_Mq_row = -s_row * Mp_row + c_row * Mq_row
        Ms[:, ps, :] = new_Mp_row
        Ms[:, qs, :] = new_Mq_row

        c_col = c.view(1, 1, -1)
        s_col = s.view(1, 1, -1)
        Mp_col = Ms[:, :, ps]                                      # (k, d, P)
        Mq_col = Ms[:, :, qs]
        new_Mp_col = c_col * Mp_col + s_col * Mq_col
        new_Mq_col = -s_col * Mp_col + c_col * Mq_col
        Ms[:, :, ps] = new_Mp_col
        Ms[:, :, qs] = new_Mq_col


def _fg_cost_and_residuals(Ms: torch.Tensor, alphas: torch.Tensor):
    """Return (fg_cost, mean_off_diag_residual)."""
    k = Ms.shape[0]
    diag = torch.diagonal(Ms, dim1=-2, dim2=-1)                   # (k, d)
    full_norm_sq = (Ms * Ms).sum(dim=(-1, -2))                    # (k,)
    diag_norm_sq = (diag * diag).sum(dim=-1)                      # (k,)
    off_norm_sq = (full_norm_sq - diag_norm_sq).clamp_min(0)
    fg_cost_num = (alphas * off_norm_sq).sum().item()
    fg_cost_den = (alphas * full_norm_sq).sum().clamp_min(
        torch.finfo(Ms.dtype).tiny
    ).item()
    fg_cost = fg_cost_num / fg_cost_den
    rels = []
    for k_i in range(k):
        full = full_norm_sq[k_i].sqrt().clamp_min(torch.finfo(Ms.dtype).tiny).item()
        rels.append(off_norm_sq[k_i].sqrt().item() / full)
    return fg_cost, sum(rels) / len(rels)


@torch.no_grad()
def joint_diagonalize(
    Cs: Sequence[torch.Tensor],
    alphas: Sequence[float],
    *,
    eps: float = 1e-6,
    dtype: torch.dtype = torch.float64,
    jacobi_sweeps: int = 2,
    jacobi_skip_tol: float = 1e-12,
    fg_cost_tol: float = 1e-14,
) -> JointDiagResult:
    k = len(Cs)
    if k != len(alphas):
        raise ValueError(f"len(Cs)={k} != len(alphas)={len(alphas)}")
    if k < 2:
        raise ValueError(f"E-Pull is undefined for k={k}; need at least 2 modalities")
    if any(a <= 0 for a in alphas):
        raise ValueError(f"alphas must be strictly positive (got {alphas})")
    if eps <= 0:
        raise ValueError(f"eps must be > 0 (got {eps})")

    Cs_d = [C.to(dtype) for C in Cs]
    alpha_t = torch.tensor(alphas, dtype=dtype, device=Cs_d[0].device)

    if k == 2:
        V_gen = _generalized_eig_k2(Cs_d[0], Cs_d[1], eps=eps, dtype=dtype)
        V = _polar_orthogonal(V_gen)
        method = "gen_eig_k2 + polar + cs_jacobi"
    else:
        raise NotImplementedError(
            f"k={k}: warm-start for k>2 not implemented (cs_jacobi loop supports general k). "
            "The current LLM+VLM-LM application is k=2."
        )

    Ms_stack = torch.stack(
        [V.t() @ C @ V for C in Cs_d], dim=0
    )

    schedule = _round_robin_pairs(V.shape[0])
    sweeps_done = 0
    for _ in range(jacobi_sweeps):
        _cs_jacobi_sweep(
            Ms_stack, V, alpha_t,
            schedule=schedule, skip_tol=jacobi_skip_tol,
        )
        sweeps_done += 1
        fg_cost, _ = _fg_cost_and_residuals(Ms_stack, alpha_t)
        if fg_cost < fg_cost_tol:
            break

    fg_cost, off_diag_residual = _fg_cost_and_residuals(Ms_stack, alpha_t)

    lam = torch.diagonal(Ms_stack, dim1=-2, dim2=-1).clamp_min(0.0) + eps

    if k == 2:
        commutator = Cs_d[0] @ Cs_d[1] - Cs_d[1] @ Cs_d[0]
        c_num = torch.linalg.matrix_norm(commutator).item()
        c_denom = (
            torch.linalg.matrix_norm(Cs_d[0]).item()
            * torch.linalg.matrix_norm(Cs_d[1]).item()
        )
        commutator_residual = c_num / max(c_denom, torch.finfo(dtype).tiny)
    else:
        commutator_residual = float("nan")

    return JointDiagResult(
        V=V,
        V_inv=V.t().contiguous(),
        eigvals_per_modality=lam,
        off_diagonal_residual=off_diag_residual,
        fg_cost=fg_cost,
        commutator_residual=commutator_residual,
        method=method,
        n_jacobi_sweeps=sweeps_done,
    )


def per_direction_eigenvalues(V: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """λ_j := (V^T C V)_{j,j}. Helper for diagnostics (not used in merge)."""
    return torch.diagonal(V.t() @ C @ V)
