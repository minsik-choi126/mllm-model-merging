"""Sanity test of the E-Pull math against the closed-form properties.

Runs on tiny synthetic linear layers (no model load) and checks:

  1. Limit (uniform π_j → g_j = 0): E-Pull == RegMean (per-direction).
  2. Limit (one-hot π_j → g_j = 1): E-Pull == owner-take-all.
  3. Strict per-modality improvement on dominant directions
     (Theorem self_improve_app):  L_r^(j)(W^**) = η_j L_r^(j)(W^*)  for r=r*(j).
  4. Aggregate cost identity (Theorem aggregate):
         Σ_r α_r [L_r(W^**) - L_r(W^*)] = Σ_j Z_j g_j^2 ||u_{r*}^j - m_j^*||^2

All four are exact identities under the formulas; numerical tolerance ~1e-5.
"""

import math
import torch

from method.merge import EpullConfig, _epull_direction_combine
from method.joint_diag import joint_diagonalize


def _per_dir_loss_term(m_j: torch.Tensor, u_rj: torch.Tensor, lam_rj: float) -> torch.Tensor:
    """L_r^(j)(W) = λ_{r,j} ||m_j(W) - u_r^j||^2."""
    return lam_rj * (m_j - u_rj).pow(2).sum()


def _full_per_modality_loss(
    delta_W: torch.Tensor,    # (out, in)
    delta_W_r: torch.Tensor,  # (out, in)
    V: torch.Tensor,          # (in, in)
    lam_r: torch.Tensor,      # (in,)
) -> torch.Tensor:
    """L_r(W) = Σ_j λ_{r,j} ||(W-W_base) v_j - ΔW_r v_j||^2."""
    M = delta_W @ V          # (out, in) — columns are m_j(W)
    U_r = delta_W_r @ V      # (out, in)
    diff = M - U_r           # (out, in)
    return (lam_r.unsqueeze(0) * diff.pow(2).sum(0)).sum()


def synthesize(d_in=64, d_out=128, k=2, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    base = torch.randn(d_out, d_in, generator=g)
    Ws = [base + 0.1 * torch.randn(d_out, d_in, generator=g) for _ in range(k)]
    Cs = []
    for _ in range(k):
        A = torch.randn(d_in, 2 * d_in, generator=g)
        C = A @ A.t() / (2 * d_in)
        Cs.append(C)
    return base, Ws, Cs


def _run_epull(base, Ws, Cs, alphas):
    jd = joint_diagonalize(Cs, alphas, dtype=torch.float64)
    V = jd.V.to(torch.float64)
    V_inv = jd.V_inv.to(torch.float64)
    lam = jd.eigvals_per_modality.to(torch.float64)
    alphas_t = torch.tensor(alphas, dtype=torch.float64)
    delta_Ws = [(W - base).to(torch.float64) for W in Ws]
    delta_merged, info = _epull_direction_combine(
        delta_Ws, V, lam, alphas_t, V_inv=V_inv,
    )
    return V, V_inv, lam, delta_merged, info


def test_limit_uniform():
    """π_j uniform → E-Pull == RegMean (g_j=0)."""
    d = 32
    g = torch.Generator(device="cpu").manual_seed(1)
    base = torch.randn(d, d, generator=g)
    W1 = base + 0.1 * torch.randn(d, d, generator=g)
    W2 = base + 0.1 * torch.randn(d, d, generator=g)
    C = torch.randn(d, 2 * d, generator=g)
    C = C @ C.t() / (2 * d)
    Cs = [C, C]
    V, V_inv, lam, dmerged, _ = _run_epull(base, [W1, W2], Cs, [0.5, 0.5])

    # RegMean per-dir with shared C: m_j^* = (1/k) Σ ΔW_i v_j (since π=uniform)
    delta_Ws = [(W - base).to(torch.float64) for W in [W1, W2]]
    M_star = sum(0.5 * (dW @ V) for dW in delta_Ws)
    delta_regmean = M_star @ V_inv
    err = (dmerged - delta_regmean).abs().max().item()
    print(f"  test_limit_uniform: max-abs-err = {err:.3e}")
    assert err < 1e-8, f"expected RegMean, off by {err}"


def test_limit_onehot():
    """λ_{1,j} >> λ_{2,j} for all j → owner = modality 0, g_j ≈ 1, output ≈ ΔW_0."""
    d = 32
    g = torch.Generator(device="cpu").manual_seed(2)
    base = torch.randn(d, d, generator=g)
    W1 = base + 0.1 * torch.randn(d, d, generator=g)
    W2 = base + 0.1 * torch.randn(d, d, generator=g)
    A = torch.randn(d, 2 * d, generator=g)
    C1 = A @ A.t()
    # C2 ≈ identity (small but non-degenerate) so gen-eig stays well-conditioned;
    # C1 has much larger magnitude → π_j is nearly one-hot on modality 0 for all j.
    C2 = torch.eye(d, dtype=torch.float64) * 1e-3
    # Scale up C1 so λ_{1,j} >> λ_{2,j} after V^T C V, regardless of dir.
    C1 = C1 * 1e6
    V, V_inv, lam, dmerged, _ = _run_epull(base, [W1, W2], [C1, C2], [0.5, 0.5])
    delta_W1 = (W1 - base).to(torch.float64)
    err = (dmerged - delta_W1).abs().max().item()
    print(f"  test_limit_onehot: max-abs-err vs ΔW_owner = {err:.3e}")
    assert err < 1e-3, f"expected owner takes all, off by {err}"


def test_self_improve_dominant():
    """For r=r*(j): L_r^(j)(W^**) = η_j L_r^(j)(W^*)  with η_j = (H_j/log k)^2."""
    base, Ws, Cs = synthesize(d_in=24, d_out=20, k=2, seed=3)
    alphas = [0.5, 0.5]
    V, V_inv, lam, dmerged, info = _run_epull(base, Ws, Cs, alphas)

    delta_Ws = [(W - base).to(torch.float64) for W in Ws]
    Us = [dW @ V for dW in delta_Ws]                  # (out, in)
    alphas_t = torch.tensor(alphas, dtype=torch.float64)
    pi = (alphas_t.unsqueeze(1) * lam) / (alphas_t.unsqueeze(1) * lam).sum(0)
    M_star = sum(pi[i].unsqueeze(0) * Us[i] for i in range(2))   # (out, in)
    M_starstar = (dmerged @ V)                                   # (out, in)
    H = -(pi.clamp_min(1e-12) * pi.clamp_min(1e-12).log()).sum(0)
    log_k = math.log(2)
    eta = (H / log_k).pow(2).clamp(0, 1)                         # (in,)

    r_star = pi.argmax(0)
    max_rel = 0.0
    for j in range(M_star.shape[1]):
        r = int(r_star[j])
        u_rj = Us[r][:, j]
        lam_rj = float(lam[r, j])
        L_star = _per_dir_loss_term(M_star[:, j], u_rj, lam_rj).item()
        L_ss = _per_dir_loss_term(M_starstar[:, j], u_rj, lam_rj).item()
        if L_star < 1e-14:
            continue
        rel = abs(L_ss - eta[j].item() * L_star) / max(L_star, 1e-12)
        max_rel = max(max_rel, rel)
    print(f"  test_self_improve_dominant: max relative error = {max_rel:.3e}")
    assert max_rel < 1e-6, f"identity broken, rel err {max_rel}"


def test_aggregate_cost():
    """Σ_r α_r [L_r(W^**) - L_r(W^*)] = Σ_j Z_j g_j^2 ||u_{r*}^j - m_j^*||^2."""
    base, Ws, Cs = synthesize(d_in=20, d_out=24, k=2, seed=4)
    alphas = [0.4, 0.6]
    V, V_inv, lam, dmerged, info = _run_epull(base, Ws, Cs, alphas)

    delta_Ws = [(W - base).to(torch.float64) for W in Ws]
    Us = [dW @ V for dW in delta_Ws]
    alphas_t = torch.tensor(alphas, dtype=torch.float64)
    pi = (alphas_t.unsqueeze(1) * lam) / (alphas_t.unsqueeze(1) * lam).sum(0)
    M_star = sum(pi[i].unsqueeze(0) * Us[i] for i in range(2))
    delta_regmean = M_star @ V_inv

    H = -(pi.clamp_min(1e-12) * pi.clamp_min(1e-12).log()).sum(0)
    log_k = math.log(2)
    g = (1.0 - H / log_k).clamp(0.0, 1.0)
    Z = (alphas_t.unsqueeze(1) * lam).sum(0)
    r_star = pi.argmax(0)

    lhs = 0.0
    for r in range(2):
        L_ss = _full_per_modality_loss(dmerged, delta_Ws[r], V, lam[r]).item()
        L_st = _full_per_modality_loss(delta_regmean, delta_Ws[r], V, lam[r]).item()
        lhs += alphas[r] * (L_ss - L_st)

    rhs = 0.0
    for j in range(M_star.shape[1]):
        r = int(r_star[j])
        diff = (Us[r][:, j] - M_star[:, j]).pow(2).sum().item()
        rhs += float(Z[j].item()) * float(g[j].item()) ** 2 * diff

    rel = abs(lhs - rhs) / max(abs(rhs), 1e-12)
    print(f"  test_aggregate_cost: lhs={lhs:.6e}  rhs={rhs:.6e}  rel-err={rel:.3e}")
    assert rel < 1e-5, f"aggregate identity broken, rel err {rel}"


def test_degenerate_cpc_avg():
    """Regression: exact CPC where Σ α_i Λ_i is degenerate.

    Under sum-PCA, eigh of (Σ α_i C_i) returns an arbitrary basis inside the
    fully degenerate subspace and breaks downstream. Generalized eigvec
    recovers the joint eigenbasis up to column rescaling and permutation,
    and E-Pull is invariant to those.
    """
    d = 8
    g = torch.Generator(device="cpu").manual_seed(101)
    A = torch.randn(d, d, generator=g, dtype=torch.float64)
    V_true, _ = torch.linalg.qr(A)
    c = 1.5
    lam_1 = 0.5 + torch.rand(d, generator=g, dtype=torch.float64)
    lam_2 = 2 * c - lam_1
    C1 = V_true @ torch.diag(lam_1) @ V_true.t()
    C2 = V_true @ torch.diag(lam_2) @ V_true.t()
    avg = 0.5 * C1 + 0.5 * C2
    spread = (torch.linalg.eigh(avg)[0].max() - torch.linalg.eigh(avg)[0].min()).item()
    assert spread < 1e-10, f"construction error: avg eigval spread {spread}"

    base = torch.randn(6, d, generator=g, dtype=torch.float64)
    W1 = base + 0.1 * torch.randn(6, d, generator=g, dtype=torch.float64)
    W2 = base + 0.1 * torch.randn(6, d, generator=g, dtype=torch.float64)

    # Ideal merge in the true basis.
    lam = torch.stack([lam_1, lam_2], dim=0)
    alphas_t = torch.tensor([0.5, 0.5], dtype=torch.float64)
    pi = (alphas_t.unsqueeze(1) * lam) / (alphas_t.unsqueeze(1) * lam).sum(0)
    H = -(pi.clamp_min(1e-12) * pi.clamp_min(1e-12).log()).sum(0)
    g_ = (1.0 - H / math.log(2)).clamp(0, 1)
    delta_Ws = [W - base for W in [W1, W2]]
    Us = torch.stack([dW @ V_true for dW in delta_Ws], dim=0)
    M_star_true = (pi.unsqueeze(1) * Us).sum(0)
    r_star = pi.argmax(0)
    M_owner_true = torch.stack(
        [Us[r_star[j], :, j] for j in range(d)], dim=1
    )
    M_ss_true = (1 - g_).unsqueeze(0) * M_star_true + g_.unsqueeze(0) * M_owner_true
    delta_ideal = M_ss_true @ V_true.t()

    V, V_inv, lam_hat, delta_merged, _ = _run_epull(
        base, [W1, W2], [C1, C2], [0.5, 0.5]
    )
    rel = (
        torch.linalg.matrix_norm(delta_merged - delta_ideal).item()
        / torch.linalg.matrix_norm(delta_ideal).clamp_min(1e-12).item()
    )
    print(f"  test_degenerate_cpc_avg: relative Frobenius err vs ideal = {rel:.3e}")
    assert rel < 1e-5, f"sum-PCA failure mode reintroduced (rel err {rel})"


def test_modality_swap_symmetry():
    """Under the paper's CPC ansatz, the FG joint diagonalizer is unique up
    to column permutation/sign — and the closed-form merge is invariant to
    those — so swapping modalities (C_1, W_1, α_1) ↔ (C_2, W_2, α_2) must
    leave ΔW^** unchanged.

    We test on **exact orthogonal CPC** — the regime the paper assumes.
    Outside CPC the FG cost has multiple local optima and the merge no
    longer has a paper-meaningful "right answer", so we don't test there.
    """
    d_in = 16
    d_out = 12
    g = torch.Generator(device="cpu").manual_seed(303)
    A = torch.randn(d_in, d_in, generator=g, dtype=torch.float64)
    V_true, _ = torch.linalg.qr(A)
    lam_1 = 0.5 + torch.rand(d_in, generator=g, dtype=torch.float64)
    lam_2 = 0.5 + torch.rand(d_in, generator=g, dtype=torch.float64)
    C1 = V_true @ torch.diag(lam_1) @ V_true.t()
    C2 = V_true @ torch.diag(lam_2) @ V_true.t()
    base = torch.randn(d_out, d_in, generator=g, dtype=torch.float64)
    W1 = base + 0.1 * torch.randn(d_out, d_in, generator=g, dtype=torch.float64)
    W2 = base + 0.1 * torch.randn(d_out, d_in, generator=g, dtype=torch.float64)
    alphas = [0.4, 0.6]
    alphas_swapped = [0.6, 0.4]

    _, _, _, dmerged_a, _ = _run_epull(base, [W1, W2], [C1, C2], alphas)
    _, _, _, dmerged_b, _ = _run_epull(base, [W2, W1], [C2, C1], alphas_swapped)

    diff = torch.linalg.matrix_norm(dmerged_a - dmerged_b).item()
    norm = torch.linalg.matrix_norm(dmerged_a).clamp_min(1e-12).item()
    rel = diff / norm
    print(f"  test_modality_swap_symmetry (exact CPC): rel diff between (1,2) ↔ (2,1) orderings = {rel:.3e}")
    assert rel < 1e-6, f"FG estimator not symmetric under CPC; rel diff {rel}"


def test_validation():
    """Bad inputs should raise."""
    from method.merge import EpullConfig
    from method.joint_diag import joint_diagonalize

    try:
        EpullConfig(alphas=(1.0,))
    except ValueError:
        pass
    else:
        raise AssertionError("k<2 should raise")

    try:
        EpullConfig(alphas=(1.0, -0.2))
    except ValueError:
        pass
    else:
        raise AssertionError("negative alpha should raise")

    try:
        EpullConfig(alphas=(1.0, 0.0))
    except ValueError:
        pass
    else:
        raise AssertionError("zero alpha should raise")

    C = torch.eye(4, dtype=torch.float64)
    try:
        joint_diagonalize([C, C, C], [0.33, 0.33, 0.34])
    except NotImplementedError:
        pass
    else:
        raise AssertionError("k=3 should raise NotImplementedError until Jacobi lands")

    print("  test_validation: all guards raise as expected")


def test_paper_loss_consistency():
    """Under exact orthogonal CPC, verify the paper's per-direction loss form:

        L_r(W) = tr((W - W_r) C_r (W - W_r)^T)
               = Σ_j λ_{r,j} ||m_j(W) - u_r^j||²

    with λ_{r,j} = (V^T C_r V)_{j,j} on the *orthogonal* V from gen-eig + polar.

    This is the loss the closed-form theorem actually optimizes; it must match
    the trace form to machine precision when V truly diagonalizes each C_r,
    which is the case under exact CPC.
    """
    d_in = 16
    d_out = 12
    g = torch.Generator(device="cpu").manual_seed(202)
    A = torch.randn(d_in, d_in, generator=g, dtype=torch.float64)
    V_true, _ = torch.linalg.qr(A)
    lam_1 = 0.5 + torch.rand(d_in, generator=g, dtype=torch.float64)
    lam_2 = 0.5 + torch.rand(d_in, generator=g, dtype=torch.float64)
    C1 = V_true @ torch.diag(lam_1) @ V_true.t()
    C2 = V_true @ torch.diag(lam_2) @ V_true.t()
    base = torch.randn(d_out, d_in, generator=g, dtype=torch.float64)
    W1 = base + 0.1 * torch.randn(d_out, d_in, generator=g, dtype=torch.float64)
    W2 = base + 0.1 * torch.randn(d_out, d_in, generator=g, dtype=torch.float64)

    V, V_inv, lam, dmerged, _ = _run_epull(base, [W1, W2], [C1, C2], [0.5, 0.5])

    # Confirm V is orthogonal (the whole point of polar).
    orth_err = (V.t() @ V - torch.eye(d_in, dtype=torch.float64)).abs().max().item()
    assert orth_err < 1e-10, f"V not orthogonal, max(V^T V - I) = {orth_err}"

    W_merged = base + dmerged
    Cs = [C1, C2]
    Ws = [W1.to(torch.float64), W2.to(torch.float64)]
    # Use the raw (no ε) eigvals for the loss-equality check; ε is part of the
    # paper's algorithm for routing stability, not for the lemma's loss form.
    lam_raw = torch.stack(
        [torch.diagonal(V.t() @ C @ V).clamp_min(0.0) for C in Cs], dim=0
    )
    max_rel = 0.0
    for r in range(2):
        diff = W_merged - Ws[r]
        L_trace = (diff @ Cs[r] * diff).sum().item()
        m_W = (W_merged - base) @ V                       # cols are m_j(W)
        u_r = (Ws[r] - base) @ V                          # cols are u_r^j
        per_dir = (lam_raw[r].unsqueeze(0) * (m_W - u_r).pow(2).sum(0)).sum().item()
        rel = abs(L_trace - per_dir) / max(abs(L_trace), 1e-12)
        max_rel = max(max_rel, rel)
    print(
        f"  test_paper_loss_consistency: V orthogonal max-err = {orth_err:.2e},  "
        f"trace ↔ per-direction rel-err = {max_rel:.2e}"
    )
    assert max_rel < 1e-10, f"per-direction loss formula broken, rel err {max_rel}"


def main():
    print("[E-Pull self-test]")
    test_limit_uniform()
    test_limit_onehot()
    test_self_improve_dominant()
    test_aggregate_cost()
    test_degenerate_cpc_avg()
    test_paper_loss_consistency()
    test_modality_swap_symmetry()
    test_validation()
    print("OK — all 8 closed-form / regression / symmetry / validation checks passed.")


if __name__ == "__main__":
    main()
