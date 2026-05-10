# method/ — E-Pull (Entropy-gated Pull)

Constrained-Pareto direction-wise merging in the **orthogonal** common
eigenbasis of the per-modality activation covariances. Closed form, no
hyperparameters beyond RegMean's `α`/`ε`. See paper §Method for derivations
and theorems.

## Closed form

Under the paper's CPC ansatz `C_i = V Λ_i V^T` with `V^T V = I`:

```
π_{j,i} = α_i λ_{i,j} / Σ_l α_l λ_{l,j}        # routing
H_j     = -Σ_i π_{j,i} log π_{j,i}             # modality entropy at v_j
g_j     = 1 - H_j / log k                      # entropy gate ∈ [0,1]
m_j^*   = Σ_i π_{j,i} (W_i - W_base) v_j       # RegMean per-direction
m_j^**  = (1 - g_j) m_j^* + g_j (W_{r*(j)} - W_base) v_j   # E-Pull
W^**    = W_base + Σ_j m_j^** v_j^T            # reassemble (V^T = V^{-1})
```

`g_j → 0` (uniform routing) recovers RegMean; `g_j → 1` (one-hot routing) is
direction-wise winner-take-all. The gate is parameter-free.

## Files

- `covariance.py` — input-Gram collection per linear layer (skips `down_proj`)
- `joint_diag.py` — orthogonal FG joint diagonalization (gen-eig + polar warm
  start, then parallel Cardoso-Souloumiac Jacobi sweeps)
- `merge.py` — per-layer E-Pull combination + state-dict pipeline
- `cli.py` — end-to-end: load → calibrate → merge → save
- `_self_test.py` — closed-form identities + paper-loss consistency +
  modality-swap symmetry + reviewer regressions

## Joint diagonalization

We compute the orthogonal `V` minimizing the FG objective
`Σ_i α_i ||off(V^T C_i V)||²_F` in three stages:

1. **Generalized eigvec (k=2)** via Cholesky whitening — non-orthogonal
   `V_gen` that exactly diagonalizes both `C_1, C_2`. Sidesteps the sum-PCA
   degeneracy mode (`eigh(Σ α_i C_i)` returns an arbitrary basis when the
   weighted average is degenerate, even under exact CPC).
2. **Polar projection** to the orthogonal manifold — `V_warm = polar(V_gen)`.
   Under exact CPC with shared orthogonal `V_true`, this recovers `V_true`
   exactly (verified by self-test); under approximate CPC it is the closest
   orthogonal matrix to the joint diagonalizer — already a strong start.
3. **Cardoso-Souloumiac Jacobi sweeps** to descend the FG cost on the
   orthogonal manifold. Each sweep schedules `d-1` rounds of `d/2` disjoint
   Givens rotations using a round-robin tournament; rotations within a round
   are batched. The Givens angle is
   `θ = 0.25 · atan2(2·Sxy, Sxx − Syy)`
   (the 0.25 factor follows from the double-angle relation: a coordinate-
   space Givens by `θ` corresponds to a `(a, b)`-space rotation by `2θ`).
   Verified analytically and numerically against a brute-force grid.

The sweep count `cfg.jacobi_sweeps` is configurable (default 2). Under exact
CPC the warm start is already optimal, so all rotations no-op (verified).
Under approximate CPC the FG cost descends quadratically near optimum; 1–2
sweeps are typically sufficient. The pipeline early-stops when `fg_cost`
falls below `fg_cost_tol`.

### Diagnostics

Per-layer stats expose three residuals:

- `off_diagonal_residual` — `mean_i ||off-diag(V^T C_i V)||_F /
  ||V^T C_i V||_F` on the final orthogonal `V`. Direct CPC-quality measure
  on the chosen basis.
- `fg_cost` — the normalized FG objective
  `Σ α_i ||off||²_F / Σ α_i ||V^T C_i V||²_F`.
- `commutator_residual` — `||C_1 C_2 − C_2 C_1||_F / (||C_1||_F · ||C_2||_F)`,
  estimator-independent. Commuting SPD matrices share an orthogonal
  eigenbasis (CPC), so this gauges how well the data fits the CPC ansatz.

`n_jacobi_sweeps` is also recorded per layer (after early-stop).

## Limitation

`k > 2` raises `NotImplementedError`. The Jacobi loop already supports
general `k`, but the gen-eig + polar warm start is currently k=2-only.
Lifting to `k > 2` would either initialize `V` at the identity and rely on
Jacobi alone, or chain whitenings — left as future work. The current
LLM + VLM-LM application is `k = 2`.

## Running

```bash
# 1. Self-test the math (no GPU, no model load)
python -m method._self_test

# 2. End-to-end LLM + VLM-LM merge (extract VLM-LM first, see ../extraction/)
python -m method.cli \
    --base   /path/to/Qwen2.5-7B \
    --models llm:/path/to/Qwen2.5-7B-Instruct \
             vlm_lm:/path/to/extracted/qwen25vl_7b_lm \
    --output /path/to/merged/qwen25_epull \
    --n-samples 128 \
    --save-stats /path/to/merged/qwen25_epull/layer_stats.json
```

## Cost

`O(d^3)` per linear layer for the warm start (Cholesky + eigh + SVD) — the
same order as RegMean. Each Jacobi sweep is `O(k d^3)` work split across
`d − 1` parallel rounds; in fp32 at `d = 3584` we measure ≈4 s for the warm
start and ≈16 s per Jacobi sweep on a single A6000. Entropy-gate computation
is `O(k d)` and negligible. No additional hyperparameters beyond `(α, ε)`.

## Down-projection

Per the paper, MLP down-projections (input dim = intermediate size) are
excluded from active merging — dense covariance is infeasible at that scale.
Instead they take the dominant-energy modality, scored by `α_i tr(C_i)`,
which only requires a running trace and not the full Gram.
