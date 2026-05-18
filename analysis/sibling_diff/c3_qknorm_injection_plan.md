# C3 — Architectural causality experiment design

**Goal**: directly test whether adding QK-RMSNorm to a non-QK-norm architecture
(Qwen2.5) provides architectural protection against VL-adaptation-induced
IFEval degradation.

## Hypothesis

> *If QK-RMSNorm is the architectural feature that protects sink encoding
> from W-perturbation during VL adaptation, then injecting q_norm/k_norm modules
> into Qwen2.5-7B and performing the same VL adaptation should preserve
> IFEval substantially better than vanilla Qwen2.5-VL training.*

Quantitative target: if C3 is positive, the injected variant should show
IFEval drop closer to Qwen3-VL's −3pt than to Qwen2.5-VL's −9pt.

## Variants to compare (4-cell ablation)

| Variant | Architecture | Training | Expected IFEval drop |
|---|---|---|---|
| **(a) Base** | Qwen2.5-7B-Instruct | None (no VL) | 0 (baseline 72.09) |
| **(b) Vanilla VL** | Qwen2.5-7B + standard VL train | Vision-text data | ≈ −9pt (matches existing Qwen2.5-VL) |
| **(c) QK-norm injected, no train** | Qwen2.5-7B + identity-init q_norm/k_norm | None | Should match base (norm is identity) |
| **(d) QK-norm + VL train (the test)** | Qwen2.5-7B + q_norm/k_norm + VL train | Same as (b) | **Hypothesis: drop ≪ −9pt** |

The **critical comparison is (d) vs (b)**.

## Architectural modification

Insert two new modules into each Qwen2_5Attention layer:

```python
class Qwen2_5AttentionWithQKNorm(Qwen2_5Attention):
    def __init__(self, config):
        super().__init__(config)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states, ...):
        # standard projection
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim)
        # NEW: per-head RMSNorm with learnable γ
        q = self.q_norm(q)
        k = self.k_norm(k)
        # apply rope, attention, etc.
        ...
```

`Qwen3RMSNorm` = standard RMSNorm with γ initialized to ones (identity at init).

Implementation: patch `transformers/models/qwen2/modeling_qwen2.py` or
write a wrapper class.

## Training recipe

**Critical constraint**: must use *the same VL recipe* as our reference Qwen2.5-VL
to make the comparison fair. We don't have access to Qwen team's training data
or schedule, but we can approximate with a small-scale LLaVA-style training.

### Option A — Full LLaVA replication (expensive)
- Stage 1: feature alignment, 558K image-caption pairs (LAION/CC3M subset)
- Stage 2: visual instruction tuning, 665K mixed (LLaVA-Instruct-150K + ShareGPT4V + …)
- Compute: 3-4 A100-days
- Risk: small data, different recipe → might not reproduce -9pt baseline cleanly

### Option B — Small-scale proof-of-concept (recommended)
- Stage 1: skip (use pretrained CLIP/SigLIP vision tower frozen)
- Stage 2: visual instruction tuning on **a small subset** (50-100K samples of LLaVA-Instruct)
- Compute: 1-2 GPU-day
- Risk: smaller data → less dramatic IFEval drop → harder to see effect

### Option C — Reuse existing VL-adaptation artifacts (cheapest)
- Take Qwen2.5-VL-7B's vision tower + connector as-is
- Replace the text backbone with Qwen2.5-7B-Instruct + identity-init q_norm/k_norm
- Continue VL training (fine-tune connector + LM head + new γ on small data, e.g. 50K samples)
- Compute: 0.5-1 GPU-day
- Most realistic to "what would happen if Qwen2.5-VL had QK-norm from day 1"

### Recommended: Option C with caveats

- **Caveat 1**: this measures *what if we late-inject QK-norm into existing VL pipeline*, not *what if QK-norm was there from the start*. Reviewer may push back.
- **Caveat 2**: γ_q, γ_k start at identity → need some training to discover sink patterns. If we don't train enough, γ stays trivial and the QK-norm doesn't do anything useful.
- **Caveat 3**: training data subset choice introduces variance. Need to control for data effects.

## Falsifiable predictions

If C3 is run with reasonable choices:

**Positive outcome** (hypothesis confirmed):
- Variant (d) IFEval drop ≤ −5pt (less than half of vanilla (b)'s -9pt)
- Variant (d) vision performance comparable to (b)
- → "QK-norm injection provides architectural protection without sacrificing VL capability"

**Negative outcome** (hypothesis refuted):
- Variant (d) IFEval drop ≈ −9pt (same as vanilla)
- → QK-norm is not sufficient; cross-vendor differences must be due to other factors (data, RLHF, schedule)
- → Paper trajectory must pivot: emphasize correlation, not mechanism

**Partial outcome** (most likely):
- Variant (d) IFEval drop ≈ −6pt (improvement but not full immunity)
- → QK-norm provides partial protection; other factors also contribute

## Risks

1. **Vision degradation**: adding QK-norm might hurt vision performance if γ doesn't train well. Mitigation: monitor vision metrics during training.

2. **Insufficient training**: small data + identity init may leave γ untrained. Mitigation: warm-start γ from Qwen3-8B's γ_q/γ_k (transfer learning) — but this contaminates the test.

3. **Confounding**: even with same recipe, results may differ for orthogonal reasons (e.g., random seed sensitivity). Mitigation: run 2-3 seeds.

4. **Single-run noise**: IFEval has ~2pt stderr. Need multiple seeds to detect signal.

5. **Architectural compatibility**: Qwen2.5-VL-7B's vision tower expects specific LM behavior. Injecting QK-norm may break the alignment. Mitigation: continue training enough to re-align.

## Decision points before execution

1. **Code**: write `Qwen2_5AttentionWithQKNorm` class and verify forward pass works.
2. **Data**: pick a small VL-instruction-tuning subset (LLaVA-Instruct-665K subset?).
3. **Recipe**: pick Option A/B/C.
4. **Compute**: confirm GPU availability for 0.5-2 GPU-days.

## Timeline

If executing Option C:
- Code modification + testing: 1 day
- Continued training on subset: 1 day
- Eval (IFEval + vision 6-task): 0.5 day
- **Total: ~2.5 days**

Reduced version (skip training, just inject + identity γ + IFEval): a few hours.
Even this minimal version would tell us if architectural change alone helps.

## Minimal C3 (inference-only sanity)

Before full training, run **C3-lite**:
1. Take Qwen2.5-VL-LM (the extracted text backbone, currently damaged)
2. Inject `q_norm` / `k_norm` with γ=1 (identity)
3. Run IFEval

This is a 30-min experiment. Tests whether adding the normalization layer alone (without training γ) helps recover.

If positive (IFEval improves): architectural normalization itself contributes some protection → strong story.
If null (IFEval same as vanilla): γ training is essential → need full C3.
