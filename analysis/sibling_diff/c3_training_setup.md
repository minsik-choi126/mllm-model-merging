# C3 — Training experiment setup (architectural causality)

**Goal** — Direct causal test of the hypothesis: *injecting QK-RMSNorm into a
non-QK-norm LLM provides architectural protection against VL-induced IFEval
degradation*.

**Advisor's guidance** — 2-GPU node, LLaVA recipe, 7B would take 2–3 days.
Small Qwen models (0.5B/0.6B) don't have public VLM variants, so we use
**Qwen2.5-3B / Qwen3-4B scale** for the first round.

---

## 1. Hypothesis and falsifiable prediction

**H1**: Vanilla Qwen2.5-3B-Instruct → standard LLaVA training → IFEval drops
substantially (predict ≈ –6 to –10 pt) (i.e. the phenomenon replicates at
small scale).

**H2**: Qwen2.5-3B-Instruct **with identity-init q_norm/k_norm modules
injected** → identical LLaVA training → IFEval drop is **significantly
smaller** (predict ≈ –2 to –5 pt).

**Decision rule** — If `Δ(H2) − Δ(H1) ≥ 3 pt` (i.e. injection recovers ≥ 3
pt of the drop), the architectural causality claim is supported.

Confounds to control for: same base LLM, same data, same recipe, same
hyperparameters, same seed. Only the architectural delta differs.

---

## 2. Models and variants

| Code | Base LLM | Vision encoder | QK-norm? | Training | Purpose |
|---|---|---|:-:|---|---|
| **L0** (baseline) | Qwen2.5-3B-Instruct | none | ✗ | none | text-only IFEval baseline |
| **A1** (vanilla VL) | Qwen2.5-3B-Instruct | CLIP-ViT-L/336 | ✗ | LLaVA recipe | replicate the phenomenon at 3B |
| **A2** (injection) | Qwen2.5-3B-Instruct + q_norm/k_norm identity-init | CLIP-ViT-L/336 | **✓** | LLaVA recipe (same) | **the C3 test** |
| **R1** (Qwen3 reference) | Qwen3-4B-Instruct | (use existing Qwen3-VL-4B if available) | ✓ native | (existing) | sanity that Qwen3 family preserves IF at 4B too |

The critical comparison is **A2 vs A1** (same base, same training, only QK-norm
architectural delta).

---

## 3. Code modifications

### 3.1 QK-norm injection into Qwen2.5 attention

```python
# qwen2_5_with_qknorm.py
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2RMSNorm

class Qwen2AttentionWithQKNorm(Qwen2Attention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        # Identity-init γ — RMSNorm with γ=1 acts as plain RMSNorm
        self.q_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # Initialize γ to 1 (identity at start)
        torch.nn.init.ones_(self.q_norm.weight)
        torch.nn.init.ones_(self.k_norm.weight)

    def forward(self, hidden_states, position_embeddings, attention_mask, **kw):
        bsz, q_len, _ = hidden_states.size()
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        # NEW: per-head RMSNorm with learnable γ
        q = self.q_norm(q)
        k = self.k_norm(k)
        # standard: transpose, RoPE, attention, etc.
        ...
```

**Effort**: ~1 day to write + verify.

### 3.2 LLaVA-style VLM wrapper

Use LLaVA-NeXT codebase as base
([github.com/LLaVA-VL/LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)) and:
- Replace `LlavaQwenForCausalLM` reference to `Qwen2.5-3B-Instruct` (or the
  QK-norm-injected variant).
- Confirm projector MLP matches LLaVA-1.5 design (2-layer MLP, GELU).
- Confirm vision encoder is CLIP-ViT-L-336.

**Effort**: ~1 day if using existing codebase; ~3 days if writing from
scratch.

---

## 4. Training recipe (LLaVA-1.5 standard)

### Stage 1 — Feature alignment (pretrain)

- **Frozen**: vision encoder, LLM
- **Trainable**: projector MLP only
- **Data**: LCS-558K (LLaVA-Pretrain dataset, ~558K image-caption pairs)
- **Hyperparameters** (LLaVA-1.5 official):
  - Batch size per GPU: 32 (effective 256 with gradient accumulation)
  - LR: 1e-3
  - Epochs: 1
  - Optimizer: AdamW
  - Scheduler: cosine decay, warmup 0.03
- **Time on 2× A100/A6000**: ~6–10 hours for 3B model

### Stage 2 — Visual instruction tuning

- **Frozen**: vision encoder (or unfrozen for v1.6 recipe)
- **Trainable**: projector + LLM
- **Data**: LLaVA-Mix-665K (665K mixed instruction-following pairs)
- **Hyperparameters**:
  - Batch size per GPU: 16 (effective 128 with grad accumulation)
  - LR: 2e-5
  - Epochs: 1
  - Optimizer: AdamW
- **Time on 2× A100/A6000**: ~30–48 hours for 3B model

**Total per condition**: ~36–58 hours (~1.5–2.5 days)

---

## 5. Hardware and data

### Hardware
- 2 × NVIDIA A6000 (49 GB each) or A100 (40/80 GB)
- ~500 GB local SSD for data caching

### Data download
| Dataset | HF / source | Size |
|---|---|---|
| LCS-558K (Stage 1) | `liuhaotian/LLaVA-Pretrain` | ~25 GB images + json |
| LLaVA-Mix-665K (Stage 2) | `liuhaotian/LLaVA-Instruct-150K` + COCO + GQA + … | ~30 GB |
| Vision encoder | `openai/clip-vit-large-patch14-336` | ~600 MB |

**Time to prepare**: ~2 hours download + extraction.

---

## 6. Evaluation protocol

Per condition (A1, A2):

1. **Pre-VL** (after Stage 0): measure IFEval on the base text-only model.
   For A2, the identity-init QK-norm should not change behavior — confirm
   IFEval ≈ Qwen2.5-3B-Instruct baseline.
2. **Post-Stage-1**: measure IFEval (sanity; should not move much since LLM
   is frozen).
3. **Post-Stage-2**: measure IFEval (critical) + 6-task vision benchmark
   (VQA, MMBench, etc.).
4. Repeat with 2 seeds per condition for variance estimate.

**Comparison table** to fill in:
| Condition | IFEval prompt-strict | Vision avg | IFEval Δ |
|---|---|---|---|
| L0 (base Qwen2.5-3B-Instruct) | (~70-75 expected) | — | 0 |
| A1 (vanilla VL train) | ? | ? | ? |
| A2 (QK-norm + VL train) | ? | ? | ? |

---

## 7. Timeline (optimistic)

| Phase | Days | Parallelism |
|---|---:|---|
| Code mods (Qwen2.5 + QK-norm class, LLaVA pipeline integration) | 2 | serial |
| Data preparation | 0.5 | parallel |
| A1 Stage 1+2 training | 2 | both GPUs |
| A2 Stage 1+2 training | 2 | both GPUs (sequential after A1) |
| Evaluation | 0.5 | — |
| **Total** | **~7 days** | |

If A1 and A2 can run on separate nodes (4 GPUs): could halve to ~4 days.

Per advisor's guidance: 7B would scale these up to ~3 days each → ~10–12
days total. Starting at 3B for faster iteration.

---

## 8. Risks and contingencies

### R1 — Phenomenon may not replicate at 3B
If A1 doesn't show a ≥5 pt IFEval drop (i.e. 3B doesn't manifest the same
phenomenon as 7B), the C3 test is uninformative at this scale.
- **Mitigation**: also evaluate the existing Qwen2.5-VL-3B-Instruct
  (already downloaded) for IFEval drop magnitude. If that shows ≥5 pt
  drop, the phenomenon does exist at 3B and our recipe should reproduce
  it.

### R2 — QK-norm γ may not learn meaningful amplifiers
With identity init, γ might stay near 1 throughout training (gradient
flow weak through RMSNorm scale). Then QK-norm injection wouldn't actually
produce the amplifier structure Qwen3 has.
- **Mitigation 1**: report A2 IFEval anyway — even no-amplification γ-norm
  may help by bounding magnitudes.
- **Mitigation 2**: warm-start γ from Qwen3-4B-Instruct's γ values (cross-
  family transfer). This contaminates the test slightly but is fallback.
- **Mitigation 3**: add a regularizer encouraging γ heavy-tailedness.

### R3 — Training instability
Inserting RMSNorm in attention may destabilize training initially.
- **Mitigation**: use lower LR for the new γ parameters, gradient clipping,
  warmup.

### R4 — Different recipe than Qwen team's actual recipe
Qwen2.5-VL-7B was trained with Qwen team's proprietary recipe, not LLaVA.
A1 trained with LLaVA recipe may show different IFEval drop than the
reference Qwen2.5-VL-7B (-9 pt). This is OK — what matters for C3 is the
*differential* between A1 and A2 with the same recipe.

### R5 — Cost / time
2-3 days per condition × 2-3 conditions can run a week or more.
- **Mitigation**: start with single seed; only run additional seeds if
  signal is borderline.

---

## 9. What this experiment DOES and DOES NOT prove

**DOES**:
- Direct causal evidence: with everything else fixed, adding QK-norm
  architecture changes IFEval drop.
- Isolates the architectural feature from data/recipe/RLHF confounds.

**DOES NOT**:
- Prove QK-norm is the ONLY architectural feature that helps.
- Prove the same effect scales to 7B / 72B.
- Replicate Qwen team's exact recipe.

---

## 10. Pre-flight checklist

- [ ] Hardware confirmed: 2 GPUs available for ~7 days
- [ ] Download Qwen2.5-3B-Instruct (~7 GB)
- [ ] Download CLIP-ViT-L-336 (~600 MB)
- [ ] Download LLaVA-Pretrain (~25 GB)
- [ ] Download LLaVA-Mix-665K (~30 GB)
- [ ] Set up LLaVA-NeXT codebase (clone, install deps)
- [ ] Implement `Qwen2AttentionWithQKNorm` class
- [ ] Unit-test forward pass of injected model (compare to base, should be
      identical with γ=1)
- [ ] Measure baseline IFEval of Qwen2.5-3B-Instruct (paper-grade)
- [ ] Confirm existing Qwen2.5-VL-3B-Instruct shows IFEval drop (replicates
      phenomenon at 3B)
- [ ] Run A1 (vanilla) Stage 1+2
- [ ] Eval A1 IFEval
- [ ] Run A2 (QK-norm injected) Stage 1+2 with identical seed
- [ ] Eval A2 IFEval
- [ ] Report Δ(A2 − A1) and decision

---

## 11. Decision points

After Stage 2 of A1:
- If A1 IFEval drop < 3 pt → phenomenon not strong at 3B, skip A2 and
  switch to 7B (more compute but stronger signal).
- If A1 IFEval drop ≥ 5 pt → proceed with A2.

After Stage 2 of A2:
- If Δ(A2 − A1) ≥ 3 pt → architectural causality confirmed at 3B. Proceed
  to 7B replication.
- If Δ(A2 − A1) < 2 pt → injection insufficient at this scale. Try R2
  mitigations (warm-start γ).
- If Δ(A2 − A1) ≥ 5 pt → strong positive. Paper-ready C3.

---

## Cost-benefit summary

- **Cost**: ~7 days, 2 GPUs (= ~1 GPU-week)
- **Benefit if positive**: clean causal evidence; paper acceptance probability
  rises 20-30%
- **Benefit if negative**: refutes architectural causality hypothesis; paper
  must pivot to "QK-norm correlates with robustness but isn't sufficient
  alone".

ROI: high. C3 is the single highest-leverage experiment in the project plan.
