---
name: tiny-adder-architecture-that-works
description: "Working recipe for the minimal 8-digit addition transformer - digit-pair tokens, one carry-scan attention head, seed lottery via batched ensemble."
metadata: 
  node_type: memory
  type: project
  originSessionId: 21ed282c-53e4-4b10-87b6-3f352c67a8b7
  modified: 2026-09-03T13:08:36.966Z
---

Recipe that reaches 100% held-out exact match on 8-digit addition at ~150-185
parameters (task: "Minimal 8-digit addition transformer", `/workspace`).

**Layout** (length 10, least-significant place first): pos 0 = sink/no-carry anchor,
pos 1..8 = digit place i embedded as `emb[a_i] + emb[b_i]` from one shared 10-row
table, pos 9 = carry-out slot. Every position predicts its own sum digit, so one
forward pass yields all nine digits. Unembedding tied to `emb`. Parameter-free
RMSNorm. A learned relative-position bias (or a 2-parameter ALiBi-style ramp:
slope + a separate self term) supplies recency; content supplies transparency.

**Algorithm the attention learns:** a query place attends to the most recent earlier
place that is *not* carry-transparent (a_j+b_j != 9) and reads whether that place
generated a carry. Genuinely input-dependent; freezing the attention map at its
batch mean drops accuracy to ~8-30%.

**Two things that mattered most:**
- Seed variance dominates at these sizes (same 221-param config: one seed 89%,
  another 99.4%). Train 192-256 copies simultaneously as one batched ensemble
  (leading E axis, `torch.bmm`) - ~70x the model-steps/s of one process, since tiny
  models are launch-bound and separate GPU processes just serialize. Adam is
  elementwise so this is equivalent to independent runs; clip gradients per member.
- With two blocks, block 0's attention learns nothing (0% argmax variation across
  inputs) - dead weight, and exactly the "fixed pattern dressed up as attention"
  the task warns about. Use ONE attention layer, preceded by a token-wise MLP that
  computes the transparent/generate flags (attention keys are linear in
  `emb[a]+emb[b]`, so they cannot detect `a+b == 9` without a prior nonlinearity).

See [[submission-screen-forbids-base64]] for how to package the result.
