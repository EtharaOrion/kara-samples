---
name: minimal-adder-11-param-recipe
description: Architecture and training recipe that reaches an 11-parameter transformer adding two 8-digit integers exactly
metadata: 
  node_type: memory
  type: project
  originSessionId: cd82bef6-2495-41d4-a025-a9c0d788c11d
  modified: 2026-09-07T05:10:27.860Z
---

The `/workspace` "smallest trained transformer that adds two 8-digit integers"
task is a repeated attempt (up to 50); the workspace starts empty each time.
The best known solution is **11 registered parameters** — `code_free (9,)`,
`knee (1,)`, `fold (1,)` — reached on 2026-09-07 (attempt 18), full pipeline
~6 min on one GPU.

Architecture: one block over per-place digit-pair tokens (LSB first, `(0,0)` pad
at both ends, `P = n+2`, position `p` emits answer digit `p-1`). Scalar residual
stream; token embeds as `code[a]+code[b]`; tied read-out `logits = -(r-code)^2`.
Two-unit clamp bank on one shared learned threshold `k` at two fixed slopes
`(8.0, 1.0)`: value `= u0`, key `= 400*(u1-u0)` (notched exactly where the units
disagree, i.e. at `a+b==9`). Two masks over the same key/value stream —
strictly causal gives carry-in, inclusively causal gives carry-out — plus a
recency bias `lam=-12`. Write-back `r = x + carry_in + fold*carry_out`.

Non-obvious things that took work to find:

- **Pin the two gauge freedoms as buffers** (`code[0]=0`, `carry_w=1`). That is
  what turns 13 free values into 11; it removes redundancy, not freedom.
- **Cold multi-place training deadlocks.** Warm up on `n=1` only (~20% of steps)
  before cycling places 1,2,3,5,8.
- **Train an ensemble on a leading axis** (E=4096 members, shared batch, AdamW is
  elementwise, clip per member). It is a seed lottery: ~11/4096 members converge.
- **Drop cross-entropy in the fine-tune stage** — with the logit scale pinned it
  just inflates the code scale. Use a geometric read-out margin hinge
  (`sqrt(d^2+1e-12)`) plus a class-agnostic saturation penalty over all 100 pairs.
- **Structural penalties must be class-agnostic**: ask every digit pair to be
  *decisive* about the gate, never say which pairs carry. Naming the classes
  would be smuggling in the arithmetic.
- Canonicalisation must read the value stream off the *actual* extreme tokens
  (`val_at(2*code[0])`, `val_at(2*code[9])`) — many members learn a descending
  code ramp, and assuming ascent silently breaks the substitution.

Verify with a float64 whole-domain certificate (gate class map -> analytic
attention-leakage bound -> read-out margin > 2x drift), all 3^8 carry-class
patterns, and a band study that sweeps each buffered constant with the learned
weights frozen. See [[minimal-adder-task-rules]].
