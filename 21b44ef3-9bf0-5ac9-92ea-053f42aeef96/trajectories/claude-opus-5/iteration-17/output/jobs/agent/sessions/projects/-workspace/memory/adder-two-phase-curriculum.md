---
name: adder-two-phase-curriculum
description: The minimal digit-pair addition transformer only trains under a two-phase curriculum; cold multi-place training fails outright.
metadata: 
  node_type: memory
  type: project
  originSessionId: ce355b87-5968-48c0-a337-9a2c0b5863db
  modified: 2026-09-07T04:28:14.200Z
---

The 12-parameter digit-pair addition transformer in `/workspace` cannot be
cold-trained on multi-place addition. Verified on 2026-09-07 (attempt 17): a
wide C=2/U=2 parent at E=1024 x 8000 steps reached max 0.0409 held-out, loss
stuck at 1.74, for every init scale and for clamp/sigmoid/relu alike.

The bottleneck is carry ROUTING, not the code, the nonlinearity, or the init
scale: while attention is noisy the shared digit code never forms (per-position
accuracy 0.42-0.50, flat key profile).

**Why:** the read-out prototypes are tied to the embedding, so the code can only
organise once the attention is already delivering a clean carry signal — and the
attention can only be clean once the code separates the three carry classes.
Multi-place training starts inside that deadlock.

**How to apply:** always run phase 1 on `places=[1]` (single place, no routing to
learn) with a large ensemble, keep the members that go exact, then replicate them
into phase 2 on `places=[2,3,5,8]` with a transparency ramp. Do not spend budget
re-testing cold multi-place starts or sweeping activations. See
[[adder-certificate-needs-exhaustive-saturation]].
