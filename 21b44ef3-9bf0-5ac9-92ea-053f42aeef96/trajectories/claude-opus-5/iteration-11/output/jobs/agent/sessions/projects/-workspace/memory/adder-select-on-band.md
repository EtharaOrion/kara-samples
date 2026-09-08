---
name: adder-select-on-band
description: "For the minimal adder, pick members by the derived alpha band, not by accuracy; band-randomised training collapses."
metadata: 
  node_type: memory
  type: project
  originSessionId: 9e5709ea-9a9c-43a7-8b30-8004210ba7d6
  modified: 2026-09-05T05:49:25.391Z
---

In the minimal digit-pair adder (see [[adder-score-formula]]), the clamp bank's
slope `alpha` is **not** an independent don't-care once the scale gauge is spent on
`e1 = 1`. Its admissible band is derived from the learned weights:

    d_tr = max |code[a]+code[b] - theta| over a+b == 9   (transparent pairs)
    d_nt = min |code[a]+code[b] - theta| over a+b != 9
    band = [ 1/d_nt , (1 - |lam|*P/kw)/d_tr ]

**Select members by this band width, not by accuracy.** Accuracy at one alpha is a
weak criterion: a member can score 1.0 on every sweep and have a band of only
[1.72, 2.02]. Fine-tuning at a single fixed alpha actively *degrades* conditioning,
because nothing penalises the code drifting away from an exact ramp. Re-ranking
existing members by band width put a *less* fine-tuned member on top — band 7.8x
wide vs 1.17x — and that one shipped.

**Why:** a wide band means the code is close to a true ramp and the transparent pairs
sit tightly on theta, which is the same thing as the arithmetic being sharp. It is
also what makes "the constant is a don't-care" an honest claim rather than a tuned
value, which the conduct review cares about (see [[adder-conduct-risk]]).

**How to apply:** compute the band per member and rank on it; ship a round alpha
strictly inside, and assert that in the build step. Do *not* try to train for a wide
band — randomising alpha over a band collapses every time (warm or cold, read-out
scale free or fixed), because once `alpha|x-theta| > 1` for a transparent pair the
clamp saturates and the gradient through the notch dies. Ramping alpha upward does
not collapse but yields narrower-banded members than it started from. Band width is
something to select for, never to optimise for.

Corollary worth keeping: an independent certificate that assumes a mechanism earns
its keep by *contradicting* the training loop. A corners-built-before-cfg-mutation
bug had runs training at ramped alpha while scoring at the starting alpha, reporting
1.0 for members whose notch was fully closed; the certificate caught it (true
exact-match 0.62).
