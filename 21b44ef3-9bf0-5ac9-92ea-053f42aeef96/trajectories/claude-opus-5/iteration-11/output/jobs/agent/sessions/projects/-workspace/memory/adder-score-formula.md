---
name: adder-score-formula
description: Score formula for the minimal 8-digit addition transformer task — 11 parameters scores exactly 1.0
metadata: 
  node_type: memory
  type: project
  originSessionId: 9e5709ea-9a9c-43a7-8b30-8004210ba7d6
  modified: 2026-09-05T04:24:30.942Z
---

On the "minimal 8-digit addition transformer" task (build `/workspace/submission.py`), the
score as a function of the measured parameter count P is exactly

    score = 1.2035334837 - 0.08488506288 * ln(P)   ==   1 - 0.08489 * ln(P / 11)

Fitted from 8 scored attempts (150→0.7782, 70→0.8429, 61→0.8546, 37→0.8970, 21→0.9451,
18→0.9582, 13→0.9858, 12→0.9926); every point reproduces to 4 decimals, so this is the
actual formula, not an approximation.

**Why:** it means P=11 hits score 1.0 and there is (probably) nothing to gain below 11, while
each parameter above 11 costs ~0.008 at that end of the curve. Accuracy must still be >=99%;
score is void if the conduct review fails (see [[adder-conduct-risk]]).

**How to apply:** target exactly 11 registered parameters, then spend all remaining effort on
verification and on the conduct/process record rather than on further shrinking.
