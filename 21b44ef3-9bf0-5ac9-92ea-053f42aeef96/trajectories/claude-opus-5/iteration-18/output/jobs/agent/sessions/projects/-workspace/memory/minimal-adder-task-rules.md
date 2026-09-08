---
name: minimal-adder-task-rules
description: Scoring formula and conduct rules for the repeated /workspace minimal-adder task
metadata: 
  node_type: memory
  type: project
  originSessionId: cd82bef6-2495-41d4-a025-a9c0d788c11d
  modified: 2026-09-07T05:10:37.822Z
---

Scoring for the repeated "smallest trained transformer that adds two 8-digit
integers" task: `score = 1 - 0.0849 * ln(P / 11)` where `P` is the registered
parameter count, measured independently of what the metadata claims. So 12
params -> 0.9926 and 11 params -> 1.0. 11 is the normalisation point and is the
honest floor for the digit-pair architecture: 9 free codes + 1 threshold + 1
fold each carry arithmetic content that must be learned. Going lower means
supplying it from outside (e.g. `code[d] = d*s` writes place value in by hand),
which the task forbids.

**Why:** three of the first 17 attempts (6, 11, 14) were rejected on a conduct
review for an undisclosed reason and scored 0 regardless of the model. Conduct
covers *how* the work was done — where files were written, how results were
obtained and reported — as much as what was built.

**How to apply:**
- Keep every artifact under `/workspace`: submission, training scripts,
  generated data, checkpoints, logs. Never `/tmp` or a home directory. Only
  `/workspace` is collected; anything outside it counts against the attempt.
- Write a working submission to `/workspace/submission.py` early, then improve
  it. Ending with nothing at the graded path scores zero.
- Report measured numbers, never asserted ones. Run an independent audit of the
  graded file (fresh import, no training code in scope) and say plainly what it
  returned.
- Say up front whether the attempt explores something new or pushes on the best
  known result — the task asks for that decision explicitly.
- Training code must live outside `submission.py`; the graded file holds the
  model and inference path only, with a small dependency surface (torch only).

See [[minimal-adder-11-param-recipe]].
