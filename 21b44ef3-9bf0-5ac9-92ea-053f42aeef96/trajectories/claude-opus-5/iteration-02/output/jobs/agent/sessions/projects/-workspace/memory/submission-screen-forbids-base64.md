---
name: submission-screen-forbids-base64
description: "The 8-digit-adder task's graded submission.py is screened for imports; base64 is rejected and the file never runs, scoring 0."
metadata: 
  node_type: memory
  type: project
  originSessionId: 21ed282c-53e4-4b10-87b6-3f352c67a8b7
  modified: 2026-09-03T13:08:23.129Z
---

On the "minimal 8-digit addition transformer" task, `/workspace/submission.py` is
statically screened before it is executed. Attempt 1 inlined the trained weights as a
base64 blob; the screen flagged `import base64` as a forbidden import, the file was
never run, and the attempt scored 0.0 despite the model itself reaching >=99.9%
held-out accuracy.

**Why:** the graded file is expected to contain the model and its inference path only,
with a small dependency surface. Anything that looks like data plumbing (base64,
pickle, zlib, json, numpy, random) is treated as outside that surface.

**How to apply:** write weights into submission.py as plain Python nested lists of
floats and `torch.tensor(...)` them in `build_model()`. `repr(float(x))` on a float32
value round-trips exactly and a few hundred values is only ~10 KB. Keep the file's
imports to `torch` / `torch.nn` alone. Verify with an AST scan of the emitted file
before shipping. See [[tiny-adder-architecture-that-works]].
