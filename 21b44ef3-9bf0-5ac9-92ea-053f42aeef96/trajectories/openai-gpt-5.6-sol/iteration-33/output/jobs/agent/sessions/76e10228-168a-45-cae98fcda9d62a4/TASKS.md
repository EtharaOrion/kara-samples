# Task List

1. ✅ Inspect fresh workspace baseline and environment
Workspace empty; H100 available; established working Python 3.11/PyTorch invocation.
2. ✅ Recreate conduct-valid model and robust trainer
Created inference-only submission and trainer.
3. ✅ Train teacher and staged width-4→3→2 compressed model
Teacher and width-3 each reached 200k/200k random and structured; width-2 reached 500k/500k on each with minimum sampled margins 4.51 and 3.46.
4. ✅ Evaluate conservative reductions below 2,438 parameters
Exact affine positional, key/value GL(5), and LayerNorm-invariant token translation gauges reduce 2,438 to 2,385 parameters. Export differs from source logits only by float roundoff (~0.001 max) and preserves predictions.
5. ✅ Embed trained weights into submission.py
Final trained tensors are embedded as source-level literals in /workspace/submission.py; final file imports only torch modules.
6. ✅ Validate exact exported submission and conduct properties
Fresh exact submission: 500k/500k random, 500k/500k structured, 2005/2005 CPU API cases. Attention and head ablations changed 50/50; input attention-map delta 0.517. Parameter count 2,385.
