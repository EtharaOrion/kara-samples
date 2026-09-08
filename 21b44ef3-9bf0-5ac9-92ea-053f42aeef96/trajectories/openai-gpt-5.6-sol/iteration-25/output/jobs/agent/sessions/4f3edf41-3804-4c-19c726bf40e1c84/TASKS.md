# Task List

1. ✅ Inspect fresh workspace baseline and available runtime
Workspace was empty; used H100 with system Python/PyTorch.
2. ✅ Implement conduct-valid grouped-query model and trainer
Created inference-only submission architecture plus external training/export tooling.
3. ✅ Train robust teacher and compressed candidates
Width-4 stabilized; width-3 recovered successfully; rejected collapsed width-2 trajectory.
4. ✅ Validate exact exported submission and robustness
Exact file: 499,998/500k random, 500k/500k structured, 961/961 edges, 5k/5k CPU API; attention and output ablations each changed 64/64.
5. ✅ Finalize smallest qualifying /workspace/submission.py
Final is trained width-3 rank-2 grouped-query causal autoregressive transformer with 2,550 parameters; cleaned crash dumps and discarded candidate artifacts.
