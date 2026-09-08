# Task List

1. ✅ Inspect fresh workspace and baseline files
Workspace empty; configured CUDA PyTorch.
2. ✅ Create compliant inference submission and training pipeline
Inference and external trainer implemented.
3. ✅ Train robust teacher and compressed candidates
Trained width-4 teacher, recovered width-3, then width-2. Rank-1 position experiment failed and was discarded.
4. ✅ Validate exact exported submission comprehensively
Width-2 exact export: 499,998/500,000 random autoregressive, 3,844/3,844 Cartesian edge, 400/400 CPU API; attention/head ablation changed 40/40; attention maps input-dependent (max delta 0.543).
5. ✅ Finalize smallest qualifying submission
Final /workspace/submission.py embeds trained source-level tensor weights; 2,510 registered parameters; only torch imports; no training logic.
