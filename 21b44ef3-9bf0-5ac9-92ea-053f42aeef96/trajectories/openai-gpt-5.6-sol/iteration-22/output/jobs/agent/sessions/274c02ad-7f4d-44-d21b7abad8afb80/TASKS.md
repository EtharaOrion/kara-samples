# Task List

1. ✅ Inspect fresh workspace and baseline files
Workspace is empty; PyTorch is available through /usr/bin/python3.11 with H100 GPU.
2. ✅ Implement complete rank-2 grouped-query inference submission
Complete API exists at /workspace/submission.py; architecture has exactly 2,590 registered parameters.
3. ✅ Build rank-2 trainer and train from scratch
Completed 44,000 steps. Final local accuracy: 499,947/500,000 random and 499,959/500,000 structured.
4. ✅ Validate and robustly fine-tune candidate
General and targeted recovery completed. Final checkpoint reached 99.987–99.998% sampled random/structured checkpoints and 217,537/217,564 (99.9876%) difficult curated edge Cartesian cases.
5. ✅ Export learned weights into submission.py
Embedded all 18 final trained tensors (2,590 values) as source-level torch.tensor literals; file compiles.
6. ✅ Verify exact graded file and perturbation sensitivity
Exact export: 2,590 params, 3005/3006 CPU API cases, attention ablation 40/40 changed, head corruption 40/40 changed, input attention-map max delta 0.624.
