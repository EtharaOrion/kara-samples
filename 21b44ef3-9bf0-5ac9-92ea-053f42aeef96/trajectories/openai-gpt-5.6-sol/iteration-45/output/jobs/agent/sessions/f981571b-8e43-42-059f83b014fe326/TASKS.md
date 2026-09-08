# Task List

1. ✅ Inspect fresh workspace and baseline files
Workspace is empty; system Python 3.11 has PyTorch and H100.
2. ✅ Create complete inference scaffold at /workspace/submission.py
Complete causal inference path created.
3. ✅ Implement and run staged training
Independent-projection fallback trained for 18k+8k steps and scored 500k/500k uniform and structured.
4. ✅ Export trained weights and validate exact submission
Fresh compressed import: 500k/500k random, 101198/101200 broad boundary, 1000/1000 CPU API; attention/head ablation 30/30; attention delta 0.934.
5. ✅ Apply safe exact compression below prior best if feasible
Applied proven exact FFN/final LayerNorm folding and reference-class logit gauge: 4,640 to 4,517 parameters.
