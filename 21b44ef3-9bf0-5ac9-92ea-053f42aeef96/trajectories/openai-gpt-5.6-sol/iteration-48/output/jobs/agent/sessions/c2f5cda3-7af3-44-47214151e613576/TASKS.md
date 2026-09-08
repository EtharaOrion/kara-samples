# Task List

1. ✅ Inspect baseline files and runtime/GPU environment
Workspace was empty; used /usr/local/bin/python3 with clean PYTHONPATH and CUDA H100.
2. ✅ Create an early valid inference submission and trainer
Inference-only graded file plus separate trainer/export/compression pipeline created entirely in /workspace.
3. ✅ Train grouped-query width-4 teacher and validate robustness
Seed-2025 teacher converged; staged width-3 and width-2 recovery plus broad edge replay completed.
4. ✅ Prune FFN width 4→3→2 and apply exact gauges
Exact LayerNorm folding, null directions, GL rank gauges, and reference logits produced 2,256 parameters—20 below attempt 47.
5. ✅ Validate exact exported submission and model dependence
Fresh import: 3,005/3,005 CPU API; 500k/500k random GPU; 8,555,614/8,555,625 stress grid. Attention/head ablations changed 40/40; attention score delta 33.99.
