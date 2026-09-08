# Task List

1. ✅ Inspect baseline files and runtime/GPU environment
Workspace and runtime understood.
2. ✅ Create complete inference submission and trainer in /workspace
Inference and training/export pipeline complete.
3. ✅ Train and validate high-margin width-4 teacher
High-margin perfect teacher obtained.
4. ✅ Prune and recover width-3 then width-2 model
Width-2 recovered with perfect broad sampled validation and strong margins.
5. ✅ Apply exact gauges and export embedded trained parameters
Exact LayerNorm folding yields 2442 registered parameters with 2442 finite embedded trained values.
6. ✅ Verify final import, accuracy, edge cases, and model dependence
Fresh import passed: 500k/500k uniform, 500k/500k carry-heavy, 1256/1256 CPU API; attention and classifier ablations changed 40/40; input attention-score delta 14.535.
