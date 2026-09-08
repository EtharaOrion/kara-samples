# Task List

1. ✅ Inspect fresh workspace baseline and environment
Workspace empty; PyTorch 2.6/H100 available via /usr/local/bin/python3 with PYTHONPATH unset.
2. ✅ Create complete inference submission and trainer
Implemented conduct-valid model/trainer; initialization and CUDA smoke tests pass.
3. ✅ Train strong width-4 teacher
36k-step seed eyeball teacher converged strongly and was preserved in teacher.pt.
4. ✅ Prune and recover width-3 then width-2 model
Full staged recovery and 6k hard-family replay complete. Width-2 final checkpoint preserved.
5. ✅ Export compact inference-only trained submission
Exact affine folds and positional/KV/null/reference gauges yield 2,312 registered parameters, 31 fewer than previous best.
6. ✅ Validate final exported submission thoroughly
Exact exported model: 500k/500k uniform, 300k/300k adversarial structured, 39,601/39,601 edge grid, 1k/1k CPU API. Attention/classifier ablation each changes 40/40; input attention delta 0.999998.
