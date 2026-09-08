# Task List

1. ✅ Inspect baseline files and environment
Workspace empty; PyTorch 2.6 and H100 available via system Python.
2. ✅ Implement joint-output model trainer and inference export
Inference and source-literal export implemented.
3. ✅ Train qualifying joint-output transformer
Trained width-32/FF-32 joint-output model; final calibration achieved 99.269% on 500k uniform held-out pairs.
4. ✅ Validate random, edge, and model-dependence behavior
Exact exported file: 14,464 parameters; 992/1000 independent random add calls; curated suite 99.449%; attention ablation and head corruption changed 100/100; attention-map delta 0.904.
5. ✅ Attempt safe model compression
Width-28/FF-20 candidate was smaller (10,400) but plateaued at 97.8%, so it was rejected in favor of the qualifying model.
6. ✅ Finalize /workspace/submission.py
Final inference-only source embeds calibrated trained tensors as source literals and compiles/imports successfully offline.
