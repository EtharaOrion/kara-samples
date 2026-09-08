# Task List

1. ✅ Inspect baseline files and environment
Workspace was empty; clean system Python has PyTorch 2.6/CUDA on H100.
2. ✅ Design joint-output transformer and create early valid submission
Created two-layer joint-output transformer predicting all nine digits simultaneously.
3. ✅ Train from scratch with carry and edge curriculum
Width-32 model trained from scratch and then edge-margin fine-tuned.
4. ✅ Reduce parameter count while preserving accuracy
Width-24/FF-16 reduced to 7,760 params but failed qualification; retained reliable 14,464-param model.
5. ✅ Export trained weights into submission.py
Final trained source-level tensor-literal export is at /workspace/submission.py.
6. ✅ Verify exported model accuracy, edge cases, and model dependence
Final exact file: 500k/500k random, 499999/500000 structured, 765/765 curated; attention/output interventions changed 100/100; attention delta 0.966.
