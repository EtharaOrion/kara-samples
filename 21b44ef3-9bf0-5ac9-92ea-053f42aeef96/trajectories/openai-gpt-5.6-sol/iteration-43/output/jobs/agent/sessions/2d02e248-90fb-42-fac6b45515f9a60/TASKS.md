# Task List

1. ✅ Inspect baseline files, environment, and compute
Workspace is empty; PyTorch 2.6 CUDA works via /usr/local/bin/python3 with PYTHONPATH override; H100 80GB available.
2. ✅ Create early complete inference submission and robust trainer
Created and smoke-tested full end-to-end grouped-query model and trainer. submission.py exists at graded path from the outset.
3. ✅ Train strong teacher and staged width-2 compressed model
After correcting LayerNorm initialization, teacher transitioned and stabilized at 100%; width-3 and width-2 pruning/recovery each reached repeated 20k/20k uniform and structured results with healthy margins. Exported 2,510-param weights.
4. ✅ Apply exact gauge compression and seek reduction below 2,301 parameters
Applied affine folds, positional/token/K/V gauges, reference-logit gauge, and residual all-ones quotient. Final model has 2,281 parameters; preserves all tested argmaxes and scores 500k/500k on uniform and structured validation.
5. ✅ Validate exported submission accuracy, edges, model use, and parameter count
Fresh import reports 2,281 finite trained parameters. Passed 500k uniform, 500k structured GPU cases and 3,000 direct CPU/API random+boundary cases. Attention delta 0.99998; attention and output ablations each changed 40/40. Inference imports only torch modules.
