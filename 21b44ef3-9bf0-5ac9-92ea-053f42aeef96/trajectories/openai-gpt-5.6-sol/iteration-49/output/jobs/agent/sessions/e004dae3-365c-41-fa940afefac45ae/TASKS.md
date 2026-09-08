# Task List

1. ✅ Inspect baseline files and environment
Workspace was empty; H100 available.
2. ✅ Create and train a valid grouped-query fallback submission
Trained staged width-4→3→2 model from scratch; 500k/500k uniform and structured; exported 2,510 parameters.
3. ✅ Apply exact parameter reductions beyond the fallback
Applied exact affine folds, LayerNorm null gauges, positional affine gauge, K/V basis gauges, residual gauges, and reference logits; final count 2,254 (2 below prior best). Rejected approximate scalar pruning.
4. ✅ Verify final exported submission comprehensively
Fresh exported model: 500k/500k uniform, 500k/500k structured, 1361/1361 direct CPU random+edge, margins >3.6; attention and head ablations changed 40/40.
