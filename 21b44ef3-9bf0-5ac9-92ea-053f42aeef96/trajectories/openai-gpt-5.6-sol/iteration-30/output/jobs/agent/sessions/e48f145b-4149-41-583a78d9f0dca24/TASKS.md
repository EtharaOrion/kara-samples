# Task List

1. ✅ Inspect workspace and recreate complete 2,510-parameter inference/training architecture
Created valid submission.py and trainer; verified interfaces.
2. ✅ Train and validate width-4 teacher
Teacher exact on 200k uniform and structured with strong margins.
3. ✅ Compress teacher through width-2 and export reliable trained submission
Width-2 exact on 500k uniform and 500k structured.
4. ✅ Apply exact LayerNorm affine folding reduction
Reduced 2,510 to 2,442 parameters with function-preserving affine folding.
5. ✅ Verify exact exported submission, parameter count, accuracy, and model dependence
Fresh exported model exact on 1M random, 1M structured, 38,416 Cartesian edges, and 2,000 CPU API cases; ablations changed 40/40 and attention-map delta was 0.603.
