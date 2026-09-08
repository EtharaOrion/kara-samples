# Task List

1. ✅ Inspect fresh workspace baseline and environment
Workspace is empty; system Python 3.11 has PyTorch 2.6/CUDA and an H100.
2. ✅ Create complete inference submission and training implementation
Created graded-path inference file and trainer; model has 2,510 parameters before exact gauge compression.
3. ✅ Train teacher and compress to robust width-2 model
Width-4 teacher and pruned width-3/width-2 models each reached 100% on sampled random and structured validation; final checkpoint saved.
4. ✅ Apply safe additional parameter reduction
Function-preserving LayerNorm/position/K/V/output gauges reduced the trained model from 2,510 to 2,352 parameters.
5. ✅ Export and validate exact final submission
Fresh import passed; 500k/500k uniform, 500k/500k structured, 841/841 edge; minimum sampled margin >6.3; attention/output ablations each broke 50/50; 2,352 parameters.
