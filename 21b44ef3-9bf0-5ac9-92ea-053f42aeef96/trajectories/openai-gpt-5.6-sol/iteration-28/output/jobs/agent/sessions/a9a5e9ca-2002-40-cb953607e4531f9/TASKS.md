# Task List

1. ✅ Inspect baseline files and available compute/runtime
Workspace was empty; system Python 3.11 with PyTorch and H100 used.
2. ✅ Create complete inference submission and trainer
Created inference-only graded submission plus separate trainer; source-level tensor literals used for weights.
3. ✅ Train robust width-2 model and export weights atomically
Width-4 teacher reached 100%; sequential width-3 and width-2 pruning/recovery completed. Width-2 validation 99.999% over 100k mixed examples; 2,510 parameters exported atomically.
4. ✅ Attempt FFN width-1 compression and recovery
Controlled 60k-step recovery peaked at 97.5965%, below threshold; retained width-2 exported fallback.
5. ✅ Validate exact exported submission
Fresh import matched checkpoint exactly; 500k random 100%, 500k mixed 99.9998% on final repeat, CPU API 4089/4089, attention/output ablations 40/40, input attention delta 0.856, 2,510 parameters.
