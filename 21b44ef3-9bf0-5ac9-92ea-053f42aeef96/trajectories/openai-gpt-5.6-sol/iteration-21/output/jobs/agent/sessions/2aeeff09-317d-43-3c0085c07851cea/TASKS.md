# Task List

1. ✅ Inspect fresh workspace baseline and environment
Workspace is empty. H100 available; working interpreter is PYTHONPATH=/usr/local/lib/python3.11/dist-packages /usr/bin/python3.11 with torch 2.6 CUDA.
2. ✅ Create complete rank-3 inference submission and trainer
Created /workspace/submission.py and train.py; architecture executes and independently counts to 2,635 parameters.
3. ✅ Train rank-3 model from scratch with broad curriculum
Completed all 42,000 steps. Checkpoints repeatedly scored 100% on random, structured, and corner validation with large minimum margins.
4. ✅ Embed trained weights into submission.py
Embedded 2,635 trained float values from robust step-32k checkpoint as source-level literals; submission has only torch dependency and inference code.
5. ✅ Validate exact exported submission comprehensively
Exact exported state matches checkpoint; 500k random, 500k structured, 6,561 corners, and 3,324 CPU API cases all exact. Both output corruption and attention ablation changed 64/64 answers; attention input delta 0.754.
