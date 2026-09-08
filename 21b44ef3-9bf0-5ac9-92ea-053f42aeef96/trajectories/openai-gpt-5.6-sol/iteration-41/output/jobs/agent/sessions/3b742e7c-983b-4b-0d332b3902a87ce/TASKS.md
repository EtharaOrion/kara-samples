# Task List

1. ✅ Inspect fresh workspace baseline and environment
Workspace is empty; H100 80GB is available; PyTorch environment is present.
2. ✅ Create complete inference submission and training pipeline
Created /workspace/submission.py and train.py; compile/model/data shape checks pass under system PyTorch.
3. ✅ Train robust teacher and staged width-3→2 model
Teacher, width-3, and polished width-2 each achieved 100k/100k uniform and structured; final margins 9.09/8.28.
4. ✅ Apply exact compression and export trained tensor literals
Applied exact gauges to reach 2301 parameters; zero argmax disagreements; promoted tensor-literal model to graded path.
5. ✅ Audit exact exported submission
Fresh graded-file import passed; 500k uniform, 500k structured, 64,009 edges, 3,001 CPU calls exact; attention map delta 0.346 and both ablations changed 40/40.
