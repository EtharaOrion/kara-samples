# Task List

1. ✅ Inspect baseline files and environment
Workspace is empty; PyTorch 2.6 is available via /usr/local/bin/python3 and an H100 is present.
2. ✅ Create an early working graded submission
Created importable /workspace/submission.py with the intended 2,741-parameter inference architecture and interface.
3. 🔄 Train robust compact recurrent transformer
Train rank-8/full-pass base using uniform plus broad structured curriculum.
4. ⏳ Compress and recover the trained model
Apply rank-7 positional and rank-1 pass compression, retaining only robust checkpoints.
5. ⏳ Verify exact exported submission
Test random/edge accuracy, parameter count, model dependence, attention dependence, and import.
