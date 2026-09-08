# Task List

1. ✅ Inspect baseline files and runtime environment
Workspace was empty; used system Python 3.11, PyTorch 2.6, and H100.
2. ✅ Create complete inference and training implementation
Created inference-only submission and separate data-generating trainer.
3. ✅ Train teacher and staged width-2 student
Trained width-4 model for 100k+ updates plus broad replay; selected replay-8k checkpoint.
4. ✅ Apply exact gauges and seek reductions below 2,343 parameters
Width-3 recovery remained below bar; retained robust width-4 rank-2 architecture at 2,590 parameters.
5. ✅ Validate final exported submission comprehensively
Fresh import has 2,590 params and scored 9,902/10,000 random CPU API pairs (99.02%); attention ablation and classifier corruption each changed 20/20 outputs.
