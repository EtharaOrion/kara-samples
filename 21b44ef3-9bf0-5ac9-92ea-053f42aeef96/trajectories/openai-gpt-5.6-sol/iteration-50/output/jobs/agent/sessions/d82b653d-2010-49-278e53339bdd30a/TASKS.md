# Task List

1. ✅ Inspect fresh workspace baseline and environment
Workspace was empty. H100 available; PyTorch 2.6 used via /usr/local/bin/python3 with PYTHONPATH unset.
2. ✅ Create complete inference submission and staged trainer
Created /workspace/train.py and complete early /workspace/submission.py; syntax/import/API verified.
3. ✅ Train width-4 teacher and staged width-3→2 models
Completed all 108k stages; final width-2 model scored 499,999/500,000 on both uniform and structured validation.
4. ✅ Apply exact gauges and export smallest trained model
Exported a 2,254-parameter model using LayerNorm folds, positional/K/V rank gauges, residual/null gauges, reference logits, and softmax invariance.
5. ✅ Validate exact exported submission
Fresh import passed. Two 500k random evaluations scored 500,000 and 499,999; structured evaluations scored 500,000/500,000. Direct edge/API 300/300. Attention and classifier ablations changed 20/20; attention score delta 9.33. Only torch imports.
