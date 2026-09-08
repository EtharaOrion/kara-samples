# Task List

1. ✅ Inspect workspace and create complete inference-only submission scaffold
Created and syntax-checked /workspace/submission.py.
2. ✅ Train strong width-4 teacher and validate broad exact accuracy
Teacher checkpoint validated on 100k uniform and structured.
3. ✅ Prune and recover width-3 then width-2 FFN
Width-2 reached 99.9986%; edge replay eliminated the discovered asymmetric boundary failure.
4. ✅ Apply exact parameter gauges and export trained weights
Function-preserving compact representation exported with 2,343 registered parameters.
5. ✅ Verify exact exported submission and model dependence
Fresh exported model: 499,995/500,000 random, 499,998/500,000 long-nine, 289/289 edge grid; attention and head ablations changed 40/40.
