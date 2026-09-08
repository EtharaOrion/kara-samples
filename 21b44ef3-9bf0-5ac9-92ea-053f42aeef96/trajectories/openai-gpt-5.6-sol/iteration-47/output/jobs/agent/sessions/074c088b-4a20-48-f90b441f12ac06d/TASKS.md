# Task List

1. ✅ Inspect fresh workspace and baseline files
Workspace is empty; H100 80GB is available.
2. ✅ Implement grouped-query model and early inference scaffold
Created inference scaffold and trainer; verified 2,590-param width-4 forward path with system PyTorch.
3. ✅ Train strong width-4 teacher
Exact on 100k random and 100k structured; stabilized sampled margin >11.
4. ✅ Prune and recover width-3 then width-2 model
Final width-2 exact on 500k random and 500k structured; minimum sampled margins 8.63/10.40.
5. ✅ Export trained inference-only submission
Embedded trained tensor literals and validated fresh import.
6. ✅ Apply safe exact parameter compression
Applied known exact gauges to 2,281 then new softmax-invariant key translation gauge to 2,276; perfect on 1M tests.
7. ✅ Run final accuracy, edge, model-use, and parameter checks
Exact final file: 500k random + 500k long-carry + 2521 direct edge/random; attention/output ablations changed 40/40; attention input delta 19.94.
