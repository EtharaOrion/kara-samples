# Task List

1. ✅ Inspect baseline files and environment
Workspace was empty; used system Python 3.11 with torch 2.6/CUDA 12.4 and H100.
2. ✅ Implement reproducible recurrent transformer training and export
Implemented trainer/data curriculum/pruning/export and source-literal inference submission.
3. ✅ Train width-20 shared two-pass model with structured curriculum
Width-10 FFN trained 24,000 steps; initial autoregressive validation was perfect on 100k uniform and 100k structured.
4. ✅ Prune and fine-tune FFN below width 10
Recovered widths 9 through 4. Width 4 scored 499,999/500,000 uniform and 499,999/500,000 structured (2,944 params); width 3 was below bar and rejected.
5. ✅ Validate final submission accuracy, parameter count, and model dependence
Final exact file imports on CPU, passes interface/edge checks and 1,000/1,000 random; attention ablation changed 100/100, output corruption changed 100/100, attention input delta 0.929; no forbidden encoded-weight imports.
