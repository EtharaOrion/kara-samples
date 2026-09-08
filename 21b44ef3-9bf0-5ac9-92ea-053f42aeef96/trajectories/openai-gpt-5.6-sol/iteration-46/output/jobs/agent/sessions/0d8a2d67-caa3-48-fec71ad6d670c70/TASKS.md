# Task List

1. ✅ Inspect workspace and create valid inference scaffold at graded path
Created valid inference scaffold.
2. ✅ Train strong width-4 grouped-query teacher and verify convergence
Grouped-query aborted in documented weak basin. Independent fallback trained from scratch to perfect large-sample accuracy.
3. ✅ Prune and recover shared FFN from width 4 to width 3 then width 2
Not pursued after grouped-query weak basin; retained robust independent transformer.
4. ✅ Apply safe exact compression and embed trained tensor literals
Actual trained weights embedded; exact affine folds and reference-logit gauge yield 4,517 parameters.
5. ✅ Validate fresh import, accuracy, edges, parameter count, and model dependence
Fresh submission: 500k/500k random, 500k/500k long-carry structured, 441/441 edge grid; attention/head ablation changed 30/30; attention score delta 7.07.
