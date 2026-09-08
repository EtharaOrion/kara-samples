# Task List

1. ✅ Inspect baseline files and training environment
Workspace was empty; H100 available. Reconstructed architecture from attempt history.
2. ✅ Train and export a working FFN-4 recurrent transformer
Rank-8 model achieved perfect local validation.
3. ✅ Train compressed positional/pass parameterization
Rank-7 factorization recovered to perfect validation, yielding 2,759 parameters (185 fewer than prior best).
4. ✅ Validate exact exported submission and model dependence
Exact CPU submission: 10k/10k random, 1089/1089 edge; ablation/corruption each changed 100/100; attention input delta 0.735; py_compile passed.
