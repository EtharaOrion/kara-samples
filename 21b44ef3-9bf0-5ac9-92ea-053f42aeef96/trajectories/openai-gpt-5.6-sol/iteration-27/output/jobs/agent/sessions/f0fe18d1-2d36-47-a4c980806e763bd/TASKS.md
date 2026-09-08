# Task List

1. ✅ Inspect fresh workspace and baseline files
Workspace was empty; configured working PyTorch/CUDA runtime.
2. ✅ Create complete placeholder graded submission and training pipeline
Built end-to-end causal grouped-query transformer and external trainer.
3. ✅ Train robust transformer teacher and validate
Rank-3 width-4 model trained from scratch and stabilized; best checkpoints approximately 99.8%+ on 50k random/structured.
4. ✅ Attempt safe compression if validation margin permits
Rank-2 exploration entered a weak positional basin; no further compression attempted because safety margin was inadequate. Retaining 2,635-param rank-3 model.
5. 🔄 Export trained weights and verify exact submission
Run large/boundary validation, emit source tensor literals, check API and model/attention dependence.
