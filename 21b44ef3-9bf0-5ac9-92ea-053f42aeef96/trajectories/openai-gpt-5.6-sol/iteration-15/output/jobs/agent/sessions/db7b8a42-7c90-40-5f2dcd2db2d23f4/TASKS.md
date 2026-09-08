# Task List

1. ✅ Inspect baseline files and environment
Workspace was empty; installed compatible PyTorch 2.6 CUDA 12.4 and confirmed H100.
2. ✅ Implement width-20 untied compact decoder and trainer
Created inference-only graded file early and separate trainer; architecture has 4,848 parameters.
3. ✅ Train with broad random and edge curriculum
Completed 30k steps; final trainer validation was 100% on 200k uniform and 200k structured autoregressive pairs.
4. ✅ Validate exported submission and model dependence
Exact export: 100% on fresh 100k uniform and 100k structured; hard cases pass; attention/output perturbations changed 30/30.
5. ✅ Attempt rank-7 positional compression
4,663-param model reached 99.696% random / 99.908% structured but had tiny margins, so rejected in favor of robust 4,848-param export.
