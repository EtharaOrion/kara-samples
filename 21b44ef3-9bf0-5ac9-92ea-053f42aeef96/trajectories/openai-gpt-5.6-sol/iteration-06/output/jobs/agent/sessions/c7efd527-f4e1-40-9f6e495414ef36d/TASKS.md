# Task List

1. ✅ Inspect fresh workspace baseline and training environment
Workspace was empty; found PyTorch 2.6 CUDA installation usable via Python 3.11 with isolated PYTHONPATH and an H100 GPU.
2. ✅ Create an early valid end-to-end submission at /workspace/submission.py
Created and import-tested an inference-only source-literal submission before training.
3. ✅ Train compact recurrent transformer candidates
Fixed sinusoidal positions failed. Learned-position FF widths 20, 16, 14, and 13 were trained; FF-13 plus low-LR fine-tuning reached 99.972% autoregressive random and 99.995% structured accuracy at 3,313 parameters.
4. ✅ Validate accuracy, edge cases, model dependence, and attention dependence
Exact exported CPU submission passed 11 explicit edges and 1000/1000 seeded random; attention ablation and output corruption each changed 100/100 answers; attention input delta was 0.5127.
5. ✅ Finalize smallest qualifying source-literal submission
Installed FF-13 source-literal model at /workspace/submission.py, removed unused import, syntax checked, and retained training/checkpoint/log evidence in /workspace.
