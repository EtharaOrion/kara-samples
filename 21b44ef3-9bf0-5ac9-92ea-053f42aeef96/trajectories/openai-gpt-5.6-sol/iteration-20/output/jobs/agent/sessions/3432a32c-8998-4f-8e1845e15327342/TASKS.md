# Task List

1. ✅ Inspect starting workspace and baseline files
Workspace is empty. Found usable CUDA PyTorch 2.6 via env -u PYTHONPATH /usr/local/bin/python3; H100 available.
2. ✅ Create rank-3 inference submission at graded path and matching trainer
Created and syntax-checked 2,635-parameter rank-3 submission scaffold plus trainer; placeholder add completes via model decoding.
3. 🔄 Train rank-3 transformer from scratch with broad curriculum
Run 40k-step H100 training with uniform and eight-family structured curriculum.
4. ⏳ Export trained parameters into inference-only submission.py
Use permitted source-level tensor literals and exact architecture.
5. ⏳ Validate exact exported submission, edge accuracy, and model dependence
Check parameter count, random/structured accuracy, attention ablation, output corruption, and file screening surface.
