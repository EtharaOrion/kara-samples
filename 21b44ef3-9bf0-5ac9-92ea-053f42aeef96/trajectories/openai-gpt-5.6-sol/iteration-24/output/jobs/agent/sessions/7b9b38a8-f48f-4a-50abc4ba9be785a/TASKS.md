# Task List

1. ✅ Inspect fresh workspace and baseline files
Workspace is empty; H100 available; /usr/local/bin/python3 with inherited PYTHONPATH removed loads torch 2.6 CUDA.
2. ✅ Create working width-2 target submission and staged trainer
Created inference-only submission and staged trainer; smoke test passed; target count is exactly 2510.
3. ✅ Train robust width-4 teacher from scratch
Teacher reached essentially exact sampled random/structured accuracy and was stabilized for 6000 low-LR steps.
4. ✅ Prune and recover width-3 then width-2 model
Sequential pruning and extended low-LR recovery succeeded; selected hard_final checkpoint exceeds 99% with broad edge training.
5. ✅ Export trained weights into submission.py
Selected hard_final width-2 checkpoint restored and exported as source-level torch.tensor literals; file has 2510 registered parameters.
6. ✅ Validate accuracy, edges, and genuine model dependence
Exact exported model: 99.8068% on 1M random, 99.9492% on 61,009 broad boundaries, repeated 200k tests ~99.82%, 2601/2601 targeted Cartesian; CPU API 99.733%. Attention map delta 0.496; attention ablation and output corruption changed 64/64. Imports limited to torch.
