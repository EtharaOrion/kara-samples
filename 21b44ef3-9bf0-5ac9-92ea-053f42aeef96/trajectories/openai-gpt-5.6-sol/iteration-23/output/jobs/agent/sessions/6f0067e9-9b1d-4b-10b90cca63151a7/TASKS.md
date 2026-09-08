# Task List

1. ✅ Inspect fresh workspace baseline and environment
Workspace empty. Working runtime is env -u PYTHONPATH /usr/local/bin/python3 with torch 2.6 CUDA on H100.
2. ✅ Create inference submission and width-3 training pipeline
Created /workspace/submission.py and training scripts. Width-3 architecture has 2,550 parameters.
3. ✅ Train width-3 model with robust curriculum
Trained width-4 teacher, pruned least-important FFN neuron, recovered width-3, and completed targeted boundary fine-tuning.
4. ✅ Validate random, structured, and edge accuracy
Final checkpoint exact on 500,000 random, 500,000 structured, and 83,521 broad Cartesian boundary pairs under autoregressive decoding.
5. ✅ Embed trained weights into submission.py
Embedded all trained tensors as plain source-level torch.tensor literals. submission.py is standalone and compiles.
6. ✅ Verify exact exported submission and perturbation sensitivity
Exact exported API passed 3,005/3,005 cases; 2,550 parameters; attention ablation and output corruption changed 40/40 outputs; attention input delta 0.488; only torch imports.
