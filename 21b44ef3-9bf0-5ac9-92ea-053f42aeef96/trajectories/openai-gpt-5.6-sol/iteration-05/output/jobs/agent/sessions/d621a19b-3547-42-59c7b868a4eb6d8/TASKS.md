# Task List

1. ✅ Inspect fresh baseline and environment
Workspace is empty. Found usable PyTorch 2.6/CUDA 12.4 under /usr/local/bin/python3 when PYTHONPATH is cleared; NVIDIA H100 available.
2. ✅ Create an early valid recurrent-transformer submission
Created importable inference-only /workspace/submission.py. Architecture has 3,600 parameters.
3. ✅ Train compact weight-tied transformer
Completed 20,000 main steps plus 2,000 low-rate structured edge fine-tuning steps.
4. ✅ Export weights using screen-safe source tensor literals
Final state exported directly into submission.py with torch.tensor source literals; imports limited to torch and torch.nn.
5. ✅ Validate exact final submission
Final exact artifact: 100,000/100,000 random, 17,899/17,899 structured edge, 250/250 public add calls; attention ablation and output corruption each changed 200/200; input-dependent attention delta 0.485; 3,600 parameters.
