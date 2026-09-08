# Task List

1. ✅ Inspect baseline files and environment
Workspace was empty. PyTorch 2.6/H100 used through Python 3.11 with PYTHONPATH override.
2. ✅ Create complete inference submission scaffold
Created valid /workspace/submission.py with complete API.
3. ✅ Train robust teacher and compressed width-2 model
Width-4 teacher trained, then pruned/recovered through width 3 to width 2. Final checkpoint reached 499,997/500,000 uniform and 499,994/500,000 structured pre-export.
4. ✅ Explore one safe parameter reduction
Gauge-fixed rank-2 positional factorization by fixing rows 17 and 22 to identity, preserving positional products within 1.2e-7 and reducing 2,442 to 2,438 parameters.
5. ✅ Export trained weights and validate exact graded file
Source tensor literals embedded. Two 1M trials scored 99.9987–99.9988% uniform and 99.9993–99.9995% structured; 58,564/58,564 edges; 100/100 attention/output ablations changed answers; attention map delta 0.946; imports limited to torch.
