# Task List

1. ✅ Inspect baseline files and environment
Workspace was empty; found PyTorch 2.6/CUDA 12.4 under /usr/local Python and H100 GPU. PYTHONPATH must be unset.
2. ✅ Create an early valid end-to-end submission
Created valid inference model and external trainer; smoke-tested forward and serialization.
3. ✅ Train compact transformer candidates
Selected FF16 model with fixed sinusoidal positions, bias-free projections, and non-affine normalization: 4,900 parameters.
4. ✅ Validate accuracy, edge behavior, and model dependence
Passed 100k random, 50k structured, and 1k scalar exact tests; attention delta 0.420; ablation/output corruption each changed 100/100.
5. ✅ Finalize clean inference-only submission
Cleaned temporary candidates, compiled files, confirmed fresh CPU build and representative exact sums. Final graded file is /workspace/submission.py.
