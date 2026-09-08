# Task List

1. ✅ Inspect baseline submission, trainer, and workspace constraints
Workspace was empty; found working PyTorch 2.6/CUDA H100 via Python 3.11 with PYTHONPATH unset.
2. 🔄 Train and export a conduct-safe working autoregressive model
Training a rank-9 positional, two-layer causal model from scratch; exporter writes trained source literals.
3. ⏳ Compress positional parameters while preserving architecture integrity
Rank 9 is already conservative compression vs attempt 15; consider rank 8 only after robust rank-9 checkpoint.
4. ⏳ Validate exact exported submission on random, edge, and perturbation checks
Confirm >=99%, robust edges, registered parameter count, model and attention dependence.
