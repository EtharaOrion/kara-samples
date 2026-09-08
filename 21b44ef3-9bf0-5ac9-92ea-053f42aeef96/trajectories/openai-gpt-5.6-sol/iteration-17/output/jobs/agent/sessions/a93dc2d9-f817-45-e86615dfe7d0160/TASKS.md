# Task List

1. ✅ Inspect fresh workspace baseline and available compute
Workspace was empty; H100 80GB is available. No baseline files survived the reset.
2. ✅ Create a working inference-only submission at /workspace/submission.py
Created valid interface immediately with a 4-pass shared-block joint-output transformer placeholder; trained state still needed.
3. 🔄 Implement and train a compact substantively different transformer
Training a 2,779-parameter width-20 joint-output transformer with one shared attention/FFN block iterated four times and learned per-iteration state.
4. ⏳ Compress model while preserving at least 99% exact accuracy
Test reductions only after robust checkpoint.
5. ⏳ Export trained weights and verify exact final submission
Test random, broad edge cases, parameter count, attention dependence, and output dependence.
