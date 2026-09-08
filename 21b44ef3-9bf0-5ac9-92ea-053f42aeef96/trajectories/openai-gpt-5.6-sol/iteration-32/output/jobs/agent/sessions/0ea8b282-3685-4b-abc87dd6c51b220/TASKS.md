# Task List

1. ✅ Inspect fresh workspace baseline and runtime
Workspace empty; /usr/local/bin/python3 has torch 2.6.0+cu124 and H100.
2. ✅ Create early complete inference-only submission
Complete attempt-31-style inference path exists at /workspace/submission.py; placeholder weights will be replaced atomically after training.
3. ✅ Implement training and validation pipeline
Trainer compiles; parameter counts match historical architecture and atomic checkpoints/validation are implemented.
4. 🔄 Train robust 2,438-parameter fallback
Train width-4 teacher, compress/recover widths 3 and 2, fold LayerNorm affine terms, gauge-fix positions.
5. ⏳ Explore exact gauge-based compression
Canonicalize no-op all-ones shifts and projection basis freedoms exactly; retain fallback if numerical checks differ.
6. ⏳ Export and thoroughly verify final submission
Fresh import, parameter count, random/structured/edge accuracy, attention/input/output dependence, syntax and dependency checks.
