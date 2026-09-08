# Task List

1. ✅ Inspect baseline and environment; design compliant end-to-end model
Workspace was empty; PyTorch 2.6 is available via isolated /usr/local/bin/python3 with H100. Selected causal decoder over 16 interleaved operand tokens plus autoregressive 9-digit result.
2. ✅ Create an early loadable end-to-end submission
Created /workspace/submission.py; build_model, forward, and add all run.
3. ✅ Implement and run transformer training
18,624-parameter model trained to 100% exact match on 102,400 held-out random pairs.
4. ✅ Evaluate held-out accuracy and conduct-sensitive behavior
Final candidate: 99.99976% on 409,600 unseen random pairs and 100% on 200,000 exact-100M carry-chain pairs. Input changes attention weights; disabling attention changed 100/100 answers.
5. ✅ Optimize model size while preserving >=99% accuracy
One-layer width-16 reached only 0.93%; two-layer width-16 plateaued. Width-24 two-layer reaches target with 10,896 params, reducing baseline by 41.5%.
6. ✅ Finalize graded submission and verify offline import
Rebuilt submission from final checkpoint with exact tensor equality. Clean CPU import/API passed 2,004/2,004 including carry edges; 10,896 registered parameters; only torch dependency.
