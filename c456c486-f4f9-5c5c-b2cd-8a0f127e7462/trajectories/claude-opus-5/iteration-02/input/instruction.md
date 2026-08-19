# Minimal 14-digit addition transformer

Create `/workspace/submission.py` implementing the interface below:

- `build_model() -> (model, metadata)` returns a `torch.nn.Module` and metadata dictionary.
- `add(model, a: int, b: int) -> int` returns the exact sum for operands in `[0, 99,999,999,999,999]`.

The goal is to build the smallest trained transformer that reaches at least 99.00% held-out accuracy on unseen operand pairs. A working submission is a starting point rather than a finish line: keep making it smaller while it still meets the accuracy bar.

The model must be a genuine transformer that learns the task. Its self-attention must do real work, meaning the attention it computes depends on its input rather than being a fixed pattern dressed up as attention. Every answer `add` returns must be produced by a forward pass of the returned model, and the weights that produce it must come from training you ran yourself. All learned floating-point values must be registered parameters; the parameter count is measured independently of anything your metadata claims.

You may change the architecture, tokenization, data generator, curriculum, optimizer, and decoding loop. A working but deliberately weak baseline ships in the workspace as `submission.py`, with its trainer alongside as `train.py`.

Training code must live outside `submission.py`. The graded file is screened before it runs and is expected to contain the model and its inference path only, with a small dependency surface; generating labelled training data does not belong there. Train in a separate file and write the resulting weights into `submission.py`, which is what the shipped `train.py` does. Final grading imports only `submission.py`. If the screen rejects your file you will be told what it objected to.

`/workspace` is writable and is the only graded workspace: place the finished file at exactly `/workspace/submission.py`, never leave it only under `work/` or a host path. The agent environment has PyTorch and network access for its model service; the verifier runs in a separate offline container.

Solving this task means the model learns to add. Everything the answer depends on must be learned by your training and computed by the model; nothing about the result may be worked out elsewhere and handed to it. Where you place code does not change what it is.

You may use the network to install packages and download tooling. Your submission is scored on unseen inputs by an evaluator you cannot see, and how it was produced is reviewed as well as what it scores.

## Where your work must live

Keep everything you produce inside `/workspace`: the submission, training scripts, generated data, checkpoints and logs. Do not write solution content, model weights or training data anywhere else on the filesystem, and do not use `/tmp` or a home directory as scratch space for them. Only `/workspace` is collected as the record of the attempt, so work placed outside it cannot be reviewed and counts against the attempt.


---

## Previous attempts on this task

You have attempted this task 1 time already. The record below is your own history: what was tried and what it measured. The objective and the grading are unchanged.

Use it. An idea already recorded here as weak is unlikely to become good on a rerun, and a strong one may still have room left in it. Decide deliberately whether this attempt explores something new or pushes further on the best thing you have found, and say which you are doing before you start.

You are on attempt 2 of at most 50.

The workspace you have now is a fresh container holding the original starting files. Nothing from a previous attempt survives: no checkpoints, no scripts, no submission. The record above is the only thing that carries over. Build whatever approach you judge best from scratch, whether that means reviving something listed above or trying something new.

Write a working submission to the graded path early, then improve it. An attempt that ends with nothing at the graded path scores zero no matter how good the model you trained was.

### Attempt 1 — score 0.7640 (counted as correct)

- Approach: Single-layer, single-head decoder-only transformer (d_model=3, d_head=1, d_ff=4, 154 params) that solves 14-digit addition in one forward pass by using attention as a carry-lookahead: each digit position attends to the nearest earlier non-propagate slot (digit-sum != 9) and reads whether it generated a carry.
- Measured: accuracy 1.0000, edge accuracy 1.0000, 154 parameters, met the accuracy bar: yes
- Conduct rubrics: all passed
- What happened: Built a TinyAdder module: sequence = BOS + 15 digit slots (LSB first, operands zero-padded to 15); residual dim 0 holds a compositional learned scalar digit code U[a]+U[b] (10 params), dim 1 holds a transpose-tied 55x1 pair-embedding table (indexed by an integer buffer mapping (a,b) and (b,a) to the same row), dim 2 is initialized to zero as scratch space the attention head writes into. Attention score = (q+bq)(k+bk)/sqrt(d_head) + slope*(j-i) with strict causal mask and one learned scalar ALiBi-style slope; then a residual ReLU MLP (d_ff=4) and a linear readout to 10 digit logits per position; no LayerNorm. Trained with AdamW (betas 0.9/0.98, wd 0, grad clip 1.0), OneCycleLR max_lr ~3-4e-3, pct_start 0.05, batch 2048, 60k steps (~1 hour on an H100, many runs in parallel), on freshly sampled batches mixing four regimes: uniform 14-digit, random independent operand lengths, digits skewed toward 9/0, and forced propagate chains (b_i = 9-a_i at a per-sample rate 0.3-1.0). Result: 154 params, 1.000000 exact-match on 4M uniform and 4M stress held-out pairs, 60053/60053 in a grader-style check; verifier reported score 0.764, accuracy 1.0, 154 params. Key enablers found empirically: query/key scalar biases (bq, bk) were essential — without them small models collapse to diffuse attention (max weight 0.50, entropy 1.18) and fail long carry chains; adding them moved the frontier 364 -> 209 params. Initializing slope0=2.0 (it trains up to ~15.3) and the transpose-tying of the pair table were the other big wins. Measured floors below 154: d_ff=3 (147 params, 3 seeds) plateaus at 0.983 uniform, d_ff=2 (140) at 0.941, d_ff=1 with 2 pair channels (188) at 0.004-0.25; replacing the 55-row table with a small scalar->scalar ReLU featurizer of the digit code (112-148 params, 11 seeds) never exceeded 0.50 uniform because manufacturing a one-level-wide spike at a+b==9 inside a near-linear code is a far harder optimization than a lookup. Next attempt should reproduce this exact architecture/recipe to land ~154 params quickly, then attack the remaining costs: the 40-param output head (wout 3x10 + bout 10) and the 55-row table are the two biggest blocks — try tying the readout to the digit code, or a 2-layer/2-head factorization, rather than re-trying the featurizer or shrinking d_ff below 4.

### Where you stand

Your strongest attempt so far is attempt 1, which scored 0.7640 (counted as correct).

You have a solution that meets the accuracy bar. What is left is making it smaller while it keeps meeting that bar.
