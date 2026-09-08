# Minimal 8-digit addition transformer

Create `/workspace/submission.py` implementing the interface below:

- `build_model() -> (model, metadata)` returns a `torch.nn.Module` and metadata dictionary.
- `add(model, a: int, b: int) -> int` returns the exact sum for operands in `[10,000,000, 99,999,999]`. Every graded operand carries the full digit width, never fewer digits.

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

### Attempt 1 — score 0.0000

- Approach: Tiny 2-block causal decoder-only transformer (d_model=5-6, 1 head of d_head=1-3 per block, MLP width 4-14, tied embedding/unembedding, rank-1 or ramp positional code) trained from scratch on synthetic 8-digit addition with a carry-chain-enriched curriculum, then weights base64-inlined into submission.py.
- Measured: accuracy 0.0000, edge accuracy 0.0000, 0 parameters, met the accuracy bar: no
- Conduct rubrics: all passed
- What happened: Built model_src.py (shared architecture), data.py (digit-pair sampler with a hash-bucketed held-out split plus a `chain_digits` generator producing maximal carry chains), train.py (AdamW + OneCycle, batch 2048-4096, lr 1e-2, up to 130k steps, optional attention-entropy penalty), build_submission.py (inlines model_src text + base64 float32 weight blob), and verify.py/audit.py (accuracy + attention-ablation gates). Sequence layout is length 10, LSB-first: pos0 = (0,0) sink, pos1..8 = (a_i,b_i) with a shared embedding summed over the two operands, pos9 = carry-out slot; the head at pos i predicts sum digit i-1, so all 9 digits come from one forward pass. Many configs reached >=99.9% held-out exact match (a 447-param d=6 config hit 100%; a 295-param d=5 config hit 99.97%), and the 295-param one was shipped as /workspace/submission.py. The verifier still returned score None with accuracy 0.0 and model_use_rate 0.0, and flagged: forbidden import `base64`, attention judged to be a fixed reduction, answers unchanged when the model output was corrupted, and no answer returned for every case. The base64 weight-loading path is the concrete cause of at least the import rejection and almost certainly the zero score (the file was screened out before running), so the next attempt must embed weights without base64 — write them as a plain Python nested list/tuple of floats (a few hundred values is trivial) and torch.tensor them in build_model(). Also make `add` robust (no reliance on module-level mutable flags like STORE_ATTN, no bare `import base64`), and note that the shipped model must fail loudly if perturbed: several trained checkpoints showed near-fixed attention (frozen-pattern ablation still scoring 78-98% on carry chains), so pick a checkpoint whose carry head puts ~100% of its argmax mass on generating places and whose accuracy collapses (<20% on maximal chains) when attention is frozen. Retraining recipe that worked: d_model=5, layers "1,1,4;1,1,10" or d_model=6 "1,2,4;1,3,14", q_bias + learnable per-head log_scale on attention logits, rank-1 positional code (scalar per position × one learned direction), lr 1e-2 OneCycle, batch 2048, ~100k steps, batch mix ~40% uniform / 35% carry-transparent-enriched (per-place P(a+b=9)=0.4) / 25% maximal chains, small attention-entropy penalty 0.02 warmed in after 25% of steps; note the chain generator must allow 9+0 transparent places (an earlier clamp(max=8) caused edge-case failures on inputs like 19999999+10000001).

### Where you stand

Your strongest attempt so far is attempt 1, which scored 0.0000.

No attempt has produced a working submission yet. Before optimizing anything, establish that a submission loads, runs, and returns model-computed answers at the graded path.
