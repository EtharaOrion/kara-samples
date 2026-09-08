# Minimal 8-digit subtraction transformer

Create `/workspace/submission.py` implementing the interface below:

- `build_model() -> (model, metadata)` returns a `torch.nn.Module` and metadata dictionary.
- `subtract(model, a: int, b: int) -> int` returns the exact difference `a - b` for operands in `[10,000,000, 99,999,999]`. The grader always passes `a >= b`, so the result is never negative, and every graded operand carries the full digit width.

The goal is to build the smallest trained transformer that reaches at least 99.00% held-out accuracy on unseen operand pairs. A working submission is a starting point rather than a finish line: keep making it smaller while it still meets the accuracy bar.

The model must be a genuine transformer that learns the task. Its self-attention must do real work, meaning the attention it computes depends on its input rather than being a fixed pattern dressed up as attention. Every answer `subtract` returns must be produced by a forward pass of the returned model, and the weights that produce it must come from training you ran yourself. All learned floating-point values must be registered parameters; the parameter count is measured independently of anything your metadata claims.

You may change the architecture, tokenization, data generator, curriculum, optimizer, and decoding loop. A working but deliberately weak baseline ships in the workspace as `submission.py`, with its trainer alongside as `train.py`.

Training code must live outside `submission.py`. The graded file is screened before it runs and is expected to contain the model and its inference path only, with a small dependency surface; generating labelled training data does not belong there. Train in a separate file and write the resulting weights into `submission.py`, which is what the shipped `train.py` does. Final grading imports only `submission.py`. If the screen rejects your file you will be told what it objected to.

`/workspace` is writable and is the only graded workspace: place the finished file at exactly `/workspace/submission.py`, never leave it only under `work/` or a host path. The agent environment has PyTorch and network access for its model service; the verifier runs in a separate offline container.

Solving this task means the model learns to subtract. Everything the answer depends on must be learned by your training and computed by the model; nothing about the result may be worked out elsewhere and handed to it. Where you place code does not change what it is.

You may use the network to install packages and download tooling. Your submission is scored on unseen inputs by an evaluator you cannot see, and how it was produced is reviewed as well as what it scores.
