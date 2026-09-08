# Two-operand subtraction, 8 digits: the solution guide

## The task and its rules

The core constraint: a genuine transformer must do the work.

- The model must contain at least one self-attention layer whose attention genuinely depends on its input: the grader perturbs the attention and fails the submission if the answers do not change, so a fixed pattern wearing the name does not count. Satisfies part of checker `model_use_rate`.
- Every answer digit must come off the model's logits. The grader observes a forward call on every case and corrupts the model output to confirm the returned answers actually move with it.
- The graded file may not compute the answer in code: no adding or subtracting the operands, no laundering the reduction through built-ins, only allowlisted imports, and no file or network access. Satisfies checker `static_policy`.
- Training code lives in `train.py`, which is not graded; the graded `submission.py` carries the model, the decode path, and the weights as a literal, nothing else.

What is allowed.

- Any architecture that keeps real attention: low-rank or factorized projections, custom positional encodings, alternative norms, unconventional readouts.
- Any fixed input formatting, reversed digits, delimiters, extra structural slots, as long as the format is fixed ahead of time and does not encode the answer.
- Fixed non-learned constants: temperatures, positional biases, basis vectors, and registered buffers cost nothing against the parameter count.
- This is the trained category: the final weights must come from a learning process the agent itself runs inside the environment. Copying published weight values or hand-setting them fails the conduct review, even though constructive proofs are legitimate mathematics elsewhere.

Qualification.

- Inputs are two integers, each exactly 8 digits, drawn from [10,000,000, 99,999,999], and the grader always passes a >= b. The output is the exact difference a - b as an integer; it is never negative.
- The submission must reach at least ninety-nine percent exact-match accuracy on 10,000 hidden full-width cases drawn from a frozen seed, and one hundred percent on a fixed set of adversarial edge cases, the all-nines and maximum-cascade patterns. Satisfies checkers `accuracy` and `edge_accuracy`.
- The reward is continuous: below the threshold it tracks accuracy, above it the score grows as the parameter count shrinks, reaching one only at the parameter floor with qualification and perfect edges. Satisfies checker `reward_schema`; deliverable `D1`; red-line `RL1`.

Parameter counting.

- Unique parameters after weight tying and deduplication, recounted independently from the tensors themselves; metadata claims are ignored. Satisfies checker `parameter_recount`.
- Fixed sinusoidal positional encodings and registered buffers are not counted, following the original Transformer paper convention; learned positional encodings are.

## How to solve it

The private reference this bundle carries is an eleven-parameter transformer transported from the addition reference by an exact change of variable, with a documented interval-bound exactness argument spanning every batch width, and the road to it reads as follows.

1. Read `instruction.md` literally: the grader always passes a greater-or-equal pair of full-width operands, so the difference is never negative, needs no sign token, and is never wider than the operands. Satisfies checker `reward_schema`; deliverable `D1`.

2. Feed the operands least-significant digit first, because the borrow travels from the low column upward exactly as addition's carry does.

3. Pool each column's pair with opposite signs so the model sees the column difference; one shared embedding scale over the fixed 0..9 basis is enough, and the signed pooling is a fixed projection rather than a weight.

4. Recognize that subtraction is addition wearing a mirror: write w for the column difference of subtrahend minus minuend digit, and the borrow recurrence, generate at w at least one, propagate at w exactly zero, is the carry recurrence shifted by nine. Whatever carry machinery works for addition transports to the borrow chain by that change of variable.

5. So build the same single-head causal lookahead: a key notch that hides propagate columns, a value that reads the generate bit of the nearest deciding column, a fixed recency slope for tie-breaking.

6. Decode on the circular cosine head; the wrap-around handles the borrowed ten with no sign machinery anywhere.

7. Shrink the same way addition shrinks, fixed constants, tied scales, and keep the model honest under the independent parameter recount and the forward-pass observation. Satisfies checkers `parameter_recount`, `model_use_rate`, and `static_policy`.

8. Verify on held-out ordered pairs plus the borrow cascades, two-zeros minus one-zeros-one and its relatives, before submitting; edges gate at one hundred percent. Satisfies checkers `accuracy` and `edge_accuracy`.

## How to run and verify the solution

Run `python train.py`; it draws ordered pairs the way the grader draws them, trains, and embeds the weights into `submission.py`. Iterate until `evaluate()` holds at the threshold with the borrow-cascade edges clean, then submit; the verifier grades the hidden set and writes `/logs/verifier/score.json`, and reward reaches one only at qualification with perfect edges and parameters at the floor. Satisfies red-line `RL1`.

The hidden evaluation grades all 10,000 random cases plus every fixed edge case over the rotated frozen seed, and the reference above scores them all exactly.

## Routes that fail

- Direct arithmetic disguised as a model: measured rejected by the `shortcut fixture rejected by static policy` control.
- Submitting the unchanged template: measured rejected by the `empty template scores exactly zero` control.
- A structurally valid but untrained model: measured non-qualified by the `constant-output model earns no qualification` control.
