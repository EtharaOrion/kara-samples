# Three-operand subtraction, 10 digits: the solution guide

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

- Inputs are three integers, each exactly 10 digits, drawn from [1,000,000,000, 9,999,999,999], and the grader always passes b + c <= a. The output is the exact difference a - b - c as an integer; it is never negative.
- The submission must reach at least ninety-nine percent exact-match accuracy on 10,000 hidden full-width cases drawn from a frozen seed, and one hundred percent on a fixed set of adversarial edge cases, the all-nines and maximum-cascade patterns. Satisfies checkers `accuracy` and `edge_accuracy`.
- The reward is continuous: below the threshold it tracks accuracy, above it the score grows as the parameter count shrinks, reaching one only at the parameter floor with qualification and perfect edges. Satisfies checker `reward_schema`; deliverable `D1`; red-line `RL1`.

Parameter counting.

- Unique parameters after weight tying and deduplication, recounted independently from the tensors themselves; metadata claims are ignored. Satisfies checker `parameter_recount`.
- Fixed sinusoidal positional encodings and registered buffers are not counted, following the original Transformer paper convention; learned positional encodings are.

## How to solve it

The private reference this bundle carries is a nineteen-parameter transformer running the same u-automaton over sigma with the borrow chain never leaving the admitted domain, and the road to it reads as follows.

1. Read `instruction.md` literally: the grader always passes triples with the two subtrahends summing to at most the minuend, all full width, so the result is never negative and never wider than the minuend. Satisfies checker `reward_schema`; deliverable `D1`.

2. The difficulty mirrors three-operand addition: column borrows reach two, so a binary borrow lookahead cannot read the chain directly.

3. Transport instead of reinventing: pool each column as sigma equals nine plus both subtrahend digits minus the minuend digit, a fixed signed projection with the constant nine riding in its own token slot. Sigma lives in zero to twenty-seven, and the borrow recurrence on sigma is exactly the three-operand carry recurrence, so the same u reduction applies, u equals sigma modulo ten plus the previous column's tens level, and the binary automaton does the rest.

4. The answer digit is nine minus u minus the chain bit, modulo ten, which is one reflection of the readout basis and costs nothing; the wrap handles every borrowed ten with no sign machinery.

5. The domain guarantee does the top of the number for free: columns above the operands pool to the propagate value, and since the subtrahends never exceed the minuend the chain dies at the top operand column, so the high digits read out zero on their own.

6. Keep the model under twenty registered parameters the same way the addition family does, fixed constants and tied scales, honest under the independent recount and the forward-pass observation. Satisfies checkers `parameter_recount`, `model_use_rate`, and `static_policy`.

7. Verify on held-out admitted triples and force the cancellation and double-borrow cascades, minuend exactly equal to the subtrahend sum, near-total cancellation to one, the three-zeros cascade, before submitting; edges gate at one hundred percent. Satisfies checkers `accuracy` and `edge_accuracy`.

## How to run and verify the solution

Run `python train.py`; it draws admitted triples the way the grader draws them, redrawing until the domain holds, trains, and embeds the weights into `submission.py`. Iterate until `evaluate()` holds at the threshold with the cancellation edges clean, then submit; the verifier grades the hidden set and writes `/logs/verifier/score.json`, and reward reaches one only at qualification with perfect edges and parameters at the floor. Satisfies red-line `RL1`.

The hidden evaluation grades all 10,000 random cases plus every fixed edge case over the rotated frozen seed, and the reference above scores them all exactly.

## Routes that fail

- Direct arithmetic disguised as a model: measured rejected by the `shortcut fixture rejected by static policy` control.
- Submitting the unchanged template: measured rejected by the `empty template scores exactly zero` control.
- A structurally valid but untrained model: measured non-qualified by the `constant-output model earns no qualification` control.
