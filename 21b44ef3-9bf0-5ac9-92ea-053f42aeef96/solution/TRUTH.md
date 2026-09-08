# Two-operand addition, 8 digits: the solution guide

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

- Inputs are two integers, each exactly 8 digits, drawn from [10,000,000, 99,999,999]. The output is their exact sum as an integer.
- The submission must reach at least ninety-nine percent exact-match accuracy on 10,000 hidden full-width cases drawn from a frozen seed, and one hundred percent on a fixed set of adversarial edge cases, the all-nines and maximum-cascade patterns. Satisfies checkers `accuracy` and `edge_accuracy`.
- The reward is continuous: below the threshold it tracks accuracy, above it the score grows as the parameter count shrinks, reaching one only at the parameter floor with qualification and perfect edges. Satisfies checker `reward_schema`; deliverable `D1`; red-line `RL1`.

Parameter counting.

- Unique parameters after weight tying and deduplication, recounted independently from the tensors themselves; metadata claims are ignored. Satisfies checker `parameter_recount`.
- Fixed sinusoidal positional encodings and registered buffers are not counted, following the original Transformer paper convention; learned positional encodings are.

## How to solve it

The private reference this bundle carries is an eleven-parameter trained transformer with a documented interval-bound exactness argument spanning every batch width, and the road to it reads as follows.

1. Read `instruction.md` first and take it literally: the graded outcome is one reward float, the operands are two full-width numbers, and the smallest accurate model wins, so every design choice below serves accuracy first and parameter count second. Satisfies checker `reward_schema`; deliverable `D1`.

2. Feed the operands least-significant digit first. Addition's carry travels from the low column upward, so in this order the model can emit each answer digit from local evidence instead of looking ahead.

3. Pool each column's two digits through one shared embedding so the model sees the column sum rather than separate digits. The whole task then lives in one number per column, and one learned scale over the fixed 0..9 basis is embedding enough.

4. Treat the carry chain as the entire problem. One causal attention head can be the carry lookahead: give it a key that makes propagate columns, the ones summing to exactly nine, invisible, and a value that reads whether the nearest visible lower column generates a carry. The recency slope in a fixed positional bias breaks ties toward the nearest deciding column.

5. Decode each digit on a circular cosine head, so digit equals sum plus carry modulo ten and the nine-to-zero wrap costs nothing. A linear readout has to learn a sawtooth; the circle gets the wrap for free.

6. Train with a curriculum that grows the longest forced propagate run from zero to the full width, plain cross-entropy over the digit vocabulary, no carry labels. The lookahead structure makes the carry learnable; the curriculum makes the long chains reachable.

7. Shrink by moving everything that does not need to be learned into architecture: sinusoidal or fixed positions, fixed temperatures, the fixed digit basis, one shared scale doing double duty as embedding and readout coefficient. The verifier recounts parameters independently from the tensors and watches the forward pass do real work, so the model must stay genuine while it shrinks. Satisfies checkers `parameter_recount`, `model_use_rate`, and `static_policy`.

8. Before submitting, measure exact match on held-out full-width pairs and force the nasty patterns, all nines, the maximum-carry cascades, identical operands. The hidden grader gates edge cases at one hundred percent, so a model that is merely mostly right does not qualify. Satisfies checkers `accuracy` and `edge_accuracy`.

## How to run and verify the solution

Run `python train.py` in the workspace; it generates data the way the grader draws it, trains, and writes the weights back into `submission.py` as a literal. Its `evaluate()` reports held-out exact match with the same predicate the grader uses, so iterate until it holds at the threshold with the edge patterns clean, then submit. The verifier imports the submission, recounts parameters from tensors, observes a forward call on every case, grades the hidden fixed-seed set, and writes `/logs/verifier/score.json`; reward reaches one only at qualification with perfect edges and parameters at the floor. Satisfies red-line `RL1`.

The hidden evaluation grades all 10,000 random cases plus every fixed edge case over the rotated frozen seed, and the reference above scores them all exactly.

## Routes that fail

- Direct arithmetic disguised as a model: measured rejected by the `shortcut fixture rejected by static policy` control.
- Submitting the unchanged template: measured rejected by the `empty template scores exactly zero` control.
- A structurally valid but untrained model: measured non-qualified by the `constant-output model earns no qualification` control.
