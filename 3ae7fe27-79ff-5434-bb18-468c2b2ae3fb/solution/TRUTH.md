# Three-operand addition, 8 digits: the solution guide

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

- Inputs are three integers, each exactly 8 digits, drawn from [10,000,000, 99,999,999]. The output is their exact sum a + b + c as an integer.
- The submission must reach at least ninety-nine percent exact-match accuracy on 10,000 hidden full-width cases drawn from a frozen seed, and one hundred percent on a fixed set of adversarial edge cases, the all-nines and maximum-cascade patterns. Satisfies checkers `accuracy` and `edge_accuracy`.
- The reward is continuous: below the threshold it tracks accuracy, above it the score grows as the parameter count shrinks, reaching one only at the parameter floor with qualification and perfect edges. Satisfies checker `reward_schema`; deliverable `D1`; red-line `RL1`.

Parameter counting.

- Unique parameters after weight tying and deduplication, recounted independently from the tensors themselves; metadata claims are ignored. Satisfies checker `parameter_recount`.
- Fixed sinusoidal positional encodings and registered buffers are not counted, following the original Transformer paper convention; learned positional encodings are.

## How to solve it

The private reference this bundle carries is a nineteen-parameter transformer computing u = (s mod 10) plus the previous column's generate level inside the model and running the proven two-operand lookahead over it, and the road to it reads as follows.

1. Read `instruction.md` literally: three full-width addends, exact sum, smallest model. Satisfies checker `reward_schema`; deliverable `D1`.

2. The new difficulty is the carry alphabet: three digits per column sum to as much as twenty-seven, so carries take the values zero, one, and two, and a binary carry lookahead cannot read that directly. Naive scaling of a two-operand solution fails exactly here.

3. The move that cracks it is a change of variable. Write s for the column sum, g for its tens level, floor of s over ten, and m for s modulo ten. Define u as m plus the previous column's g. Then u never exceeds eleven, the answer digit is u plus the incoming binary chain bit modulo ten, and that chain bit follows the ordinary two-operand rule, generate at u at least ten, propagate at u exactly nine. The whole three-operand problem collapses onto the binary automaton.

4. Give each row its own column's three digits and the previous column's three digits; that is presentation, not computation, and it lets a per-position feed-forward stage compute u inside the model, integer-exact step functions at ten and twenty from ReLU ramp pairs with a fixed integer combine.

5. On top of u, run the two-operand machinery unchanged: the same key notch hiding propagate columns, the same generate-bit value, the same circular cosine readout on u plus the chain bit.

6. Train or derive the thresholds; the structure is small enough that the whole model stays under twenty registered parameters while the verifier recounts them independently and watches the attention do real work. Satisfies checkers `parameter_recount`, `model_use_rate`, and `static_policy`.

7. Verify on held-out full-width triples and force the three-operand signatures, all-nines triples driving sustained carry-two chains and the alternating double-load columns, before submitting; edges gate at one hundred percent. Satisfies checkers `accuracy` and `edge_accuracy`.

## How to run and verify the solution

Run `python train.py`; it draws full-width triples the way the grader draws them, trains, and embeds the weights into `submission.py`. Iterate until `evaluate()` holds at the threshold with the carry-two edge patterns clean, then submit; the verifier grades the hidden set and writes `/logs/verifier/score.json`, and reward reaches one only at qualification with perfect edges and parameters at the floor. Satisfies red-line `RL1`.

The hidden evaluation grades all 10,000 random cases plus every fixed edge case over the rotated frozen seed, and the reference above scores them all exactly.

## Routes that fail

- Direct arithmetic disguised as a model: measured rejected by the `shortcut fixture rejected by static policy` control.
- Submitting the unchanged template: measured rejected by the `empty template scores exactly zero` control.
- A structurally valid but untrained model: measured non-qualified by the `constant-output model earns no qualification` control.
