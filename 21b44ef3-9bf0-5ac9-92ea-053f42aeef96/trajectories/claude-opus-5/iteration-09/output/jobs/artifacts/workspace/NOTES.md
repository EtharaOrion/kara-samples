# Minimal addition transformer — how the shipped model was obtained

## Files

| file | role |
|---|---|
| `submission.py` | **the graded artefact**: model class + trained weights + `build_model`/`add`. Imports `torch` only. |
| `model_src.py` | the model class. Its text is inlined verbatim into `submission.py` by `build.py`. |
| `data.py` | operand sampler. Train/val are split by a hash of the digit string, so held-out pairs are never trained on. |
| `train.py` | the trainer. Trains `E` independent members at once and can anneal a structure to a constant. |
| `xform.py` | exact re-parameterisations (each verified in float64 to leave every answer unchanged). |
| `reinit.py` | *non*-exact moves to the next rung's weight space, which training then has to make work. |
| `build.py` | writes `submission.py` from a checkpoint. |
| `verify.py` | audit: accuracy, all carry patterns, edge cases, attention ablations, numeric agreement. |

Nothing in `submission.py` computes an answer outside the model: `add` builds one
padded digit-pair sequence, runs one forward pass, and reads all answer digits
off the argmax.

## The model

One transformer block over per-place digit-pair tokens.  The sequence is
LSB-first with a `(0,0)` pad at each end; token *i* embeds place *i* as
`code[a_i] + code[b_i]` from a single learned 10-entry code table, and position
*i* predicts answer digit *i-1*.  The read-out is the negative squared distance
to that same code table, so the table is tied between embedding and output.

The block is: ReLU projection -> self-attention -> read-out.  There are two
heads over **one** set of keys and values; the only difference between them is
the mask (strictly causal vs. inclusively causal).  Attention scores are
`key_j - |lam| * (i - j)`, so a head selects the *nearest earlier* place whose
key is high.

## What the trained network actually does

Reading the weights out (all numbers learned, none set by hand):

* The code table converges to an arithmetic ramp, so `x_i = code[a_i]+code[b_i]`
  is a monotone function of the place sum `s_i = a_i + b_i` alone.
* With the bounded activation the two bank units are step-like, and the key is
  `0` for `s <= 8`, `-48` at `s == 9` and `+1.11` for `s >= 10`.  The read-out of
  the shipped model:

  ```
  code   0  1.00  1.94  2.82  3.73  4.66  5.57  6.46  7.34  8.33
  knees  7.52 (turns on between s=8 and s=9)   8.28 (between s=9 and s=10)
  k(s)   0 0 0 0 0 0 0 0 0  -47.83  1.107 1.107 ... 1.107
  ```

  notch depth 48.9 against the 8|lam| = 32 it has to beat, plateau gap 1.11
  against the |lam| = 4 it has to stay under, and `w_o * gap = 1.11` against a
  code step of 1.00 -- one carry, in code units.
* Before that, with a ReLU bank, `k(s)` learned the same **plateau with a notch**: flat at one level for
  `s <= 8`, a deep spike downwards at exactly `s == 9`, and flat at a second
  level for `s >= 10`.  The notch depth exceeds `8*|lam|`, so a place with
  `s == 9` can never win the attention even when it is eight steps nearer; the
  gap between the two plateaus is smaller than `|lam|`, so among all other
  places the nearest one always wins.
* `s == 9` is exactly the *carry-transparent* case.  So each head attends to the
  nearest earlier non-transparent place, and the value it reads back says
  whether that place generated a carry (`s >= 10`) or absorbed one (`s <= 8`).
* The strictly-causal head therefore returns the carry **into** place *i*, and
  the inclusively-causal head the carry **out of** it.  With
  `z = x + w_o*o_A + w_o2*o_B` and the learned ratio `w_o2/w_o ~ -10`, this is
  `digit = s + c_in - 10*c_out`.

This is why the attention ablations in `verify.py` are so destructive: freezing
the attention at its batch mean, or forcing it to attend to the previous
position, drops exact-match from 1.000 to under 0.01.  The pattern it computes
is genuinely a function of the operands — 4096 random inputs produce over 100
distinct attention argmax patterns.

## The ladder, as run

| params | how |
|---|---|
| 55 -> 29 | shared key/value, dead units dropped, fold bank annealed away, one unit rebalanced |
| 29 -> 25 | code translated to intercept 0 and the constant bank unit removed (`reinit.py`) |
| 25 -> 22 | two units that had converged on the same knee merged |
| 22 -> 21 | `code[1]` pinned at 1 using the code-scale symmetry (exact) |
| 21 -> 18 | bank input weights pinned to their signs (exact) |
| 18 -> 17 | read-out temperature fixed at 1 (argmax-invariant) |
| 17 -> 16 | bank moved to `clamp(.,0,1)`, which needs two units instead of three |
| 16 -> 15 | `lam` annealed onto -4 |
| 15 -> 14 | `w_o` absorbed into the shared key/value and pinned at 1 |
| 14 -> 13 | read-out temperature fixed again |

Every step is followed by training, and the graded file was rebuilt and
re-verified at 29, 22, 17, 14 and 13 parameters so that there was always a
working submission in place.

## How it was made small

Cold training below roughly 100 parameters never finds the carry mechanism, so
small models are reached by a ladder: train a larger model, then repeatedly
(a) anneal some structure onto a constant during further training, and
(b) apply an *exact* transform that removes the now-redundant parameters.

Every rung is checked by `xform.py`, which rebuilds both models in float64 and
asserts that the argmax agrees on 200k samples at three different place counts.
Two lessons that cost several runs:

* **Anneal with a per-element cap.**  Capping the residual by the *maximum* over
  all entries means a typical entry only feels the constraint in the last few
  percent of the schedule and has no time to reorganise.  Per-element caps make
  every entry anneal at the same relative rate; this alone turned three failed
  runs into three successful ones.
* **Anneal the cause, not the symptom.**  Clamping a single code entry to zero
  breaks the arithmetic ramp, because `code[a]+code[b]` stops depending only on
  `a+b`.  Removing the constant bank unit instead makes the absorb level zero,
  and the loss then translates the whole code down on its own.

Learning rate has to fall as the model shrinks: 0.006 trains the 54-parameter
model fine and destroys the 29-parameter one within 500 steps.

Two structural cuts could not be annealed at all and had to be made by hand in
`reinit.py`, with training afterwards:

* **Removing the constant bank unit.**  Its offset is what lets the code sit at
  a non-zero intercept, so the unit and the intercept have to go together;
  annealing either alone just moves the offset into the other.  Translating the
  code to intercept zero and absorbing the shift into the bank biases leaves the
  keys, values and attention untouched, and lands close enough that the model is
  still exact before any retraining.
* **Merging two units that converged on the same knee.**  A unit adds a ramp of
  slope `key_w[u]*W1[u]` above `-b1[u]/W1[u]`; two units with the same knee are
  one ramp written twice, so one of them can carry the summed slope.

These checkpoints are always retrained before anything is built from them, so
every shipped weight is the output of an optimiser step.

A harness bug worth recording: `load_state_dict` copies into the *parameter's*
dtype, so a float64 checkpoint loaded into a float32 module was silently rounded
and the exactness checks were really running at float32.  These models are stiff
enough (keys in the hundreds) that a float32 rounding moves a logit by ~1e-2,
which is enough to flip a rare answer.  `xform.py` now promotes the module
before loading and works in float64 throughout.

## What sets the floor

With a ReLU bank, the key has to be flat, then step down at `s == 9`, then step
back up and stay flat — three changes of slope, so three units.  Two is provably
impossible: with only one knee below `s == 9` and one above, the notch depth is
necessarily smaller than the plateau gap, and the ordering the attention needs
is inverted.  That fixes a floor of 16: eight free code entries, three biases,
three key weights and the two attention write scales.

A **bounded** activation removes that argument.  `clamp(z, 0, 1)` saturates, so
two units suffice: one that turns on between `s == 8` and `s == 9` carrying the
notch, and one that turns on between `s == 9` and `s == 10` carrying the model
back up to the second plateau.  The key is then piecewise constant by
construction rather than by slopes that have to cancel, and the floor drops to
14.  Morphing a trained ReLU bank onto the bounded one
(`relu(z) - t*relu(z-1)`, `t: 0 -> 1`) does *not* work — the ReLU solution
relies on its slopes summing to zero far above the knees, and capping the units
destroys exactly that, with the key at `s == 10` jumping from +2 to +57.  What
does work is keeping the trained code table, drawing the two-unit bank at random
over a wide range and training: with a thousand members, one finds it.

Beyond that, the write scale can go too.  The two heads share one set of values,
so scaling every key by `c` scales both attention outputs by `c`; setting
`w_o = 1` and folding its old value into `key_w` (and out of `w_o2`) keeps the
products `w_o*o` unchanged and only sharpens the softmax, which for `c > 1` is
the safe direction.  Annealing `w_o` down instead fails, because what has to
stay constant is the *product* of `w_o` with the plateau gap, and the cap drives
one factor down while the other is still where training left it: accuracy went
to 0.46 on two different schedules, against 0.85 for the direct move.

That leaves **13**, and every one of them is load-bearing:

| what | how many | why it cannot go |
|---|---|---|
| free code entries | 8 | ten digits, two of them held at 0 and 1; pinning a third would be asserting what a digit is worth |
| bank biases | 2 | the two thresholds that say where carry-transparency starts and ends |
| key weights | 2 | the depth of the notch and the height of the second plateau |
| `w_o2` | 1 | the fold, which is where the base lives — it has to be learned |

`q`, `lam`, `w_o` and `ls` are architectural constants (1, -4, 1, 1).  Only one
of `w_o`/`w_o2` can be pinned: fixing both would fix their ratio, and that ratio
is -10.

The attention slope `lam` is not a gauge freedom — rescaling the score changes
the softmax temperature — so it has to be annealed onto a constant.  The window
is narrower than the argmax analysis suggests, because the softmax leaks: what
matters is not `gap < |lam|` but `|lam| - gap` being several units.  Sweeping
the trained model, `lam` in `[-7, -5]` keeps it exact, `-4` drops it to 0.25
(too leaky) and `-16` to 0.92 (the notch is no longer deep enough).

Exactly two exact symmetries remain in the final architecture: rescaling the
code (absorbed by the bank biases and key weights) and the read-out temperature,
which cannot change an argmax at all.  These pay for pinning `code[1] = 1` and
`ls = 1`.  `code[0] = 0` and the position slope are architectural constants that
training is annealed onto rather than gauge freedoms.
