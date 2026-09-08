# Minimal 8-digit addition transformer

`submission.py` is the graded artefact. Everything else in this directory is the
training and reduction machinery used to produce it; none of it is imported by
the submission, which imports only `torch`.

## What the model is

One transformer block over LSB-first digit-pair tokens.

* Sequence length is `P = n + 2` for `n`-digit operands. Position 0 holds the
  `(0, 0)` place below the ones digit (an attention sink / carry-in slot),
  positions `1..n` hold the real places, and position `P-1` holds the `(0, 0)`
  place above the most significant digit, which receives the carry-out.
* Token `i` is embedded as `code[a_i] + code[b_i]` from a single shared 10-entry
  code table. The same table is reused as the output prototypes: the logit for
  digit `d` at position `i` is `-||y_i - code[d]||^2`. Input embedding and
  unembedding are therefore the same parameters.
* The block is FFN -> strictly-causal single-head self-attention -> FFN over a
  `C`-dimensional residual stream (a Macaron ordering: the pre-attention FFN is
  what builds the attention key).
* Every position predicts its own answer digit, so the whole sum comes out of
  one forward pass. No parameter is indexed by position, so the same weights run
  at any number of places; the 8-digit grading range is one slice of that.

## The mechanism it learns

Write `s_i = a_i + b_i`. A place *generates* a carry if `s_i >= 10`, *absorbs*
if `s_i <= 8`, and is *carry-transparent* if `s_i = 9` — a transparent place
passes through whatever carry arrives from below. So the carry into place `i` is
decided by the nearest place below `i` that is **not** transparent.

That is a nearest-match-below query, which is what attention is for:

1. The pre-attention FFN maps the token to a scalar `z(s)` on one axis. Three
   ReLU knees straddling `s = 9` are enough to give `z` three levels: `0` on
   generates, a small offset `A` on absorbs, and a deep spike `M` on transparent
   places. In the shipped weights the knees sit at `x = 1.035, 0.987, 1.072`
   against `x(8) = 1.155`, `x(9) = 1.049`, `x(10) = 0.944`, so all three units
   are on for `s <= 8`, one is on at `s = 9`, and none for `s >= 10`; the three
   read-out coefficients sum to `3.4e-4` rather than to a free value, which is
   what makes the absorb level flat across all nine absorbing sums.
2. The attention logit from `i` to `j < i` is `b_q * z_j + lam * (i - j)` with
   `lam < 0`. The spike makes transparent places unattendable, so the argmax is
   the nearest non-transparent place below — provided `|b_q * A| < |lam|`, so
   that distance and not the generate/absorb contrast decides among candidates,
   and `|b_q * M| > 2 |lam| L` for the longest carry chain `L` the model must
   cross. The shipped weights give `b_q A = -2.04` against `lam = -2` and
   `b_q M = -17.5`, which covers chains of eight.
3. The value read from that place lies on the same axis (`v = z * w_v`), so the
   residual receives `A * w_v` exactly when the nearest non-transparent place
   below absorbed the carry, and `0` when it generated one.
4. The post-attention FFN adds the carry and folds mod 10.

`diag.py` prints `z(s)`, `lam` and the attention argmax on a maximal carry
chain, so this description can be checked against the shipped weights rather
than taken on trust.

The local shortcut — always attend to `i-1` — reaches about 97% per digit and is
a strong attractor. Two things push training past it: transparent places are
heavily over-represented in the data (`lib.sample_digits`), and `lam` is
initialised small, since a distance penalty larger than the key notch makes the
correct routing unreachable.

## Files

| file | role |
| --- | --- |
| `model_src.py` | the model class; its text is copied verbatim into `submission.py` |
| `lib.py` | config, data sampling, and an E-member-batched mirror of the forward pass |
| `train.py` | trainer |
| `check.py` | asserts `lib.fwd` and the shipped `nn.Module` agree (relative logit diff and arg-max) |
| `xform.py` | exact re-parameterisations, each verified in float64 |
| `reorient.py` | re-expresses bank 2 in bank 1's orientation, as a warm start for a tie run |
| `kvinit.py` | re-points a `kv="share"` checkpoint at `kv="one"`, and reports the notch ratio that architecture would need |
| `diag.py` | inspects a trained member's learned mechanism |
| `build.py` | writes `submission.py`: class text + weights as float literals |
| `verify.py` | audits the shipped file end to end |

## How it was trained

Models this small have enormous seed variance, so `train.py` trains `E = 256`
independent members at once: every member has its own parameters and its own
per-member gradient clipping, and Adam is elementwise, so they are independent
runs that merely share a data pipeline. Members are scored by exact-match on
held-out pairs and only the best few are kept.

Data is generated on the fly from a mix of uniform digits, transparent-rich
digits and maximal carry chains, with the number of places varied from 3 to 14
so that nothing can be keyed to a fixed sequence length. A hash of the operand
pair (`lib.heldout_mask`, 1-in-16 bucket) defines a held-out split; training
batches drop pairs in that bucket and every score reported is measured on it.

## How it was made small

Two kinds of step, alternated, starting from a comfortably over-parameterised
parent:

* **Annealed structural cuts** (`train.py --shrink`). A parameter group is
  frozen and driven linearly to its target — zero for a removed unit or residual
  axis, or the value it is being tied to — over a fixed fraction of the run, then
  held there while training continues. Because the schedule is fixed rather than
  penalised, the cut cannot be traded off against the loss: the model must
  recover the accuracy without it.
* **Exact re-parameterisations** (`xform.py`). These change coordinates rather
  than function. The network's output is unchanged; only the number of numbers
  needed to write it down shrinks. Each op is verified in float64 against the
  original (relative logit difference `< 1e-9` and arg-max agreement `1.0`)
  before it is accepted:
  * `drop` — delete units and residual axes whose weights are already exactly 0.
  * `share` — merge the split key/value read-outs once the value is exactly
    parallel to the key axis.
  * `tie` — bank 2 reuses bank 1's weights, evaluated on the post-attention
    residual (the mod-10 fold and the carry step are the same "did it reach 10"
    step function).
  * `pinrow` — the residual stream has a scale (and at `C = 2` a rotation)
    freedom; use it to set one code row to `e_0`, which then costs nothing.
  * `signin` — `relu(w x + b) = |w| relu(sign(w) x + b/|w|)`, so a unit's input
    magnitude can be absorbed into its readers, leaving only a fixed `+-1`.
  * `opin` — the key axis has a scale freedom; use it to set `o[0] = 1`.
  * `zerobias` — the `C = 1` stream has a scale *and* a shift freedom. A token is
    `code[a] + code[b]`, so the residual picks up twice a code shift while the
    read-out compares against one code entry and picks up it once; the leftover
    is exactly what the training-only read-out bias `by` holds, so the two
    freedoms fix `by` and the pinned code row at the same time and `by` costs
    nothing. A negative scale reverses the stream, which the ReLU banks only
    commute with if their fixed input signs reverse too — the rewrite handles
    that case, so the gauge is available in both orientations.
  * `lamfix` — after `lam` has been annealed onto a constant, it is a constant.
  * `dropscale` — remove the training-only read-out temperature `exp(ls)`. Every
    logit is divided by one strictly positive number, so the arg-max decode, and
    hence every answer, is bit-identical.

After every re-parameterisation the model is retrained in its new coordinates,
so the shipped weights are always the output of a training run and never a
hand-written constant. The ladder that produced the shipped file, each rung a
training run followed by the exact rewrite its result made available:

| params | what came off |
| --- | --- |
| 150 | `C=2, U=U2=6`, split key/value, learned `lam` — the first thing that worked |
| 61 | residual axes and bank units annealed to zero, then `drop` |
| 37 | `share` (value parallel to the key axis), `signin`, `lamfix` |
| 24 | `pinrow`, `opin`, fold restricted to two bank-2 rows (`p_rows`) |
| 21 | `tie` — bank 2 *is* bank 1, run on the post-attention residual |
| 20 | `prows` — the tied bank's third unit drops out of the fold |
| 19 | `zerobias` — the training-only read-out bias absorbed by the shift gauge |
| 18 | `dropscale` — the training-only read-out temperature |

The tie is the interesting rung, because it is not obvious that one ReLU bank
can do both jobs. Bank 1 must classify the *local* sum into absorb / transparent
/ generate; bank 2 must fold the *post-carry* residual mod 10. Both are "did
this reach ten" step functions, and once the code gauge is pinned they turn out
to be the same step function in the same place: the fold's two active knees have
to fall between `r(c=0, s=9)` and `r(c=1, s=9)`, which is exactly the window
bank 1's knees already straddle. Tying them costs a bank of biases and a
read-out matrix, and the fold then reuses only two of the three units, which is
where the next parameter comes from.

The bank-2 cut needed one extra idea. `code[a] + code[b]` carries twice the
code's offset while the read-out compares against a single code entry, so the
fold has to supply a constant, and in the wider models a whole always-on ReLU
unit was being spent on it. `y_bias` gives that constant a scalar to live in, the
unit anneals away against it, and `zerobias` then removes the scalar with the
shift gauge — so the constant ends up in the attention value's absorb baseline,
where it is free.

Three things make that retraining possible at all once the model is small:

* **Read-out temperature.** Pinning the residual gauge normalises the code to a
  spacing of about 0.1, and the logits are `-||y - code[d]||^2`, so a *perfect*
  model still has a cross-entropy near `log 10`. The gradient is then almost
  entirely "sharpen the read-out" and it tears the carry mechanism apart. Runs
  on a pinned model therefore carry the training-only temperature `exp(ls)`
  (`--ls_auto` calibrates it so the median margin between the correct prototype
  and its runner-up is a few nats) and `xform dropscale` removes it afterwards,
  which cannot change any answer.
* **Relative step size** (`--relscale`). A reduced model's weights span several
  orders of magnitude — a deep attention notch needs a large key gain next to an
  O(1) code — and Adam's step is `lr` in *absolute* units, so a single learning
  rate either freezes the large weights or destroys the small ones. Training
  `q = p / |p_init|` makes `lr` a relative step size instead. Warm-start noise is
  multiplicative for the same reason.
* **An anneal has to be a constraint, not a re-parameterisation.** Driving two
  weights together by interpolating them in the forward pass does not work, and
  fails silently: `b1` and `b2` reach the loss only through `mid +- s(b1-b2)/2`,
  so gradient descent simply grows the raw gap as fast as `s` falls. The live
  model stays at 1.00 for the whole schedule while the tied model it is
  supposedly converging on never improves at all — which is exactly what several
  tie runs did before the gap was instrumented. The schedule is now a hard cap on
  the raw residual, applied after every optimiser step and recorded before the
  first one, and `train.py` prints both the surviving gap and the accuracy of the
  fully cut model at every evaluation so this cannot hide again.

## What is fixed and what is learned

The shipped file registers 18 parameters and 4 buffers. Every buffer is an
architectural or gauge constant, not a fitted value:

| buffer | value | why it is not learned information |
| --- | --- | --- |
| `code_pin` | `1.0` | the residual stream's scale gauge; any positive value gives the same model |
| `o_pinned` | `1.0` | the key axis's scale gauge, likewise |
| `s1` | `+1.0` | the ReLU bank's input orientation. Either choice trains: the code simply comes out with the opposite sign, and `reorient.py` rewrites a bank from one orientation into the other. There is one such constant, not one per unit — a per-unit sign would be learned information smuggled out of the count, so `f1_in="sign"` shares a single `+-1` across the bank |
| `lam_c` | `-2.0` | the fixed ALiBi distance slope. It is a round number rather than a fitted one — the `-3.0` run reached the same accuracy |

There is no `s2` buffer: under `tie` the second bank *is* the first one, so it
reads through `s1`.

No per-unit sign, mask or table sits outside the parameter count, and the code
table is *not* parameterised as an affine function of the digit: all nine free
code values are learned, and their coming out evenly spaced (`code[d]` is within
half a percent of `1.0 - 0.1056 d`) is a result, not an assumption. Writing that
line as two parameters would save eight more, but it would be handing the model
the number line rather than letting it learn one, so it is not done.

Under those rules 17 is the floor for this architecture: three ReLU units are
the minimum that can cut a *notch* rather than a ramp (with two units `z` is
monotone in the local sum, so `s = 9` cannot sit below both of its neighbours),
two fold rows are the minimum that can make a step, nine free code values follow
from not parameterising the code, and all three of the `C = 1` gauge freedoms —
code scale, code shift, key-axis scale — are already spent.

## The 17th parameter, and why it is still there

The one reduction left is `kv="one"`: make the attention key scale and the
carry-value scale the *same* parameter, so that what a place broadcasts and what
it hands on are literally one projection. It is implemented (`lib.py`,
`model_src.py`, `xform.py --ops kvone`, and `check.py` covers three `kv="one"`
configs), and it does not work here. The reason is worth recording, because it
is a constraint rather than a failure to search hard enough.

With one shared scalar `w`, the key gap and the value gap are the same number:

    k_abs - k_gen = w * z_abs      must be well inside the distance bias |lam|
    v_abs - v_gen = w * z_abs      must be exactly one code step, i.e. -beta

so the constraint is `key gap = -beta`. Under the residual scale gauge the key
gap is invariant and `beta` scales, so *some* gauge always satisfies it — but
that gauge is the one where `code[0] = -19.3`, and `code[0] = 1` is already
spent pinning the scale. Pinning the code and sharing the scalar both consume
the same single degree of freedom. Keeping both is therefore a real equation on
the trained weights, and it forces the transparent notch to be about 250 times
`z_abs` rather than the 10 times the shipped model has, because the notch still
has to clear `2 |lam| L` for the longest carry chain while the key gap has
shrunk to one code step.

That target is representable — `kvinit.py` prints the ratio each member has and
the ratio it would need — but it is not reachable from the shipped model, and
the direct search does not find it either:

* A warm start with the shared scalar set to the parent's key scale trains to
  about 0.08 (four runs, 256 members each, sigma 0.002 to 0.008). It starts at
  0 accuracy, because the carry is then 20x too large, so there is no gradient
  pointing back at the mechanism.
* Aiming a scan straight at the predicted solution — sweep `z_abs` across its
  required range by moving the one bank-1 knee that is off at `s = 9`, and sweep
  the shared scale around `-beta / z_abs` — peaks at 0.08 as well. The reason is
  the tie: `b1` sets the fold's knees *and* the key geometry, so three numbers
  are already serving two masters and `kv="one"` adds a third.
* Annealing `code[0]` from the kv-one gauge down to the pinned one is a smooth
  path, but it has to cross `code[0] = 0`, where the shared scalar makes the
  key gap and the carry vanish together. There is no path around it.
* Training the whole thing from scratch — 768 seeds over three runs, with the
  fold untied so that bank 1 has only the key geometry to serve — plateaus at
  0.13 to 0.17, which is the "always attend to `i-1`" shortcut plus a little.
  The mechanism has never been found from scratch at this size in this project;
  every working model came down a ladder from a much wider parent, and the
  ladder is what `kv="one"` blocks.

Two things would buy the parameter and are deliberately not done. Pinning the
code row to `-19.3` instead of `1.0` makes the constraint free — but that
constant is this model's fitted ratio wearing a buffer's clothing, which is the
one thing a gauge constant must not be. Imposing `p_out[1] = -p_out[0]` (the
fold coefficients sum to `4e-3`, and they must sum to zero or the fold would
not saturate) also saves one — but that they cancel is something the training
discovered about addition, not a coordinate freedom, and building it into the
architecture is the same move as writing the code table as `c0 + beta * d`.

## Verifying the shipped file

`python verify.py` imports `submission.py` the way the grader does and reports:
parameter and buffer counts against the metadata; `add()` on random full-width
pairs; bulk batched accuracy on random 8-digit pairs; all 6561
generate/transparent/absorb carry patterns; hand-picked edge cases (maximal
chains, carry-out, all-nines); and an ablation that replaces the attention map
with its batch mean, which collapses accuracy — the attention pattern has to
depend on the input.
