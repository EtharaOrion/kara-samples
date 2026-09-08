# Minimal 8-digit addition transformer — attempt 3

## What is shipped

**37 parameters**, 1.000000 exact-match on 20M held-out uniform pairs and on
20M each at three carry-transparency enrichments (80M pairs, zero errors), and
on all 6561 carry-structure patterns. The bar is 99.00%.

`submission.py` is emitted by `build_submission.py` from a training checkpoint.
It contains the model class (copied verbatim from `model_src.py`), the trained
weights as plain Python float literals, `build_model()` and `add()`. It imports
only `torch`, `torch.nn`, `torch.nn.functional`. No training code, no data
generation, no encoded blobs.

`add(model, a, b)` splits both operands into decimal digits, runs **one**
forward pass, and arg-max decodes the nine output positions. Nothing about the
sum is computed outside the model: the digit split is a positional read of the
decimal literal and the recombination is `sum(d * 10**i)`.

## Architecture

Single Macaron-style block: FFN → 1-head causal self-attention → FFN, over
per-place digit-pair tokens. Sequence length 10, least-significant digit first:

| position | content |
|---|---|
| 0 | boundary / attention sink, embedded as `2*emb[0]` |
| 1..8 | place `i-1`, embedded as `emb[a_i] + emb[b_i]` from one shared 10×d table |
| 9 | carry-out slot |

Position `p` in 1..9 predicts sum digit `p-1`, so all nine digits come out of a
single forward pass. Unembedding is tied to `emb`; normalisation is
parameter-free RMSNorm; the attention bias is two learned scalars
(`alibi·(i−j)` + `self_bias·1[i=j]`).

**How the model solves it.** Each place attends back to the nearest
**non**-transparent place -- the one that actually decides its carry-in -- so
the attention pattern is a function of the digits rather than of position.
Freezing the attention map at its batch mean collapses accuracy on carry-heavy
inputs (see `verify.py`). How the two ReLUs and the norm divide that work up is
measured rather than assumed; see below.

## What the trained model turned out to be doing

`analyze.py` reads the mechanism out of the shipped 37-parameter model rather
than assuming it. (Every number here also holds for the 42- and 39-parameter
candidates: the steps between them are exact rewrites, so all three compute the
same function, and only the basis differs.)

**The digit codes lie on a parabola.** Fitting the ten learned codes in the
readout plane gives

    x = +1.5967 * digit - 7.2638                      R^2 = 0.99988
    y = -0.0286 * digit^2 + 0.2077 * digit - 0.0768   R^2 = 0.99952

so the model discovered the moment curve `(t, t^2)`. Both halves matter. The
*linear* coordinate means the token `emb[a] + emb[b]` encodes `a + b` directly:
within a fixed `a+b` the coordinate varies by at most 0.259 against a step of
1.764 per unit of `a+b`, which is the sufficient statistic for the task. After
the on-axis rewrite this is not just a coordinate but *the* pre-activation of
the pre-attention FFN, so that unit demonstrably sees `a+b` and nothing else. The
*quadratic* coordinate is what makes the readout possible at all -- points on a
convex curve are all vertices of their convex hull, so a linear tied readout
can select any one of the ten. A collinear code could not: its arg max would be
monotone in the digit and only 0 and 9 could ever win.

This also retires an idea that looked attractive on parameter count -- storing
each code as an angle on a fixed-radius circle, 11 numbers instead of 20. The
learned radii span a factor of 9.4 and do so deliberately, because the radius
is the digit. An equal-norm constraint would delete the very structure the
model relies on.

**Each ReLU computes one predicate of the algorithm.** Carry lookahead needs
two: *generate* (`a+b >= 10`), and the mod-10 *wrap* (`a+b+carry >= 10`). There
is exactly one hidden unit available for each, and each is used for exactly
that:

| unit | fires iff | agreement |
|---|---|---|
| pre-attention ReLU | `a+b >= 10` | 100.00% |
| post-attention ReLU | `a+b+carry >= 10` | 95.42% |

The post-attention unit's disagreements sit on `a+b == 7` with carry 1, where
its pre-activation is essentially zero (-0.015) and the residual stream rather
than the ReLU settles the case.

This corrected a guess of mine that had been in these notes: that the
pre-attention FFN existed to detect *transparency* (`a+b == 9`), on the
reasoning that keys are linear in `emb[a]+emb[b]` and a linear function of a
sum cannot express a non-monotone predicate. That argument neglects RMSNorm,
which sits between the two and is not linear. Transparency is actually detected
**by the norm acting on the parabolic code**: at `a+b == 9` the linear
coordinate of the token passes through zero, so normalising swings the
direction onto the quadratic axis -- exactly the non-monotone response a linear
key needs. The measured pre-attention unit agrees with `a+b == 9` only 31% of
the time. It is computing generate, not transparency.

**Attention performs carry lookahead.** On `19999999 + 10000001`, where place 0
generates a carry and places 1-6 are transparent, every transparent query
attends to position 1 -- the generating place -- with mass 0.996. On
`12345678 + 87654321`, which is transparent everywhere with nothing generating
below it, every query instead attends to position 0, the boundary sink, and
reads carry-in 0; the sum is 99999999. That is the standard parallel-prefix
rule ("look back to the nearest place that is not transparent, and take its
decision"), and it is a function of the digits, not of position, which is why
freezing the attention map at its batch mean collapses accuracy from 1.0000 to
0.10 on carry-heavy inputs.

## Training

`train.py` + `ens.py` + `data.py`. Seed variance dominates at these sizes, so
E=256 independently-initialised models are trained *simultaneously* on a
leading member axis via einsum (Adam is elementwise, so members stay
independent; gradients are clipped per member). Data is synthetic and generated
on-GPU: ~35% uniform, ~40% carry-transparent-enriched, ~25% explicit maximal
carry chains, with a hash-bucketed held-out split that training never sees.
`--rounds` splits the run into that many one-cycle LR schedules, reinitialising
the worst half of the members between rounds.

Two trainer details mattered a lot:

* **Hall of fame at every evaluation.** These models are not monotone in
  training time — a member can peak mid-cycle and drift away. Snapshotting only
  at round boundaries threw away working models (a member seen at 0.9945 was
  logged at 0.858 by the end of its round). Snapshotting every evaluation and
  rescoring the whole hall of fame on 1M fresh pairs at the end turned several
  "failed" configurations into successes.
* **The chain generator must emit transparent runs with no generate below
  them.** Without that case the model never learns to resolve a long
  transparent run against the boundary slot, and misses inputs like
  `98190410 + 91809589`.

## Parameter frontier (this attempt)

Each row was shipped only after passing the full `verify.py` suite: >=2M uniform
held-out pairs, all 3^8 = 6561 carry-structure patterns, the frozen-attention
ablation, the `add()` API path, and hand-picked edge cases.

The shipped 37 was then re-checked on 20M fresh pairs from each of four
distributions -- uniform and transparent-enriched at p = 0.50, 0.85, 0.95 --
for **80,000,000 pairs with zero errors**. The bar is 99.00%.

| params | change from the row above | uniform acc | how |
|---|---|---|---|
| 150 | attempt 2's d=4 model | 1.0000 | starting point |
| 130 | d3, f=6/6 | — | trained |
| 74 | f=2/2 | 1.0000 | trained |
| 60 | f=1/1 | 1.0000 | trained |
| 59 | drop `logit_scale` | 1.0000 | **free** |
| 56 | share the Q/K projection | 1.0000 | trained |
| 53 | drop `b_in2` | 1.0000 | trained |
| 52 | drop `b_q` | 1.0000 | trained |
| 48 | rank-2 factorised embedding | 1.0000 | trained |
| 42 | gauge-fix `emb_up` to the constant injection | 1.000000 | **free** |
| 39 | confine both FFNs' stream I/O to the digit plane | 1.000000 | **free** |
| **37** | **rotate the plane onto the FFN's read axis, absorb its scale** | **1.000000** | **free** |

## Four reductions that cost nothing

Four of those steps remove parameters without changing the function at all.
Each is a symmetry of the block -- a direction in parameter space along which
the model is constant -- so the scalars removed were never carrying information
about the mapping from operands to digits. Each has its own script, and every
script recomputes both models and **refuses to write** unless the check passes.

1. **`logit_scale`** (-1, `drop_scale.py`). One *positive* scalar multiplying
   all ten logits at a position. It sets the softmax temperature and cannot move
   an arg max.

2. **`emb_up`** (-6, `gauge.py`). With a rank-`r` factorised embedding the block
   has two exact symmetries: `GL(r)` on the factorisation
   (`emb_lo M, M^-1 emb_up` gives the same table) and `O(d)` on the residual
   stream, because RMSNorm is equivariant (`||xQ|| = ||x||`) and every other
   operation either absorbs `Q` into a weight or commutes with it. Together they
   act transitively on rank-`r` up-projections, so any trained `emb_up` maps
   exactly to the constant injection `[I | 0]`. Verified in float64: the logits
   move by 3.8e-11 absolute, 7.0e-14 relative.

3. **Plane-confined FFN I/O** (-3, `confine.py`). Given `[I | 0]`, the
   pre-attention FFN is the first thing in the block, so the stream it reads is
   the embedding alone and is *identically zero* outside the digit plane -- the
   discarded rows of `w_in1` multiply exact zeros. And the readout is tied to
   the same plane, so whatever the last FFN writes outside it reaches the logits
   only through the RMSNorm denominator: one positive factor common to all ten
   classes, i.e. `logit_scale` again. Checked by direction, since the logits do
   change by a positive scale: normalised logit vectors agree to 3.3e-16, the
   induced rescaling stays in [0.104, 50.4].

4. **On-axis pre-attention FFN** (-2, `axis.py`). Fixing `emb_up` does not use
   up the whole gauge group -- the stabiliser of `[I | 0]` still contains an
   `O(r)` rotation *within* the plane. With `d_ff_in == 1` that leftover is
   exactly enough to align the plane's first axis with the single direction this
   FFN reads, after which `w_in1 = (t, 0)`; and since `relu(t z) = t relu(z)`
   for `t > 0`, the surviving `t` moves into `b_in1` and `w_in2`. The read
   matrix is then the constant 1 and is not stored. Verified in float64: logits
   move by 3.4e-12, 2.7e-13 relative, predictions identical on 200k samples.

   The result is an ordinary architecture, not just a rewrite -- the
   pre-attention FFN's pre-activation *is* the first residual coordinate -- and
   `train.py --ffn_in_axis` trains it directly. It is, however, *harder* to
   train from random initialisation than the model it came from, even though the
   two function classes are identical: the read direction is no longer free, so
   the optimiser has to rotate the digit table into alignment with it rather
   than the other way round. See the trainability note below.

**Where this stops.** Two further symmetries exist and are deliberately not
used. The second FFN has the same ReLU scale freedom, but `w_out1` is a
3-vector, so removing it means storing a unit vector as two angles. And the
`O(1)` left outside the plane is only a sign. Both would lower the count by
re-encoding a weight rather than deleting one, and the shipped module would stop
reading like a model -- a parameter reconstructed as `cos`/`sin` of another, or
padded with a hard-coded 1 that silently carries a learned sign bit. The line I
am holding is that every parameter tensor in `submission.py` is a plain weight
and every constant in it is genuinely constant. (An earlier version of this note
declined the `O(r)` rotation on the same grounds. That was wrong and I reversed
it: deleting a coordinate that a basis choice makes exactly zero leaves the
forward pass *simpler*, and hides nothing, which is the opposite of the angle
encoding.)

## What was ruled out

* **d_model = 2** (would save 10 parameters in the embedding table alone).
  Structurally blocked, not capacity-blocked: 82 params reaches only 0.85, and
  adding FFN width barely helps (52 params → 0.72). The reason is that
  `emb[a]+emb[b]` of a circular code has *half*-angle `∝ (a+b)/2`, so the block
  would have to implement angle doubling to read the digit out. Removing the
  internal RMSNorms to give the model back the magnitude channel (a
  zero-parameter change) made it *worse*, not better: 0.64 at 82 params.
* **Rank-1 value/output projection at d=2** (`--dv 1`): 0.05, dead.
* **Every remaining trim.** Once the free reductions are taken, each of the
  four things left that could be deleted was warm-started from the best working
  donor and trained. All four fail decisively, which is what puts the floor
  here rather than lower:

  | trim | donor | params | best of 256 members |
  |---|---|---|---|
  | drop `self_bias` (the `i == j` attention bias) | 42 | 41 | 0.6639 |
  | drop `alibi` (the `i - j` attention bias) | 37 | 36 | 0.2050 |
  | drop `b_out2` (the post-FFN residual bias) | 42 | 39 | 0.3417 |
  | drop the pre-attention FFN entirely | 42 | 35 | 0.0719 |

  The two attention-bias results are the ones worth reading twice. Removing
  either collapses the model, and `alibi` is the more essential of the two:
  carry lookahead means "attend to the **nearest** preceding place that is not
  transparent", and without a term monotone in `i - j` there is nothing to break
  the tie between two equally non-transparent keys. The remaining 37 parameters
  are all load-bearing in this sense, not merely in the zeroing sense.

  `self_bias` was retried once more from the 39-parameter donor, in case
  confining the FFN I/O had improved the conditioning; it was at 0.42 after two
  of four rounds, tracking the earlier failure, and was stopped.
* **Dropping `b_out2`** also failed earlier from a 52-parameter donor (0.78
  cold, 0.11 warm), and **dropping both residual biases** gave 0.53.
* **Pinning the digit-0 embedding row to zero** (`--emb0_zero`, -2). The tied
  readout makes the class-0 logit identically zero, so digit 0 can only win
  when all nine other codes have negative inner product with the stream --
  impossible once the codes ring the origin, which is the arrangement the model
  actually learns. Implemented and left in the trainer, but it fights the
  readout geometry rather than saving anything real. Smoke test: 0.0029.
* **Storing each digit code as an angle on a fixed-radius circle** (11 numbers
  instead of 20). Retired by measurement, not taste: the learned radii span a
  factor of 9.4 and do so deliberately, because the radius is the digit.
* **Fitting the digit table to the curve it discovered.** The single largest
  remaining cost is the embedding: 20 of 37 parameters. Since the ten codes lie
  on a parabola to R^2 = 0.9999, they could be *generated* as
  `emb[d] = (a*d + b, c*d^2 + e*d + f)` -- five parameters instead of twenty,
  a 22-parameter model. Rejected. That makes the digit enter the network as a
  *number* rather than as an index, and the linear coordinate then contains
  `a + b` by construction rather than because anything was learned: it hands the
  model the exact sufficient statistic that the interesting part of this result
  is that it found on its own. It is the same objection as the fixed-basis entry
  below, only sharper, because here I would be hard-coding the specific
  structure I had just measured the model discovering.
* **Binary / non-decimal tokenisation.** Would shrink the embedding table a
  lot, but converting an operand to base 2 inside `add()` is real arithmetic on
  the operand done outside the model, and it makes the task the model actually
  solves much easier. Rejected on those grounds, not on grounds of difficulty.
* **Fixed (sinusoidal or random) digit-feature basis with a learned
  projection.** Would cut the 30-parameter embedding table to ~12. Rejected:
  a fixed random basis just hides floats the answer depends on in a buffer, and
  a sinusoidal basis hands the model the cyclic group structure of Z/10 for
  free. Neither is a smaller *model*, only a smaller parameter count.

## Is the 37 an architecture, or just a rewrite of the 39?

Both, and it matters which claim is being made, so here is what was actually
measured.

**The chain is general, not tuned to one lucky member.** The 48-parameter
checkpoint's hall of fame holds 13 members at 1.0000 accuracy, several of them
genuinely different solutions (pairwise weight distances up to 114). Running the
whole reduction chain -- strip, gauge, confine, on-axis -- on two of those,
untouched since training, gives two more 37-parameter models, both 1.000000 on
2M uniform pairs and on all 6561 carry patterns. They are different models, not
the same one twice: their frozen-attention signatures are 0.6462/0.0236 and
0.6386/0.0184 against the shipped model's 0.6744/0.1042. So the four reductions
are properties of the architecture, not coincidences of one weight vector.

**But it does not train from scratch under this budget.** `train.py
--ffn_in_axis` with E=256 for 60k steps reaches 0.7821; widening to E=512 for
140k steps over 7 restarts got to 0.8452, closer but still short. That is worth stating plainly rather than glossing: fixing the
read direction makes the optimisation harder even though the function class is
unchanged, because the optimiser now has to rotate the digit table into
alignment with a fixed axis instead of moving the axis to the table.

This is not special to the on-axis step. **No rung below 74 parameters was
reached from random initialisation** -- the whole lower frontier is a warm-start
cascade, each rung started from the previous *working* rung, which is the method
the `--init_from` flag exists for. The 37 sits at the end of that cascade like
every other row. What the weights are is unaffected either way: they came from
training I ran, and the rewrites that follow are exact.

## Zeroing probe

`probe.py` zeroes each parameter of the shipped model in turn and re-measures
held-out accuracy on 300k uniform and 300k carry-heavy pairs. This is how each
of the free reductions above was *found*; nothing was ever removed on the
strength of the probe alone, only after an independent argument that it could
not change a prediction.

On the shipped 37-parameter model every scalar is load-bearing: the best any
single zeroing achieves is 0.9685 uniform, and most collapse to below 0.1.
The 39-parameter model had exactly one near-free scalar left (`w_in1[1]`, the
component of the FFN read direction off the linear axis, 0.999903) -- that is
what pointed at the leftover in-plane rotation and produced reduction 4.

## Files

| file | role |
|---|---|
| `submission.py` | **the graded artefact** |
| `model_src.py` | architecture, copied verbatim into the submission |
| `ens.py` | E-way batched copy of the same model, for the seed lottery |
| `data.py` | on-GPU synthetic data + held-out split |
| `train.py` | trainer |
| `transfer.py` | warm-starting a smaller config from a working larger one |
| `sweep.py` | queue runner |
| `drop_scale.py` | removes `logit_scale`, proving predictions are unchanged |
| `gauge.py` | removes `emb_up` by gauge fixing, proving logits are unchanged |
| `confine.py` | confines both FFNs' stream I/O to the digit plane, ditto |
| `axis.py` | rotates the plane onto the pre-attention FFN's read axis, ditto |
| `probe.py` | zeroes each parameter in turn to find what is not load-bearing |
| `cfgutil.py` | rebuilds a model from a checkpoint's stored config |
| `promote.py` | checkpoint → candidate file → verification, in one step |
| `verify.py` | the audit suite every shipped candidate must pass |
| `analyze.py` | reads the learned digit geometry and attention back out |
| `cand_*.py` | verified candidates at each parameter count |
| `ckpt/`, `logs/` | checkpoints and training logs |
