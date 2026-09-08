# What the model does, and why it is the size it is

The graded file is `submission.py`.  Everything else here is how it was made:
`train_ens.py` trains, `cascade*.py` / `descend.py` / `lean1.py` sequence the
shrinking, `warm.py` re-expresses a trained parent on a smaller architecture,
and `verify.py` audits the result.  Nothing computes an answer outside the
model: `add` reads one forward pass, and every number in the file came out of
gradient descent.

## The mechanism

Tokens are digit *pairs*, least-significant place first, with a `(0,0)` sink at
position 0 and a virtual place 8 at position 9, so one forward pass emits all
nine output digits.  A token embeds as `code[a] + code[b]`, one learned table of
ten numbers that is also the set of output prototypes.

Addition is local except for the carry, and the carry is what the attention is
for.  Place `i` carries in iff the most recent earlier place that does not
*propagate* (digit sum != 9) *generates* (digit sum >= 10).  So:

* `f1` reads the embedding and writes two features — a **key** that dips at
  digit sum 9 and is level elsewhere, and a **value** that separates sums <= 8
  from sums >= 10;
* the attention scores `(key + b_q)` against itself plus a relative-position
  bias, which makes it pick the *nearest* non-nine, and reads that place's
  value;
* `f2` folds the result mod 10 on the answer axis, which the readout compares
  against the ten prototypes.

The measured tables (`scratch/`) match this exactly.  In the shipped model the
two features share one residual axis, so there is a single learned function of
the digit sum doing both jobs: it is level for sums up to 8, notches down at 9,
and drops to zero from 10 up.  Measured on the shipped weights:

    digit sum   0..8      9        10..18
    feature     +0.089   -1.679    0.000

The notch is what the attention reads as "not a nine"; the gap between the two
level regions either side of it is what it reads as "this place carries out",
and the attention's output scale turns that 0.089 into 0.134 on the answer
axis, against a code step of 0.126.  Three ReLUs, three levels — which is the
width argument below, seen from the other end.

## Why the count is what it is

The shipped model has **21 parameters**:

| | | |
|---|---:|---|
| `code_p` | 9 | ten digit codes, one of them pinned by `code_fix 1` |
| `f1_b` | 3 | the pre-attention FFN's three knees |
| `f1_o` | 3 | its three amplitudes, onto the one shared key/value axis |
| `b_q` | 1 | the query offset |
| `w_o` | 1 | the attention's output scale |
| `f2_b` | 2 | the fold's two knees |
| `f2_o` | 2 | the fold's two amplitudes |

Neither FFN can be deleted, and the reasons are not empirical:

* **`f1` must exist.**  Without it the attention keys are `code[a] + code[b]`
  directly, and no table of ten numbers makes that quantity mark `a + b = 9`.
  It would need `c_a + c_b` constant on the anti-diagonal and strictly larger
  off it; the first gives `c_0 + c_9 = c_1 + c_8`, and the second applied to
  `(0,8)` and `(1,9)` gives `c_8 > c_9` and `c_9 > c_8`.  A nonlinearity is
  the only way out.
* **`f2` must exist.**  Dropping it means reading the answer off `code[a] +
  code[b]` plus the carry with no fold, which needs `code` to turn addition
  mod 10 into addition in the reals — a homomorphism from `Z/10` into a
  torsion-free group, so the zero map and nothing else.

Both widths are floors too:

* **`f1` needs three units.**  Its key must be low at 9 and level on *both*
  sides of it.  Two units give three regions but the middle level is forced
  between the outer two, so the dip cannot be the minimum; the third knee is
  what buys an independent middle.  (Measured directly: the two-unit fit puts
  `key(9)` above `key(<=8)`.)
* **`f2` needs two.**  The mod-10 fold is a step, and a step is a ramp that
  stops — one knee to start it, one to stop it.  Neither unit is spare: with
  both facing up the flat part above the step needs their amplitudes to cancel,
  which is what makes the step height `o1 * (b1 - b2)` and leaves nothing over
  to write a constant with.

`b_q` is not spare either.  Without it a query whose key feature happens to
evaluate to zero scores every earlier place identically on content, and the
position bias alone decides — the wrong place, for that query.  Giving the
offset to an always-on fourth `f1` unit instead would cost two parameters to
save one.

The count does not depend on where the redundancy is spent.  There are exactly
two scale freedoms in the block, one on the key/value axis and one on the
answer axis, so pinning any two of `w_o`, the position slope and `code[1]`
lands on the same 21; the three-axis layout floors at 23 the same way.

## Gauge versus structure

The residual axes have no natural units, so several parameters are redundant
rather than useful, and `warm.py` folds each into the code table exactly before
the child retrains.  All of these cuts are exact — the remapped child scores
1.0000 on both held-out sets *before* any training, which is the check in
`scratch/zeroshot.py` — and each child is then retrained anyway, so every
shipped number is one gradient descent chose.

    three axes  LEAN(3,2)  31 -> strict 30 -> alibi_fix 29
                           -> code_fix 1 28 -> f1_id_in 25 -> f2_id_in 23
    two axes    LEAN1(3,2) 29 -> strict 28 -> alibi_fix 27
                           -> code_fix 1 26 -> f1_id_in 23 -> f2_id_in 21

The full lineage of the shipped weights, every rung at 1.00000 held-out:

    cs_43_p 38 -> o53 38 -> o43 35 -> od_33 32 -> od_32 29 -> og32_* 21

One sign condition governs whether `*_id_in` is available at all.
`sign(f_w * code[1])` is gauge invariant, and `code_fix 1` rescales the answer
axis by `1 / code[1]`; pinning a read weight to `+1` is only exact when that
product is positive.  A parent with `code[1] < 0` therefore blocks the cut, and
the way through is to refit at the rung above until the pool picks an
up-facing bank — which is what happened at 4 -> 3 units.

Two of these have a direction that matters and cost a rung each to find:

* **`alibi_fix` must be pinned at or above the slope the parent learned.**
  Pinning it multiplies the whole score by the ratio; the key axis undoes that
  on the content term (the score is quadratic in the axis, so the axis takes
  the square root), but the softmax temperature goes with it.  Pinning low
  hands the child a blurred copy of its parent's attention.
* **the loss temperature has to follow the code's scale.**  `code_fix` renames
  the answer axis' unit, which shrinks the prototype gap ~8x and the squared
  margins ~60x; a fixed `--temp` then means something different on every rung.
  `train_ens.py` scales it by the gap present at initialisation.

`code_fix 2` is *not* gauge and should not be treated as one.  Moving the code's
origin needs a constant written back onto the answer axis, and nothing left in
the block can write one (`f1`'s feature is zero above sum 9, so the attention
carries nothing through there, and both `f2` units are spent on the fold).
Worse, pinning digit 0 at 0 *and*
digit 1 at 1 forces an affine code to be exactly `d`, so it is not a
normalisation the model can absorb — it is a different solution it has to find.
It fails from every warm start tried, which is the expected result.

## The two-axis merge

Sharing one axis between the attention's key and its value is not a gauge move:
the value's share of the score is real, and the key's share of the answer is
real.  Both are only tolerable in a band —

    key drift < 0.0375 * notch depth / value spread

— and a parent that never had to keep its key flat lands outside it (measured:
drift 0.063 against a bound of 0.051).  Merging at the *widest* rung instead,
where the extra units let the child flatten its own key, works: `lean1.py`
takes LEAN(4,3) to LEAN1(5,3) and trains it back to 1.0000.

## Is 1.0000 actually 1.0000

`code_fix` names the answer axis' unit, and the unit it happened to name leaves
the ten prototypes about 0.126 apart.  A readout that measures a squared
distance between numbers that close is worth checking rather than trusting, so
three things were measured beyond the held-out sets:

* **`scratch/margin.py`** — how far the residual stream ever lands from a
  decision boundary.  Over 400k held-out pairs and all 6561 carry structures
  the worst case is **0.0371 prototype steps**, and the float32-against-float64
  disagreement on the same quantity is 4.2e-5 steps: about 880x of headroom.
* **`scratch/adv.py`** — a search for a wrong answer rather than a sample of
  them.  Coordinate descent over the eight digit pairs, 4096 random restarts in
  parallel, each round trying all 100 pairs at one place and keeping whichever
  drives the model closest to a wrong digit.  Every restart converges to the
  same 0.03705 and none goes below it, so that number is a structural floor of
  the decision function and not an artefact of sampling.  Zero wrong answers.
* **`scratch/devcheck.py`** — CPU float32, CUDA float32 and CUDA with TF32
  matmuls all give identical answers on 406,561 inputs.  The margin is not
  hardware dependent.
