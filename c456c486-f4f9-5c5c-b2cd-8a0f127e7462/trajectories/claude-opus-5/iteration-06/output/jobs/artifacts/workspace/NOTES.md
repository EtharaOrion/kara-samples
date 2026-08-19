# Attempt 6 — notes

## Decision: push further on attempt 5's line, changing the radix

Attempt 5 = 21 params (score 0.9230), decimal, 1-layer/1-head carry-lookahead.
Of those 21, **10 were the digit embedding `U`** (one learned scalar per decimal token).
The record says the only remaining cut there was `U[d] = alpha*d + beta`, which was
correctly rejected as hard-coding the arithmetic into the tokenizer.

A decimal rebuild with every exact fold I can find still lands at ~19-20 params
(score ~0.93) — i.e. essentially no improvement over what I already have.

The task statement explicitly lists **tokenization** as a design axis I may change.
So: keep the identical learned mechanism, tokenize the operands in base 2.

* digit-token table: 10 learned scalars -> 2 learned scalars
* per-slot feature layer: 3 ReLU units -> 2 (only 3 distinct token-pair sums exist)
* sequence: 49 slots (sentinel + 48 bit positions) instead of 16
* the *hard* part is unchanged and in fact harder: carry chains are longer
  (propagate probability 1/2 per slot, chains up to 47) and the attention must
  still pick the nearest non-propagate slot.

Nothing about the answer is computed outside the model: `add()` does a radix
conversion of the operands (`format(n,'048b')`), one forward pass, and a radix
reconstruction of the output tokens (`int(s, 2)`) — exactly the decimal pipeline
with a different base.

## Result

**Shipped: 7 parameters**, verified error-free on 4,000,000 unseen uniform 14-digit
pairs, 4 x 1,000,000 stress pairs, and every forced carry-run length 0..47
(details at the bottom of this file). The 8-parameter version below was built and
verified first and is kept as `submission_8param_backup.py`.

## Architecture (8 parameters, the first working version)

Slots: 0 = phantom (0,0) sentinel, 1..48 = bit pairs of a and b, LSB first
(operands < 10^14 < 2^47 so slot 48 is always (0,0); the sum < 2^48 fits).

    z_i   = U[a_i] + U[b_i]                      # U: 2 learned scalars
    h1_i  = relu(z_i + b1) ; h2_i = relu(z_i + b2)   # b1,b2 learned; W1 fixed to (+1,+1)
    c1_i  = w1*h1_i + w2*h2_i                    # key/query feature; w1 FIXED = -1 (gauge, see below)
    c2_i  = h1_i                                 # value feature (value projection = e_1)
    score(i,j) = (c1_i + bq)*c1_j + slope*(j-i)  # bq learned; slope FIXED = 4.0 buffer
    cin_i  = softmax over j <  i  of score, applied to c2      (strictly causal)
    cout_i = softmax over j <= i  of score, applied to c2      (causal with self)
    y_i   = z_i + wo1*cin_i + wo2*cout_i         # wo1,wo2 learned
    logit(i,d) = 2*y_i*U[d] - U[d]^2             # weightless tied readout (nearest centre)

Learned: U (2), b1,b2 (2), w2 (1), bq (1), wo1,wo2 (2) = **8**.

### Why `w1 = -1` is a gauge fixing, not an imposition
The parametrisation has an exact 1-dim invariance:
`(U, b, w) -> (lambda*U, lambda*b, w/lambda)` for lambda > 0 leaves c1 and every
score identical, scales z, c2, y by lambda, and scales the readout logits by
lambda^2 — argmax unchanged. So one degree of freedom among {scale of U, scale of
b, scale of w} is provably redundant. I fix the scale on `w1` (an internal
feature scale) rather than on U, so **both digit-embedding values stay learned**.
Sign: a notch at the propagate token requires w1 < 0 in every case (checked by
enumerating the piecewise-linear shapes), so fixing the sign loses no solution.
Same argument makes the value-feature scale redundant with `wo`, so `c2 = h1`
costs nothing.

### Why the mechanism needs what it has
* 2 ReLU units: c1 must be a *notch* over the 3 distinct z values
  {kill, propagate, generate}; one unit is monotone in z, so 2 is the minimum.
* two masks, one head: cin_i = cout_{i-1}; the self-inclusive read supplies the
  wrap-by-10 (here wrap-by-2) subtraction. Costs 1 param (wo2) instead of an MLP.
* the fixed ALiBi slope makes recency break ties among non-propagate slots;
  the learned notch must beat it by ~slope*chain_length, which is what forces
  c1(generate) ~= c1(kill).

## Hand-checked closed form (architecture sanity only — NOT used as weights or init)
U=(0,1), b=(-0.5,-1.5), w2=3, bq=1000, wo=(2/3,-4/3) satisfies every case.
Shipped weights come from training from random inits only.

## 8 -> 7 parameters

Two changes, neither of which tells the model anything about binary arithmetic:

**(a) the value path is linear.** In the trained 8-param cell `h1 = relu(z + b0)`
was active at all three token-pair sums, i.e. the ReLU on the value feature was
doing nothing. A standard transformer value projection *is* linear, so `c2 = z`
(the residual stream itself, with its scale absorbed into `wo`). This does not
change what the model can express here; it removes a nonlinearity that was
never firing.

**(b) `b0` is an exact gauge freedom.** With `c2 = z + b0` and
`c1 = w1*(z + b0) + w2*relu(z + b1)`, the transformation

    U -> U + eps ,   b0 -> b0 + delta ,   b1 -> b1 - 2*eps ,   bq -> bq - 2*w1*eps

leaves every prediction identical when
`eps = -(wo0 + wo1)*delta / (1 + 2*(wo0 + wo1))`. Proof sketch:
`z -> z + 2eps`; `relu(z + b1)` is unchanged by construction; `c1` picks up the
constant `2*w1*eps`, and a constant added to every **key** shifts each score row
by a term that depends only on the query index, so the softmax is unchanged
(the matching query shift is absorbed by `bq`); the attention read therefore
moves by exactly `2eps + delta`, so `y -> y + 2eps + (wo0+wo1)*(2eps+delta)`
while the readout centres move by `eps`. Nearest-centre decoding is invariant
iff those two are equal, which is the condition above. So `delta` is free and
`b0 = 0` costs nothing. Confirmed numerically: applying the map to the shipped
8-param weights reproduces its logits.

Learned in the 7-param model: U (2), b (1), w2 (1), bq (1), wo (2) = **7**.

### Why I stopped at 7 and not 6
The obvious further cuts are all *impositions*, not gauges, and each one hands
the model a fact about base-2 addition that it is supposed to learn:
* `w2 = 2` (or any fixed value) hand-designs the propagate detector — the notch
  in the key feature is exactly the thing the model has to discover.
* `wo = (1/2, -1)` is the value the closed form needs; writing it down is
  writing down the carry arithmetic.
* `U0 = 0` hand-places the additive identity in the embedding.
None of these is a redundant direction in parameter space: perturbing them
changes the function. 7 is the floor for this parametrisation.

## Training story (what actually made it work)

Plain descent reliably lands in a **blurred-carry** local optimum: attention
spreads over many earlier slots, the read is analogue rather than a lookup, and
accuracy dies on carry chains longer than ~8. `diag.py` measures the two
conditions the mechanism needs — `q*notch > slope*48` (attention must find the
nearest non-propagating slot over a full-length chain) and `q*spread < slope`
(among non-propagating slots recency, not content, must win). In the blurred
basin `q*notch` was 27 against a requirement of 192. A grid showed that
multiplying `bq` by 8 *and* `wo0` by ~2 reaches perfect accuracy while either
move alone makes the loss worse: a coupled ridge that gradients will not climb.

Three things together escape it, all in `train.py` / `train7.py`:
1. a **population** of M cells trained in parallel (one leading model dimension),
2. **basin hopping** — every few hundred steps, cull the worst 75%, resample
   them from the elite with a log-normal coordinate perturbation, and zero their
   Adam state,
3. **random ALiBi slope jitter** — the recency slope is multiplied by a random
   factor each step. This was the decisive one: a blurred read can be calibrated
   to one fixed slope, but not to a slope that keeps moving, so the only stable
   solution is a sharp lookup.

A late lesson: the hardening curriculum plus hopping can *destroy* good cells
once the difficulty reaches full stress, so `train()` now snapshots the best
population it has ever seen to `<out>.best`; two 24k-step runs were lost to this
before it was added.

## Verification of the shipped 7-param model

`verify.py` generates its own data and calls the shipped module's forward pass.

    parameters: 7
    U  [-18.62794, -4.72366]   b 24.82960   w2 1.80859   bq 2.73475
    wo [  0.63350, -1.13790]
    uniform 14-digit : acc 1.000000   0 wrong / 4,000,000   min logit margin 174.58
    propagate p=0.5/0.9/0.99/1.0 : acc 1.000000, 0 wrong / 1,000,000 each
    forced carry runs 0..47      : worst acc 1.000000
    edge cases (0+0, 10^14-1 twice, ...) : all correct
    add() on 3000 random pairs   : 3000/3000
    attention over 64 inputs     : max per-cell spread 1.0000 (not a fixed pattern)
    slot 40 attends to slot      : 38, 38, 39, 36, 38, 35, 36, 39 (input-dependent)

`diag7.py` on the shipped cell:

    kill(0,0): z -37.2559  key +37.2559   prop(0,1): z -23.3516  key +26.0247
    gen (1,1): z  -9.4473  key +37.2676
    notch depth 11.231   -> q*notch  = 323.0  (needs > 192)
    kill/gen spread 0.012 -> q*spread =  0.34  (needs < 4)

Selection among the 188 converged cells (`select7.py`) was by worst-case logit
margin over 491,520 pairs spanning every regime and every forced run length;
the chosen cell decides by 174.6 where the first error-free cell I found decided
by 45.8.

Training found solutions on both sides of my hand-derived one: here the ReLU
fires on the propagate *and* generate tokens and lifts the generate key up to
meet the kill key; an earlier cell lifted the kill key up to meet the generate
key instead. Only the notch at the propagate token matters, not which side
builds it.
