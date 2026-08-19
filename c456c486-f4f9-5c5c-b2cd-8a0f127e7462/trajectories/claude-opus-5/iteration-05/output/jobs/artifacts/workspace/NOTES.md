# 14-digit addition in 21 parameters

`submission.py` holds a 21-parameter single-layer transformer that is exact on every
test run (20M sampled pairs across 5 regimes, a 3M-pair sweep over forced-propagate run
lengths 0-14, all edge cases, and an exhaustive sweep of all 100 digit pairs at all 15
slot positions inside a full propagate chain). Minimum readout margin 5.41 logits, so
the result is structural rather than a float32 coincidence.

## The circuit

Carry lookahead expressed as attention. Per slot the digit pair is embedded as one
scalar `z = U[a] + U[b]`. A width-3 ReLU layer turns `z` into two features:

* `c1` — zero on both plateaus (digit sum <= 8 and >= 10), a sharp negative spike at
  sum == 9. Sum 9 is exactly the *propagate* condition.
* `c2` — a two-level step: one value when the slot *generates* a carry (sum >= 10),
  another when it does not.

One head scores `(c1_i + bq) * c1_j + slope * (j - i)`. The spike makes propagate slots
unattractive as keys, and the positional term breaks ties by recency, so slot `i`
attends to **the nearest earlier slot that is not a propagate slot** and reads whether it
generated a carry. That is carry lookahead, and it is why long 999...9 chains cost no
depth.

The same head is applied under two masks sharing all projections: strictly causal
(`j < i`) gives the carry *into* a slot, causal-with-self (`j <= i`) the carry *out of*
it. Then `y = z + wo1 * c_in + wo2 * c_out` performs the mod-10 wrap in two parameters,
and the digit is read out weightlessly against the same embedding table,
`logit_d = 2*y*U[d] - U[d]^2` (argmax = nearest centre).

`evidence.py` shows the attention is genuinely input-dependent: on `55555555555555 +
44444444444444` (every digit sum 9) all 15 slots skip back to the sentinel, on ordinary
inputs each attends to its predecessor; 14/16 rows change argmax between the two.
Replacing the learned pattern with a fixed previous-slot pattern drops accuracy from
2000/2000 to 493/2000.

## What made training work

1. **Propagate-rate curriculum** (the decisive one). Carry lookahead is a chicken-and-egg
   basin: with no `c1` notch the carry-in read is useless, so `wo1 -> 0`, so nothing
   pressures the notch to form. Training carry-free first, then ramping the per-slot
   propagate rate 0 -> pmax, then the full stress mix, walks around it. Loss 0.9 -> 0.001.
2. **Untied output centres `V` during training**, folded back onto `U` afterwards.
   Prevents the digit code from collapsing early.
3. **Dead-unit resurrection** — ReLU units that are always-on or always-off get their
   threshold re-placed inside the observed `z` range and their Adam state zeroed.
4. Ensemble training (M models along a leading dim) with evolutionary restarts.

## Getting from 36 to 21

Each step is an exact reparameterisation followed by a short fine-tune, and the graded
path was re-verified at every size before moving on.

| params | change |
|---|---|
| 36 | trained model, untied readout centres |
| 26 | `tie_analytic` — fold `V` onto `U`. Needs the *offset*, not just the scale: the tied optimum forces the code intercept to `B = -(wo1+wo2)*v0 = 2b - b'`, i.e. `U := V + q` with `q` the intercept of `z_U = p*z_V + q`, plus the matching `(p, q)` rescale of the feature layer, which keeps every ReLU activation bit-identical. Getting only the scale right zeroes accuracy. |
| 23 | `fold_unit_scale` — `relu(w z + b) v == abs(w) * relu(sign(w) z + b/abs(w)) * v`, so the input scales move into `W2a0`/`W2a1` and only the signs remain. Function-preserving: 40/40 cells stayed at 1.0 with no retraining. |
| 22 | `drop_value_unit` — with slopes at +-1, two entries already fix both the slope and the level of `c2` on the ranges that are read. The third only changes `c2` at sum == 9, and a sum-9 slot is a propagate slot, which the attention never selects as a key, so its value is never read. |
| 21 | fix the ALiBi-style positional decay to the constant 4.0 (ALiBi's slopes are not learned either) and let the attention's sharpness be learned through the scale of `W2a0`. |

## Why 21 is the floor for this circuit

Every remaining parameter is load-bearing:

* `U` (10) — the learned digit code. Left as a free table on purpose. Parameterising it
  as `alpha*d + beta` would reach 13, but linearity of the code in the digit *value* is
  the arithmetic content the model is supposed to learn, not something to hand it.
  Same reasoning ruled out binary/base-change tokenisation.
* `b1a` (3) — `c1` must vanish on both plateaus and spike between them; two units force
  the degenerate solution `w = 0`. Three thresholds are necessary and sufficient.
* `W2a0` (3) — two constraints (zero slope and zero level on the plateaus) plus the
  spike scale.
* `W2a1p` (2) — one constraint (flat plateau) plus the level.
* `bq` (1) — without it a non-propagate query has `c1_i ~ 0`, every key ties, and
  attention degenerates to pure recency. Dropping it would also cost the query's role in
  the attention, which is not worth one parameter.
* `wo` (2) — the ratio `wo2/wo1 ~ -10` is the base-10 wrap. It must be learned, not set.

## Layout

`lab/` — `core.py` (model + data), `train.py`, `fold.py` (the exact folds), `export.py`
(bakes a cell into `submission.py`), `verify.py`, `diag.py`, `gather.py`, checkpoints,
logs. `margin.py` and `evidence.py` are at the top level. Nothing was written outside
`/workspace`.
