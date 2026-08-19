# Minimal 14-digit addition transformer

`submission.py` is generated: it contains the model class (inlined verbatim from
`model_def.py`), the learned weights, `build_model()` and `add()`. Nothing else.
All training lives in `train.py` / `data.py`.

| file | role |
| --- | --- |
| `model_def.py` | architecture + inference path (the only code inlined into the submission) |
| `data.py` | operand sampling and ground-truth sum digits (training only) |
| `train.py` | training loop, evaluation, and the submission emitter |
| `emit.py` | re-emit `submission.py` from a saved run in `runs/` |
| `analyze.py` | accuracy breakdown + evidence the attention is input-dependent |
| `verify.py` | grader-style check: imports only `submission.py`, tests `add()` |
| `runs/`, `logs/` | checkpoints, metrics and training logs for every sweep run |

## The idea

Addition is a local digit-sum plus a *non-local* carry. Written out, the carry
into digit position `i` is decided by a single lookup:

> scan down from `i-1` to the first position `j` whose digit-sum `a_j+b_j` is
> not 9 (positions with digit-sum 9 merely *propagate* whatever arrives).
> The carry into `i` is 1 exactly when that position *generated* one, i.e.
> `a_j+b_j >= 10`.

"Find the nearest earlier position matching a content predicate, and read a
value off it" is precisely what one attention head does. So the whole task fits
in **one transformer block**, and no autoregressive decoding is needed: all 15
output digits are produced by a single forward pass.

## Architecture (154 parameters)

Sequence: a BOS slot followed by 15 digit positions, least-significant first
(operands are zero-padded to 15 so the carry-out has a home).

```
slot i     ->  residual dim 0:  U[a_i] + U[b_i]   (learned per-digit scalar code)
               residual dim 1:  transpose-tied pair table row for (a_i, b_i)
               residual dim 2:  zero -- scratch space for the head to write into
attention  ->  1 head, d_head = 1, strictly causal (j < i)
               score(i,j) = (q_i + bq)(k_j + bk) + slope * (j - i)
MLP        ->  ReLU, d_ff = 4
readout    ->  linear to 10 digit logits, per position
```

**d_model = 3, d_head = 1, d_ff = 4, 1 layer, 1 head, no LayerNorm.**

| block | parameters |
| --- | --- |
| `digit_emb` (10) + `pair_emb` (55x1) + `bos` (3) | 68 |
| `wq`, `wk`, `wv` (3x1 each), `bq`, `bk` (1 each), `wo` (1x3), `slope` | 15 |
| `w1` (3x4), `b1` (4), `w2` (4x3), `b2` (3) | 31 |
| `wout` (3x10), `bout` (10) | 40 |
| **total** | **154** |

- **Hybrid embedding.** Residual dim 0 gets a *compositional* code `U[a]+U[b]`
  built from ten learned scalars, so the model has to discover a digit code
  whose sum is informative. Dim 1 is a pair table — a lookup, where the
  non-linear carry predicate has to live. Dim 2 carries no token signal at all;
  the head writes the retrieved carry into it (the learned `wo` is
  `[0.125, 0.067, -2.535]`, i.e. almost entirely into that spare dim).
  A pure pair lookup instead costs 55 more parameters.
- **Transpose weight-tying.** `(a,b)` and `(b,a)` share a row, so the 100 pairs
  need 55. That is a symmetry of the task (`a+b == b+a`) expressed as weight
  sharing; the row contents are still learned, and the index map is an integer
  buffer, not a parameter. No arithmetic on the digits happens outside the
  model: `add()` only splits the operands into decimal digits and reassembles
  the digits the model predicts.
- **Query/key biases.** One scalar each (`d_head = 1`). They let a position ask a
  nearly content-free question ("who is the nearest non-propagate slot?"), which
  is what the carry rule actually needs. Without them, runs at this size collapse
  into diffuse attention; adding them moved the frontier from 364 to 209
  parameters in one step.
- **Recency bias.** One learned scalar `slope` multiplies `(j - i)` in the
  attention logits, so nearer positions win ties. *Which* position wins is
  decided by the content term.

## Does the attention do real work?

`analyze.py`, over 512 random inputs, at the most-significant position:

| model | mean max attention weight | attended index varies over | entropy |
| --- | --- | --- | --- |
| 154-param submission | 0.994 | 15 distinct positions | 0.013 |
| 195-param predecessor | 0.970 | 14 distinct positions | 0.091 |
| a failed 364-param run | 0.50 | 1 position | 1.18 |

The head is sharp and points at a *different* source position depending on the
digits it is given. The failed run collapsed onto a fixed, near-uniform pattern,
and its accuracy degraded exactly where that shortcut breaks (92% on inputs with
propagate-chains of 7+, versus 100% for the shipped model).

Reading the shipped weights out directly shows the mechanism *is* the carry
rule, not a correlational shortcut:

- the learned digit code `U[k]` is **monotone and near-linear in `k`**
  (-5.414, -3.990, -2.278, -0.830, 0.647, 2.031, 3.678, 5.155, 6.602, 7.314;
  successive differences 1.45 +/- 0.15), so residual dim 0 of a slot is a
  faithful copy of `a_i + b_i`;
- the single pair channel has **collapsed onto the digit sum** — rows sharing a
  value of `a+b` agree to within 0.02 for most sums — and `s = 9` is an extreme
  outlier: `-20.45`, against `[-0.59, +1.07]` for all eighteen other sums. That
  is the propagate predicate, learned;
- the **key** for `s = 9` is `+74.15`; for every other sum it lies in
  `[-3.49, +1.95]`. The query is nearly constant (mean -3.035, sd 0.055), so a
  propagate slot is penalised by about 228 logits. Against a learned recency
  slope of 15.321 logits per position, a non-propagate slot still outbids a
  propagate slot from ~15 positions away — exactly the reach a 15-digit operand
  needs, and no more;
- the **value** read out is a clean binary generate signal: `+2.8 … +5.2` for
  `s <= 8` against `-7.4 … -12.1` for `s >= 10`, i.e. the test `a_j + b_j >= 10`;
- the BOS slot's key (`-2.67`) sits in the ordinary non-propagate band and its
  value (`+3.83`) in the "no carry" band — the default that position 1, which
  can see nothing else, falls back on.

## Training

Fresh operand pairs every step (so every evaluation pair is unseen), AdamW +
one-cycle LR (`lr = 4e-3`, batch 2048), cross-entropy over the 15 output digits.
`data.py` mixes four regimes: uniform over the full range, random operand
*lengths* (small operands), digits skewed towards 9/0, and deliberately long
propagate chains (`b_i = 9 - a_i` at a per-sample rate). The last one is what
forces the head to learn the real rule instead of a short-range heuristic.

At this size training is a seed lottery: identical configurations either find
the mechanism and drop to exactly zero loss, or plateau near loss 0.5 with
diffuse attention. The shipped 154-parameter configuration converged on 1 of
7 seeds. Every seed, won or lost, is kept in `runs/`/`logs/`.

## Results

Exact-match accuracy (whole 15-digit answer correct) on freshly sampled,
never-trained-on pairs — 4,000,000 samples per column for the shipped model,
2,000,000 for the rest:

| params | config | uniform in [0, 10^14) | stress mixture |
| --- | --- | --- | --- |
| **154** | **d=3, dh=1, ff=4, 1-channel pair table (shipped)** | **1.000000** | **1.000000** |
| 161 | d=3, dh=1, ff=5, 1-channel pair table | 1.000000 | 0.999990 |
| 195 | d=3, dh=1, ff=2, 2-channel pair table | 1.000000 | 1.000000 |
| 209 | d=3, dh=1, ff=4, 2-channel pair table | 1.000000 | 1.000000 |
| 338 | d=3, dh=1, ff=16, full pair embedding | 1.000000 | 1.000000 |
| 441 | d=4, dh=1, ff=16 | 1.000000 | 1.000000 |
| 707 | d=4, dh=2, ff=24, untied | 1.000000 | 1.000000 |

`verify.py` additionally drives the submission the way a grader would (import
`submission.py`, call `add()` one pair at a time): **60,053/60,053 correct**,
including every all-9s carry chain from `9+1` to `99999999999999+1`, with zero
digit errors at every one of the 15 output positions and exact accuracy 1.0000
at every propagate-chain length from 0 to 7+.

The "stress" column is far harder than uniform sampling — it is saturated with
maximal carry chains, which random operands essentially never contain. The bar
is 99% and the shipped model makes no errors on either.

### Where the floor is

For `d_model = 3` the count is `126 + 7*d_ff` with a 1-channel pair table and
`181 + 7*d_ff` with two channels. Sweeps over `d_model` in {2,3,4,6,8}, `d_head`
in {1,2}, `d_ff` in {1..32}, tied/untied and hybrid/pair/compositional
embeddings, and many seeds, are all in `runs/`/`logs/`. What sets the floor:

- **`d_model = 2` never learns the task** (F: 323 params, 0.498; L: 261, 0.375;
  three further seeds all failed). Three dimensions is the minimum: the digit
  sum, the propagate predicate, and somewhere for the head to deposit the carry.
- **A purely compositional embedding fails.** `U[a]+U[b]` with a token-wise
  featuriser MLP deriving the carry features (220–319 params) reached only
  0.0009 / 0.062 / 0.009 / 0.947 across four configurations. The propagate
  predicate `a+b == 9` is not linearly recoverable from a sum code, and the tiny
  featuriser could not manufacture it. The hybrid embedding above is the
  compromise that does train: a sum code where a sum code suffices, table lookup
  where it does not.
- **Dropping the 10-parameter output bias fails** (199–206 params, three seeds,
  all stuck at loss 0.08–0.59), even though the ten digit classes are
  near-equiprobable.
- **`d_ff` below 4 fails once the pair table is narrowed to one channel.**
  `d_ff = 3` (147 params, three seeds) tops out at 0.9826 uniform / 0.891
  stress and `d_ff = 2` (140 params, two seeds) at 0.9414 / 0.846 — under the
  bar, and both plateau rather than converge. With two pair channels `d_ff = 2`
  still trains (195 params, two seeds) but `d_ff = 1` does not (188 params, four
  seeds, 0.004–0.25 uniform): two ReLU units are the minimum for folding
  `digit_sum + carry` around the mod-10 wrap, and with one pair channel the MLP
  additionally has to clean up a noisier carry signal.
- **Replacing the 55-row table with a featuriser fails** (`--feat`, 112–148
  params: dim 1 becomes a small scalar-to-scalar ReLU network applied to the
  digit code, so nothing is looked up at all). Eleven seeds across four widths;
  the best plateaued at loss 0.14 and 0.41 batch-exact. Manufacturing a
  one-level-wide spike at `a+b == 9` inside a near-linear code is a much harder
  optimisation problem than putting 55 numbers in a table, and it is where the
  55 parameters are actually going. Spreading the featuriser's ReLU kinks across
  the code range at initialisation helped a lot (loss 1.3 -> 0.14) but not
  enough.
