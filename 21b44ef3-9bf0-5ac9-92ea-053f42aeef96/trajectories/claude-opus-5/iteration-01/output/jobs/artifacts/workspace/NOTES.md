# Minimal 8-digit addition transformer — working notes

## Files

| file | role |
|---|---|
| `submission.py` | **the graded artifact**. Generated, never hand-edited. Model definition + trained weights + `build_model()` / `add()`. |
| `model_src.py` | single source of truth for the architecture. Its text is inlined verbatim into `submission.py`. |
| `data.py` | operand sampling, carry-chain curriculum, and the deterministic train/held-out split. |
| `train.py` | training loop. Never imported by `submission.py`. |
| `build_submission.py` | checkpoint → `submission.py`. |
| `verify.py` | audits the built `submission.py`: accuracy, independent parameter count, attention ablations. |
| `inspect_ckpt.py` | reads the learned circuit out of a checkpoint (attention entropy, carry read-out, embedding geometry). |
| `launch_wave.sh`, `summary.sh` | run a batch of configurations; one-line-per-run results table. |
| `runs/` | logs + checkpoints for every configuration tried. |

## Task framing

Operands are 8-digit (`[10_000_000, 99_999_999]`), so the sum has 8 or 9 digits.
The model sees a length-10 sequence, digits least-significant-first:

```
pos:      0        1    2    3    4    5    6    7    8        9
a:        0       a0   a1   a2   a3   a4   a5   a6   a7        0
b:        0       b0   b1   b2   b3   b4   b5   b6   b7        0
predict:  -       s0   s1   s2   s3   s4   s5   s6   s7       s8
```

Position 0 is a causal attention sink; position 9 is where the final carry
surfaces. Because both pad places are genuine zero digits, one uniform rule
covers every position: `out_i = (a_i + b_i + carry_i) % 10`. The two operand
digits at a place share one embedding table and are added, mirroring the
commutativity of addition. Reading is a single forward pass — all 9 output
digits come out at once, no autoregressive loop.

## The circuit the model has to find

Carry propagation is the only non-local part. With `s_j = a_j + b_j`:

* place `j` **generates** a carry if `s_j >= 10`
* place `j` is **carry-transparent** if `s_j == 9` (it passes a carry through)

so `carry_i = 1` exactly when the *nearest place below i that is not
transparent* generates. That is a content-addressed look-up, which is what the
attention layer is for: query at place `i`, keys score "am I non-transparent",
with the positional term breaking ties towards the nearest place.

This dictates the layer layout. Attention scores are bilinear in the residual
stream, so before the attention runs, the stream at place `j` is
`E[a_j] + E[b_j] + pos_j` and any score is *additively separable* in `a_j` and
`b_j`. No additively separable function of `(a_j, b_j)` can detect `a_j+b_j==9`
(only linear functions of `a+b` are expressible that way), so a nonlinearity
must precede the carry attention. Hence:

```
embed → MLP (derive generate / transparent flags) → attention (carry look-up) → MLP (mod-10) → tied read-out
```

Run `W4` is the control for this argument: the same model with the first MLP
removed (a single attn+MLP block) plateaus around 36% instead of ~99%.

## The trap: 99.9% without learning to add

There is a second, *wrong* way to compute the carry, and gradient descent finds
it first. Note

```
carry_i = 1   ⟺   Σ_{j<i} 10^j · s_j  ≥  10^i
```

and softmax attention with a purely positional key `k_j = c·j` puts weight
`∝ e^{c j}` on place `j` — a geometric ramp. With `v_j = s_j` (a plain linear
read-out of the residual) the attention output is a geometric-weighted mean of
the place sums, and thresholding it approximates exactly that inequality.
No content in the key, no look-up: **a fixed pattern**.

This solution is numerically doomed — separating `Σ = 10^i − 1` from `Σ = 10^i`
needs ~`10^-8` relative precision by the top digit — but it is *very* good on
uniform operands, because a random pair almost never contains a long carry
chain. It is what the earlier 304-parameter model (`P3`) actually learned:

```
P3 layer-1 mean attention          measured input-dependence: std = 0.034
  i=8   0.00  0.00  0.01  0.02  0.05  0.10  0.24  0.58        (ratio ≈ 0.41/place)
  i=9   0.00  0.00  0.01  0.01  0.02  0.05  0.10  0.24  0.57
```

99.96% held out, and `12345678 + 87654321 → 109999999`. Freezing its attention
to a fixed average pattern still scored 79.8%. That is the "fixed pattern
dressed up as attention" failure mode, so it was not shipped.

Two changes kill the shortcut:

* **maximal carry chains in the batch** (`data.chain_digits`) — one place
  generates, every place above it sums to exactly 9. The geometric blur scores
  near zero on these, so it stops being a local optimum. Uniform sampling
  essentially never produces them.
* **worst-case checkpoint selection** — the saved checkpoint maximises
  `min(uniform accuracy, chain accuracy)`, not uniform accuracy.

An optional attention-entropy penalty (training-time only, warmed up so it does
not just freeze the initial pattern) pushes the same direction.

## Auditing: what actually distinguishes the two

`audit.py` scores every checkpoint in `runs/` against the same gates
`verify.py` applies to the built submission. Getting the gates right took two
corrections, both worth recording because the obvious version of each is
misleading.

**The frozen-pattern ablation has to be scored where a fixed pattern cannot
work.** Replacing the computed attention with the mean pattern it produces --
every weight and the whole value path left intact -- is the natural test for
"is this really a fixed pattern". But its result depends entirely on what you
score it on. On uniform operands the mean pattern is close to a one-place
shift, and shift + MLP recovers the purely *local* carry rule ("carry iff the
place below generates"), which by itself gets about half of all 9-digit sums
exactly right. So the shipped 447-parameter model, whose carry head is a
verified hard look-up, still scores 52.9% under the ablation -- a fact about
the task, not about the model. The maximal carry chains are the discriminating
set: resolving them *requires* reading a place selected by content, so any
input-independent pattern collapses there. On chains the same ablation gives
2.1% for a genuine circuit and 65-98% for the geometric-ramp shortcut.

**The look-up metric must not demand one particular implementation.** Two
refinements were needed. First, only the rows where the true carry is 1 pin
down where the head has to look: when the nearest non-transparent place is
*blocking*, every non-generating place (the sink included) is an equally
correct target, so scoring the argmax on those rows measures nothing.
Second, "argmax == nearest non-transparent place" is only one way to be right.
The 295-parameter model attends to *a* generating place on 100% of the rows
that need a look-up, but to the nearest one only 55% of the time -- and its
carry read-out is still perfect. It found an equivalent content-addressed rule,
not the textbook one. The gate is therefore "the argmax lands on a place that
generates", which no input-independent pattern can pass.

The four measurements that separate real from fake, on maximal chains:

| | 447p (L9) | 295p (A2) | ramp shortcut (P3/B1/A3) |
|---|---|---|---|
| carry-head row entropy | 0.010 | 0.105 | 0.61 – 1.16 |
| argmax on a generating place | 100.0% | 100.0% | 41 – 42% |
| carry read-out from attention mass | 100.0% | 100.0% | 57 – 59% |
| chains after freezing the pattern | 2.1% | 26.8% | 65 – 98% |

There is no threshold tuning here: the gap is 100% vs 42%.

## What the crisp solution looks like

The 447-parameter model:

```
attention layer 1 (the carry head)
  mean row entropy                                    0.010 nats   (uniform = 2.2)
  argmax == nearest non-transparent place below i     99.9 %
  (attention mass on generating places > .5) == carry 100.0 %
embedding cosine similarity by digit distance
  d=0 +1.000  d=1 +0.788  d=2 +0.346  d=3 -0.134  d=4 -0.496  d=5 -0.649
```

i.e. a hard content look-up that reads the carry out perfectly, on top of a
learned clock code (`cos(2*pi*d/10)` = 1, .809, .309, -.309, -.809, -1) that
makes `E[a]+E[b]` a usable representation of the place sum. Its layer 0 is a
learned one-place shift feeding the flag MLP; freezing *that* layer alone costs
almost nothing (100% -> 92.7%), which is the expected signature of a positional
shift and is why the per-layer ablation is reported separately.

## A coverage hole in the chain curriculum

`chain_digits` clamped the transparent places to `a_j <= 8`, so `9+0` and `0+9`
never appeared inside a carry chain -- in the curriculum *or* in the stress
set. The clamp is only needed at the top place, where `b_j = 9 - a_j` has to
stay non-zero to keep both operands a full 8 digits wide. The 295-parameter
model was trained under the buggy generator and still misses a few chains that
run through a `9+0` place (`19999999 + 10000001` is one). Fixed in `data.py`
and `verify.chain_cases`; every run from wave K onwards trains on the full
distribution.

## Parameter-count levers that worked

* **Tied read-out** — logits are `E @ h`, reusing the input embedding. Saves
  `10*d` and, surprisingly, *trains better* than an untied head (N2 79% vs N3
  3% at matched budget): the clock-like code that makes `E[a]+E[b]` useful is
  the same code that makes `E[k]·h` a good classifier.
* **No BOS row** — the sink and the carry-out slot are fed digit `0`, which is
  already non-transparent and non-generating. Saves a vocabulary row and makes
  the positional rule uniform.
* **Rank-1 positional encoding** — `pos_i = scale_i * dir`, i.e. `SEQ + d`
  params instead of `SEQ * d`. The task is translation-invariant across digit
  places; all attention needs from position is a monotone ordering. Fully
  learned, just factored. Also converged *faster* than the full table (W1 vs W2).
* **Narrow attention** — the carry head's score only has to express
  "non-transparent" plus a positional ramp, and its value only has to carry one
  bit ("generates"). `k_j = c·j + m·[s_j ≠ 9]` with `m > 9c` already puts the
  argmax on the right place, so in principle `d_head = 1` suffices; wave A
  tests how far that holds in practice.

Two tiny additions that cost ~4 params and buy a lot of optimisation speed:
`q_bias` (an input-independent query component, so "prefer non-transparent
places" doesn't have to be routed through the residual stream) and
`learn_scale` (a multiplicative logit temperature, so a head can sharpen toward
a hard argmax in far fewer steps than growing `wq`/`wk` elementwise).

## Training recipe

* Batch mixture: 40% uniform (the graded distribution), 35% drawn with
  per-place probability `prop_p` of forcing `a_j + b_j == 9`, 25% maximal carry
  chains.
* AdamW, OneCycle, **lr 1e-2**. This mattered more than anything else: at
  lr 3e-3 the model settles into a soft interpolation (~99.6%, blurry
  attention); at lr 1e-2 it finds the crisp look-up.
* Step count matters more than sample count. Batch 4096 for 30k steps (123M
  samples) underperformed batch 1024 for 60k steps (61M samples). The crisp
  solutions have needed ~100k steps.
* The model is small enough that a step is CUDA-launch-bound, and the sampler
  was issuing more kernels than the forward and backward passes together.
  `SamplePool` generates a 2M-pair pool per mixture component and refreshes it
  every 400 steps; only a few percent of a pool is consumed before it is
  replaced, and it is filled by the same holdout-excluding sampler.

## Held-out protocol

A deterministic hash of `(a, b)` assigns every operand pair to one of 128
buckets. Bucket 0 is **never** trained on — training batches resample any pair
that lands there — and all reported accuracy is measured on bucket 0 only. So
"unseen" is guaranteed by construction rather than by the low collision
probability of random sampling. `verify.py` additionally measures unrestricted
uniform pairs, to rule out the held-out bucket being distributionally special.

## Status

Shipped: `runs/A2.pt`, **295 parameters** (d_model=5, `attn(1x1)+mlp(4)` ->
`attn(1x1)+mlp(12)`), 99.969% on 200k held-out pairs, 99.956% on maximal carry
chains, and a carry head that passes every crispness gate above. It replaced
the 447-parameter `runs/L9.pt` (100.000%, also a clean ACCEPT, kept as
`cand/L9.py`) because the task asks for the smallest model above 99%, not the
most accurate one. Known blemish: a handful of chains that run through a `9+0`
transparent place -- the coverage hole described above -- still fail.

A candidate replaces the shipped file only if it is both smaller *and* returns
`=> ACCEPT` from `verify.py`.
