# 12-parameter transformer that adds two 8-digit integers

`submission.py` holds a single-block transformer with **12 trained parameters**. It is
exact on every pair of 1–9 digit operands, verified by exhaustive proof rather than by
sampling.

```
parameters            12
held-out accuracy     1.0000   (20,000 random 8-digit pairs, 0 errors)
whole-domain proof    PASSED   (9,693,835 checks, widths 1..9, 0 wrong)
worst read-out margin 0.4895 residual units out of a possible 0.5
imports               torch, torch.nn
```

## What the 12 parameters are

| tensor | count | what it is |
|---|---|---|
| `code[1..9]` | 9 | the residual value of each digit; the same table is tied as the read-out prototypes. Digit 0 is pinned at the origin (translation gauge). |
| `knee[0..1]` | 2 | the two clamp-bank thresholds |
| `fold` | 1 | the mod-10 write-back scale |

Everything else is a fixed architectural constant in a buffer, and each one is fixed for
a reason that is *not* "we fitted it and then hid it":

- `bank_w = 8.0`, `key_w = (-400, +400)`, `val_w = (0, 1)` — the bank output is a hard
  clamp, so multiplying its input slope by α and dividing the knees by α is a gauge
  freedom of the function. Fixing the slope removes the redundancy; the knees absorb it.
  Similarly the gate saturates to exactly 0/1, so the key and value read-offs only set a
  scale, not a shape.
- `carry_w = 1.0` — the residual-stream scale gauge. The model is invariant under
  `(code, knee⁻¹, fold, carry_w) → (s·code, …)`, so one of them must be pinned; pinning
  `carry_w` lets `code` carry the learned scale.
- `lam = -12.0` — the relative-distance bias slope. Any value large enough to make the
  recency ordering strict gives the identical decoded output.
- `ls` (read-out temperature) — argmax-invariant, so it is not in the model at all.

`build_ship.py` asserts every one of these is *exactly* at its architectural value in the
trained checkpoint before it will emit a file, so a fitted number cannot leak into a
buffer. The parameter count is what `sum(p.numel() for p in model.parameters())` reports.

## What the model learned

The nine `code` values were trained with no constraint tying them together. They came out
as an arithmetic ramp:

```
code = [0, 0.99993, 1.99409, 2.99055, 3.98817, 4.98579, 5.98341, 6.97988, 7.97403, 8.97397]
best-fit line  code[d] = 0.996906·d + 0.000903,  max deviation 2.1e-03
fold = -9.97339,   fold / mean_step = -10.0023
```

Base-ten place value is the *result* of training here, not an input to it. Nothing in the
loss mentions the number ten; the ramp and the factor of ten in `fold` are what the
optimiser found. The two knees landed at residual thresholds 8.195 and 9.162 — i.e.
between `a+b = 8` and `9`, and between `9` and `10` — which is the three-way split the
mechanism below needs.

The mechanism is **carry lookahead**, discovered rather than prescribed. A place embeds as
`code[a] + code[b]` ∝ `a+b`. The clamp bank sorts places into absorb (`a+b ≤ 8`),
transparent (`a+b == 9`) and generate (`a+b ≥ 10`), and writes a key that is notched far
down on exactly the transparent places, making them unattendable. With the recency bias,
each query therefore lands on the **nearest earlier non-transparent place**, whose value
says whether it generated a carry. The strictly-causal head returns the carry *into* a
place, the inclusively-causal head the carry *out of* it, and `fold` subtracts ten code
steps when the place carries.

## Evidence the attention does real work

`python ablate.py` — the content-dependent part of the attention score is the only thing
removed; positions, masks and recency bias are untouched.

```
            regime   shipped    no key  mean key
    uniform digits    1.0000    0.7112    0.1044
       carry-heavy    1.0000    0.4946    0.3739
```

Measuring on uniform digits alone would understate this. A transparent place is rare under
uniform sampling, so "attend to the previous position" usually coincides with "attend to
the nearest earlier non-transparent position" and the ablation looks mild for the wrong
reason. On carry-heavy inputs the shipped model stays at 1.0 while pure-recency attention
falls to 0.49.

Which position each query lands on, with positions held fixed and only digits changing:

```
88888889 + 11111111   place 0 generates, places 1-7 all transparent
  query pos 1..9 attends to [0, 1, 1, 1, 1, 1, 1, 1, 1]     <- jumps up to 8 back
88888888 + 11111111   every place transparent, nothing generates
  query pos 1..9 attends to [0, 0, 0, 0, 0, 0, 0, 0, 0]     <- one digit changed, all re-routed
11118111 + 11111111   transparent at place 3
  query pos 1..9 attends to [0, 1, 2, 3, 3, 5, 6, 7, 8]     <- query 4 skips over place 3
```

## How it was produced

Training lives entirely outside `submission.py` (`train_parent.py`, `reduce.py`,
`stage3.py`, on top of `arch.py`/`data.py`/`lab.py`). The graded file contains the module
and its inference path only. Four stages:

1. **`train_parent.py`** — cold-start 1024 independent members simultaneously (stacked on a
   leading ensemble axis, per-member gradient clipping; Adam is elementwise so members stay
   independent). A wide pose: 2-d residual, 2 clamp units, 38 free values. 15/1024 members
   reached 1.0 held-out in ~108 s. This is a seed lottery — the mechanism is found by
   search, not by initialisation.
2. **`reduce.py`** — project the 2-d residual onto its principal direction (SVD) to get a
   scalar stream, compensating the bank bias for the discarded component. The projection
   alone kept all 15 exact members at 1.0; retraining 512 perturbed copies gave 256/512.
3. **`stage3.py`** — canonicalize the gauge (8 discrete poses from clamp-complement and
   unit-reorder symmetries; pick the one matching the absorb/transparent/generate gate
   pattern), then rewrite each constant to its architectural value by an exact
   transformation. Every step preserved 64/64 exact members. Order matters and is
   enforced: `canonicalize → value_norm → translate → rescale → set_bank_slope → sharpen`.
4. **polish** — retrain only the 12 free parameters *in the exact shipped form*, against a
   read-out-margin objective plus a saturation term. 503/512 members exact; best worst-case
   margin 0.4895.

`build_ship.py` emits the file and asserts the emitted module reproduces the training-time
forward bit-exactly (max logit difference 0.000e+00).

## Why 12 and not fewer

- **2 knees is a floor, not a choice.** The key must be *low exactly on the middle class*
  (transparent) and high on both absorb and generate. That is non-monotone in `a+b`, and a
  single clamp unit can only produce a monotone threshold. Two units is the minimum that
  can isolate a middle class.
- **1 fold** is the mod-10 write-back; there is nothing to merge it with.
- **9 code values could be collapsed to 1** by parameterising `code[d] = d·step`, giving a
  4-parameter model. I did not do this, and I think it would be the wrong answer to this
  task. The entire evidence that the model *learned* to add is that nine unconstrained
  numbers came out as an arithmetic ramp and `fold` came out at −10 steps. Pinning them to
  a ramp installs base-ten place value by hand and hands the model the structure it is
  supposed to discover. Per the task: where the code sits does not change what it is.

So 12 is where I stopped: the smallest count I can reach without writing the answer into
the architecture.

## The whole-domain certificate

`python certify.py` proves exactness on every operand pair without sampling, in float64:

1. Over all 100 digit pairs, check the clamp bank is **fully saturated** (slack 1.5000, so
   every unit is ≥1.5 away from leaving 0/1) and splits into (0,0) / (1,0) / (1,1) on
   absorb / transparent / generate. Once that holds, a place's key and value depend only on
   its carry class.
2. Enumerate all 3ⁿ carry-class patterns and evaluate both softmaxes exactly — this
   *accounts for* attention leakage rather than bounding it.
3. For every pattern, position and consistent digit pair, check the residual decodes to the
   correct answer digit.

This covers every pair because behaviour depends only on the class pattern and the
per-place digits. Result: 9,693,835 checks, 0 wrong, worst margin 0.489497 across widths
1–9 (the graded width, 8, is included). Separately, float32 and float64 decode identically
on 8192 pairs, so the proof transfers to the precision the graded model actually runs in.

The saturation slack is the load-bearing part. An earlier candidate scored 1.0 on sampling
but failed step 1 with slack −0.346: the margin objective has no gradient on a knee while
the gate is saturated, so the knees random-walked until one straddled the transparent
class. It would have shipped and probably scored fine, but it was fragile for a reason
sampling could not see. Adding the saturation term to both the loss and the member
selection fixed it.

## Limitations, honestly

- The certificate covers widths 1–9 (the graded width, 8, is inside it). The metadata
  advertises `max_digits: 30`, which is *tested* — 3000 random pairs plus the worst-case
  patterns (`8…89 + 1…1`, `9…9 + 9…9`, `9…9 + 1`) are exact at every width up to 30 — but
  not *proven*: enumerating 3³⁰ class patterns is not feasible. It does eventually break:
  at n = 40 two of the three hard patterns are wrong. That is the expected failure mode —
  attention leakage accumulates over a long transparent run — and it is well outside both
  the graded domain and the advertised limit.
- `add` assumes non-negative integers. Negative inputs are out of the stated domain and
  would index the code table with a negative digit.
- The member shipped is one of 503 exact members; the choice among them was by worst-case
  margin, which is the quantity the certificate bounds.
