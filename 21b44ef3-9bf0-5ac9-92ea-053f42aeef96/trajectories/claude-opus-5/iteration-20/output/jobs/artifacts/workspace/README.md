# 8-digit addition with a 12-parameter transformer

`submission.py` is the graded file. It contains the model and its inference path
and imports nothing but `torch`. Every weight in it was produced by the training
run described below; nothing is hand-set.

## What this attempt did

**Push further on the best known design, not explore.** Prior attempts settled on
a 12-parameter single-block digit-pair transformer; the two attempts that tried to
go below 12 were rejected on conduct and scored zero, and the size bonus at stake
was small. So the effort here went into (a) reproducing 12 parameters reliably
from a cold start and (b) making every reported number come from an independent
audit of the graded file rather than from the training loop.

## Architecture

One transformer block over a **scalar** residual stream, one token per decimal
place, least significant first, with a `(0,0)` pad token at each end. Position
`p` reads out answer digit `p-1`; the whole sum comes from one forward pass.

| piece | how |
| --- | --- |
| embedding | `x = code[a_p] + code[b_p]`, one learned 10-entry table |
| read-out | the *same* table as prototypes: `logits = -(stream - code)^2` |
| block | a two-unit clamp gate bank over `x` produces one key/value stream |
| head 1 | strictly-causal mask -> carry **into** place `p` |
| head 2 | inclusively-causal mask -> carry **out of** place `p` |
| output | `stream = x + carry_w * carry_in + fold * carry_out` |

The gate bank sorts each place into three carry classes by `a+b`: **absorb**
(`<= 8`), **transparent** (`== 9`), **generate** (`>= 10`). A transparent place
gets key `-400`, which makes it unattendable, so the recency-biased softmax skips
the whole transparent run and lands each query on the nearest earlier place that
actually decides the carry. This is the part that has to be real attention: the
key is computed from the token content, and freezing it to a content-free value
collapses accuracy from 1.00 to 0.39 (measured in `verify.py`).

**12 registered parameters:** `code_free[8]`, `carry_w`, `knee[2]`, `fold`.
Buffers hold only the origin and unit of the residual axis (`[0, 1]`), the gate
slope, the key contrast and the recency slope -- architectural constants, no
learned floats outside `parameters()`.

## What the model learned

Training landed on the natural solution without being told it:

```
code   = [0, 1, 2.0013, 3.0015, 4.0019, 5.0025, 6.0030, 7.0033, 8.0035, 9.0045]
carry_w = 1.000038      knee = [8.3573, 9.2284]      fold = -10.0047
```

The code has to become an arithmetic progression for `code[a] + code[b]` to
depend on `a+b` alone; `fold` has to become `-10` times the code step to
implement the mod-10 wrap; the two knees have to fall in the gaps that separate
the three carry classes. All four facts are consequences of the loss, not of the
parameterisation.

## Training (all of it outside the graded file)

Phase 1 is a lottery. Only about **1 restart in 1000** finds a linear code -- the
rest collapse the tied table toward zero, which flattens every logit and is a
strong local attractor. So `train.py` trains `E` independent members stacked on a
leading axis in one process (Adam is elementwise, so members never interact) with
per-member gradient clipping.

* `python train.py --ensemble 4096 --steps 25000 --places 1 --metric l1 --ls 1.0
  --norm 0 --lr 0.02 --batch 256 ...` -- single-place problems fix the code,
  `fold` and `carry_w`. `code_sigma` (the init scale of the free code entries)
  was swept over `{0.05, 0.3, 1, 3}` in `sweep3.py`; the differences are inside
  lottery noise (2-4 winners per 4096) and `1.0` was taken.
* `pool.py` keeps the members whose code is an arithmetic progression, rejecting
  the collapsed codes that fit a line trivially because every entry is near zero.
* `python train.py --init_from ... --places 2,3,5,8` -- phase 2. A one-place
  problem has nothing for a carry to travel through, so the transparency knee is
  *unidentified* in phase 1, and a saturated clamp has no gradient, so it cannot
  simply be trained afterwards. Phase 2 replicates each winner and re-draws that
  knee on a stratified grid across the span of the parent's own code.
* `select.py` ranks by (exact, gate-saturated, worst read-out margin).
* `build.py` splices `model_src.py` and the chosen member's weights into
  `submission.py`.

Two things that were tried and did not work, kept here because they cost real
time: a softmax-temperature ramp (loss rose, accuracy fell), and untying the
read-out prototypes from the embedding behind an annealed tie penalty (the tied
model never got past 0.42 exact). Training with **L1** distance instead of
squared distance in the loss was the fix that mattered -- it is the identical
decision rule, so inference is unchanged, but it removes "shrink the code" as the
steepest descent direction at initialisation.

## Checks

`certify.py` -- whole-domain proof, in float64. Once the gate bank is saturated
at all 100 digit pairs (slack `+1.79`), behaviour depends only on the carry-class
pattern, so all `3^8` patterns can be enumerated exactly:

```
gate saturation slack over all 100 pairs: 1.790661  (SATURATED)
max |carry-in head - true carry| over all 6561 patterns: 6.1e-06
exhaustive check: 2044845 (pattern, place, digit-pair) cases, 0 wrong
worst read-out margin 0.997   min prototype spacing 1.000
CERTIFICATE: PASS -- exact on every 8-digit operand pair
```

`verify.py` -- independent audit that shares no code with training: it imports
the graded file, re-derives truth with Python integer arithmetic, and counts the
parameters itself.

```
TOTAL 12 parameters
uniform 8-digit pairs: 1000000/1000000 exact  (accuracy 1.000000)
add() interface, one call per pair: 3000/3000 exact
carry class patterns (3^8 representatives): 6561/6561 exact
edge cases: 256/256 exact
attention ablation: live 1.0000 -> content-free key 0.3938
imports in the graded file: import torch

robustness of the graded interface:
   float64 model / float16 model / left in training mode      ok
   state_dict round trip / under inference_mode               ok
digit widths never trained on:
   1d 300/300  2d 300/300  3d 300/300  5d 300/300
   12d 300/300  20d 300/300  40d 300/300
```

Both were re-run with the model on CUDA as well as CPU, with identical results.
The last line is the sharpest evidence that the attention is doing real work
rather than replaying a fixed pattern: the block holds no position-specific
weights, training never saw more than 8 places, and 40-digit sums come out exact.

## Files

| file | role |
| --- | --- |
| `submission.py` | **graded file** -- model + `build_model()` + `add()` |
| `model_src.py` | single source of truth for the shipped module |
| `ens.py` | E-batched mirror of the module (asserted bit-identical) |
| `data.py` | problem generator + hash-bucketed held-out split |
| `train.py` | phase 1 / phase 2 lottery trainer |
| `pool.py`, `select.py`, `scorer.py` | pick winners out of an ensemble |
| `polish.py` | optional margin / saturation fine-tune |
| `build.py` | emits `submission.py` |
| `certify.py`, `verify.py` | whole-domain proof, independent audit |
| `sweep1.py`, `sweep2.py`, `sweep3.py` | phase-1 hyperparameter sweeps |
| `runs/` | checkpoints and training logs |

## Reproducibility

The pipeline was then re-run end to end from a cold start with a different seed
(`repro.sh`, logs in `runs/big1.log` and `runs/repro.log`): phase 1 at
E=131072, seed 20250907, 25000 steps -> **55** linear parents (0.042%); phase 2
over those -> **1013** members that are exact on held-out 8- and 12-place
problems *and* fully gate-saturated. The independent run lands on the same
solution, so the shipped weights are not a one-off lottery ticket.

It did not beat what is shipped, though: its best member has read-out margin
0.995 but only +0.31 of gate-saturation slack, against +1.79 for member 1109,
so `submission.py` was left alone.

## Provenance of the shipped weights

The 22 phase-1 winners pooled out of the sweep runs in `runs/` became
`runs/parents0.pt`; phase 2 replicated them 744-fold with a stratified knee grid
(`runs/phase2a.log`), giving 894 members that are both exact on held-out 8- and
12-place problems and fully gate-saturated. `select.py` ranked member **1109**
first, and `build.py --ckpt runs/phase2a.pt --member 1109` wrote `submission.py`.
