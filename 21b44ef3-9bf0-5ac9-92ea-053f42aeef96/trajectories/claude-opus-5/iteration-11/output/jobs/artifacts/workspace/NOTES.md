# Minimal 8-digit addition transformer — what is learned and what is a coordinate

## Files

| file | role |
| --- | --- |
| `adder.py` | the architecture, as a pure function of a parameter dict with a leading ensemble axis |
| `data.py` | on-GPU problem sampler + the 1-in-16 hash-bucketed held-out split |
| `train.py` | trains E independent members simultaneously (the "seed lottery") |
| `reduce.py` | rewrites a trained parent into the 11-parameter coordinates |
| `certify.py` | whole-domain correctness certificate + the derived `alpha` band |
| `pick.py` | ranks every member of every checkpoint by how well conditioned it is |
| `build.py` | emits `submission.py`: model class + weights as plain float literals |
| `verify.py` | audits the shipped file |
| `check_arch.py` | expressivity check only — hand-set weights, never shipped |

Nothing outside `/workspace` is written. `submission.py` imports only `torch`.

## The mechanism the training found

The residual stream is one scalar per position. A place token embeds as
`code[a] + code[b]` from a single learned 10-entry table, and that same table
supplies the read-out prototypes, so the model has to find a code in which
summing two digit embeddings lands near the embedding of their sum.

Carry propagation is the part attention does. Write `s = a_i + b_i`. A place
kills a carry (`s` small), passes one through unchanged (`s` at the transparent
value), or generates one (`s` large). So the carry into place `i` is: look back to
the nearest earlier place that is *not* transparent, and ask whether it generated.
That is a content-addressed lookup whose target depends on the digits — a fixed
distance pattern cannot do it, because the distance to the nearest non-transparent
place is data-dependent.

The bank builds both halves of that lookup from one learned threshold `theta`:

    upos = clamp(alpha * (x - theta), 0, 1)
    uneg = clamp(alpha * (theta - x), 0, 1)
    upos + uneg = min(1, alpha * |x - theta|)

`upos + uneg` is 1 everywhere except a notch to 0 at `x = theta`; used as the
attention key (times a large gain) it hides exactly the transparent places, so the
softmax lands on the nearest earlier non-transparent place. `upos` alone is the
value: 1 at generating places, 0 at killing places. Two heads share that one
key/value stream and differ only in their mask — strictly causal gives the carry
*in*, inclusively causal gives the carry *out*, and the carry out times `e2` is
what folds the digit sum back below ten.

## Parameter accounting (11)

Free, and all of them learned facts about base-10 addition:

| what | count |
| --- | --- |
| `code_free` — the digit code / read-out prototypes for digits 1..9 | 9 |
| `theta` — the transparent value, i.e. where the carry threshold sits | 1 |
| `e2` — the mod-10 fold weight | 1 |

Fixed (buffers). Each is a coordinate choice, a decision-rule invariance, or a
constant with a measured wide working band — none is a fitted value:

| what | why it is not a parameter |
| --- | --- |
| `code_pin` = code[0] = 0 | origin of the 1-D residual stream. With a residual bias `rb` the model is exactly translation-covariant (`code += t`, `rb -= t`, `theta += t` leaves every logit unchanged); the parent is trained with `rb` free and gauged so that the origin sits at digit 0. |
| `e1` = 1 | unit of the residual stream. The model is exactly scale-covariant (`code, rb, theta, e1, e2` scale by `g`, `alpha` by `1/g`, `ls` by `1/g^2`); the parent is gauged so the carry-in weight is the unit. |
| `alpha` | bank slope: sets the notch width. Its admissible band is *derived* from the weights — see below — and the model is trained at whatever value it will be run at. |
| `kw` = 2000 | key gain: only has to exceed `|lam| * sequence length` so the notch outranks the distance bias. One-sided; band measured. |
| `lam` = -8 | relative-distance bias: only has to be negative enough that the *nearest* eligible place wins the softmax. Band measured (-4 to -8 at the shipped weights). |

There is no read-out temperature in the shipped model at all: `argmax_d -(y-code_d)^2`
needs no scale, so the parent is trained with one free and it is dropped on the way
out rather than being shipped as a constant equal to 1.

## What `alpha` actually has to satisfy

`alpha` is *not* an independent dial, and an earlier version of this file was wrong
to call it a plain don't-care. The scale gauge was already spent on `e1 = 1`, so
`alpha` is locked to the code scale: rescaling it without rescaling the code is a
real change of function. What it has to do is separate two distances,

    d_tr = max |code[a]+code[b] - theta|  over  a+b == 9    (transparent pairs)
    d_nt = min |code[a]+code[b] - theta|  over  a+b != 9    (everything else)

Saturation needs `alpha * d_nt >= 1`; keeping the notch open needs
`alpha * d_tr <= 1 - |lam|*P/kw`. So the working band is

    [ 1/d_nt ,  (1 - |lam|*P/kw)/d_tr ]

which is non-empty exactly when the learned code is linear enough that `d_nt > d_tr`.
`certify.py` prints this band, and for the shipped member it comes out as
**[1.20, 9.40]** with `alpha` shipped at 3. That is a prediction, and `verify.py`
step 8 confirms it by measurement: `alpha` = 1.5, 2, 4 and 8 all give exactly
1.0000 held-out, while 1.0 and 30 fail. `d_nt` is pinned near one code step by the
`e1 = 1` gauge, so the lower end is always ~1; only the upper end moves, and it
moves as the learned code gets closer to an exact ramp.

Selecting on this rather than on accuracy is what `pick.py` does, and it changed the
answer. Accuracy at a single alpha is a weak criterion — an earlier candidate scored
1.0 everywhere in `verify.py` but had a band of only [1.72, 2.02], because
fine-tuning at one fixed alpha lets the code drift away from linearity with nothing
to penalise it. Re-ranking every member by band width instead put a *less*
fine-tuned member on top, with a band 7.8x wide instead of 1.17x. Widening the band
and sharpening the arithmetic are the same thing.

## The constants carry no task information

Direct evidence rather than an argument: the same trained parent was reduced to the
11-parameter coordinates at three unrelated choices of the whole constant triple —
`(alpha, kw, lam)` = `(2, 2000, -8)`, `(4, 6000, -6)` and `(1.4, 800, -4)` — and each
reaches 1.00000 held-out at 11 parameters (`runs/red_a4.pt`, `runs/red_a14.pt`, via
`altconst.sh`). Nothing about the task lives in those numbers.

The counts are gauge-invariant: `(code, rb)` carry 11 numbers with one translation
freedom and one scale freedom, so 9 essential code numbers remain, and no
rewriting can do better without pinning something that is actually learned.

Deliberately **not** done, because these would be arithmetic facts smuggled into
constants rather than gauges: parameterising the code as a ramp `code[d] = d*g`,
fixing `e2` at `-10`, fixing `theta` at 9, or tying `theta` to the code scale.
All of those are learned by gradient descent here.

## Why the bank is one threshold and not two

An earlier version of this architecture (my attempt 10) used two independent
clamp knees, one for `upos` and one for `uneg`, and cost 12 parameters. Forcing
them to coincide makes the bank a symmetric V feature — the standard
`|z| = relu(z) + relu(-z)` construction — with the *location* still learned. It is
a choice about the shape of the feature, like choosing `clamp` over `relu`; the
model still has to discover where the transparent value sits and to learn a code
that puts the neighbouring sums outside the notch. That is the whole reduction
from 12 to 11.

## Why 11 and not 10

Worth being explicit, because 11 is a floor I chose rather than one I hit by accident.
The raw numbers are 10 code entries + `theta` + `e1` + `e2` + `rb` = 13, less one
translation and one scale freedom = 11 essential. So no further *gauge* move exists;
going to 10 needs a structural claim, and every available one is the arithmetic itself:

* Requiring `code[a]+code[b] ~ code[a+b]` for `a+b <= 9` already forces the code to be a
  ramp `code[d] = d*g`. Parameterising it that way would collapse the table to one
  number — but "the digit embeddings are equally spaced and addition is translation
  along that line" *is* the fact the model is supposed to discover. The training does
  discover it: the learned table comes out at `code[d] ~ 1.06*d` with no ramp imposed.
* `theta = code[9]` holds in the solution and would save a parameter, but choosing
  index 9 by hand is choosing the base.
* `e2 = -10*code[1]` likewise, and fixing `e2 = -10` even more directly.

Each of those hands the model an answer instead of letting it learn one, so none is
taken. 11 is where this architecture stops without smuggling.

## Training

Cold, from random initialisation, no warm start and no ladder: E=256 independent
members share one data stream, batch 1024, AdamW lr 0.012 one-cycle, per-member
gradient clipping. Sequence length varies across the run (3, 5, 8, 11 places) so
the attention cannot fall back on a fixed offset. Reduction to the 11-parameter
coordinates is done with the exact rewrites above.

Provenance of the shipped weights, exactly:

    train.py  (cold, 18 free parameters)      -> runs/cold_v_s1.pt   acc 1.0
    reduce.py (gauge moves + snapped constants) -> runs/red_s1.pt    acc 1.0, 11 params
    pick.py   (rank by derived band width)    -> member 2, alpha = 3
    build.py                                  -> submission.py

No fine-tuning step survives in the shipped model: the 11-parameter fine-tunes were
run and were *rejected*, because selecting on band width rather than on accuracy
showed they had made the code less linear. The gauge moves in `reduce.py` are exact
(measured delta `0.00e+00`) and the constant snapping is measured in the same script,
so every number in `submission.py` traces back to the cold run.

Two negative results, recorded because they shaped the recipe:

* **Cold training directly in the final pinned coordinates does not work.** It sits at
  ~0.157, the "attend to `i-1`" shortcut, and does not leave even with a soft-to-sharp
  attention curriculum (`logs/final_anneal.log`, flat for 7k steps). Pinning the
  read-out scale fights the code scale during the search. The parent is therefore
  trained with those free and gauged afterwards, which costs nothing: the gauge moves
  are exact and measured at `0.00e+00`.
* **Randomising the shape constants over a band does not work at all**, warm or cold.
  Once `alpha|x-theta| > 1` for a transparent pair the clamp saturates and the
  gradient through the notch dies, so the high-alpha samples contribute noise rather
  than signal. Tried from a warm 1.0 start over `alpha` in [1.5, 6] and [1.2, 12]
  (`logs/aband_6.log`), and from the well-conditioned reduced member over [1.5, 8]
  with the read-out scale both free and fixed (`logs/wide_ls.log`,
  `logs/wide_fix.log`) — all four fall from 1.0 to below 0.25. Ramping `alpha`
  upward during training (`logs/ramp*.log`) does not collapse, but the members it
  produces are *narrower*-banded than the ones it started from.

  So a wide band is something to **select** for, not to train for: `pick.py` ranks
  members that already exist by the band their weights imply. That is why the shipped
  model is a reduced member rather than a fine-tuned one.

One tooling bug worth recording, since it briefly produced confident nonsense: the
`--alpha_end` ramp built its evaluation corners before overwriting `cfg["alpha"]`, so
those runs trained at the ramped `alpha` while *scoring* at the starting one and
reported 1.0 for members whose notch was fully closed. `certify.py` disagreed with the
reported accuracy, and the certificate was right — the true exact-match for one such
member was 0.62. Fixed by building the corners after the mutation. The lesson kept:
an independent check that assumes a mechanism is worth having precisely because it
can contradict the training loop.
