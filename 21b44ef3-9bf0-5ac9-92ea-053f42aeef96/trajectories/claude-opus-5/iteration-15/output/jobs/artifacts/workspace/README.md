# An 8-digit adder in twelve trained parameters

`submission.py` holds one transformer block with **12 registered parameters**.
It is exact on every 8-digit addition — not "99% on a sample", but proved over
the whole input domain (see *Certificate*).

Reproduce the whole thing, random init to graded file, with `bash run_all.sh`.

---

## What is in the model

A sequence of *places*, little-endian. Position `p` carries the digit pair of
one place; the residual stream is a single scalar channel that starts at
`code[a_p] + code[b_p]`. Position 0 is a leading pad (it gives the
strictly-causal head something to attend to at the least-significant place and
supplies carry-in 0); position `n+1` is where the final carry surfaces. All
`n+1` answer digits come out of **one** forward pass.

```
x  = code[a] + code[b]                      residual stream, 1 channel
g  = clamp(bank_w * x + knee, 0, 1)         feed-forward sublayer, 2 units
k  = g · key_w        v = g · val_w         key / value stream
logit[p,j] = k[j] + lam * (p - j)           content term + recency bias
carry_in  = softmax(logit, strictly causal)  · v
carry_out = softmax(logit, inclusively causal) · v
y  = x + carry_w * carry_in + fold * carry_out
out= argmin_d (y - code[d])^2               read-out tied to the code table
```

### The twelve parameters, and what the model had to learn

| parameter | size | what it is |
|---|---|---|
| `code` | 9 | the codes of the digits 1..9 |
| `knee` | 2 | the two thresholds of the feed-forward bank |
| `fold` | 1 | write-back weight of the base fold |

Nothing about base ten is written into the file. Training had to discover all
three of the facts that make addition work, and it did:

* **the codes are a linear ramp** — the fitted table is
  `[0, 1.083, 2.128, 3.173, 4.225, 5.278, 6.331, 7.376, 8.421, 9.504]`, i.e.
  `code[d] ≈ d × 1.056`, so that `code[a] + code[b]` is a faithful stand-in for
  `a + b`. It was free to choose any ten reals and chose a ruler.
* **the carry thresholds** — the bank's knees land at digit sums 8.58 and 9.62,
  splitting the 100 digit pairs into *absorb* (`a+b ≤ 8`), *transparent*
  (`a+b = 9`) and *generate* (`a+b ≥ 10`). Nobody told it that 9 and 10 are
  where addition turns over.
* **the mod-ten fold** — `fold = -10.536 ≈ -10 × 1.056`, exactly one full turn
  of its own ruler, which is what makes the read-out wrap.

### How the attention earns its name

The block is a carry-lookahead adder that the training found on its own. A
transparent place (`a+b = 9`) is pushed to a large negative key, which makes it
*unattendable*; the recency bias then routes every query to the nearest earlier
place that actually settles the carry. That distance is a function of the
digits, not of the position.

Measured on 4096 carry-heavy inputs (`verify.py`):

| variant | accuracy | distinct routings | hop |
|---|---|---|---|
| as shipped | **1.0000** | 256 | max 9, mean 2.05 |
| attention map frozen at its batch mean | 0.1311 | — | — |
| content term dropped, recency bias kept | 0.3640 | 1 | always 1 |

The third row is what "a fixed pattern dressed up as attention" would score.
Freezing the map at its average over inputs destroys the model, and queries
reach as far back as 9 positions depending on what the digits are.

### The buffers, stated plainly

Six buffers hold nine floats: `code0 = 0`, `bank_w = 8`, `key_w = ∓400`,
`val_w = (0, 1)`, `carry_w = 1`, `lam = -12`. **These are not fitted values.**
Two of them (`code0`, `carry_w`) fix the block's exact gauge freedoms — the
model is invariant under `code → code + t` and under a global rescale, so
pinning one offset and one scale removes redundancy rather than information.
The other four are round numbers I chose for the architecture: a bank gain, a
key contrast, a value read-out and a recency slope. None encodes anything about
base ten, none varies between members or between runs, and the model is trained
under them (stages 3 and 4 below), not merely rewritten to use them.

---

## How the weights were produced

`run_all.sh`, four stages of ordinary gradient descent, each seeded from the
survivors of the last. Members are trained as one batched ensemble on a leading
axis — Adam is elementwise and gradients are clipped per member, so members
never interact. Seed variance dominates at this size, so a wide lottery is much
cheaper than sequential restarts.

| stage | what is free | result (seed 15) |
|---|---|---|
| 1 cold | everything, from random init, 2 channels, E=2048 | 19 members ≥ 0.999 |
| 2 project | codes collapsed onto their principal direction (a re-init: PCA residual 0.0035, singular ratio 0.0013), then retrained at 1 channel, E=256 | 242 ≥ 0.9999 |
| 3 sharpen | attention temperature and key contrast replaced by `-12` / `±400`, then **retrained under them**, E=256 | 58 ≥ 0.9999 |
| 4 ship | rewritten into the exact 12-parameter shipped form; those 12 values trained under the shipped constants, E=256 | 122 ≥ 0.9999, of which **99 certify exact** |

Two notes on why the ladder has four rungs rather than one.

* A one-channel code is not reachable from a cold start — E=512 for 3000 steps
  from random init at C=1 got to 0.0054. The parent needs two channels to find
  the mechanism; stage 2 then collapses it and trains the collapse back out.
* Substituting the sharp constants is *not* sufficient by itself. The learned
  key contrast is about 30 against `lam ≈ -2`, so a transparent place is only
  ~2.5 distance-steps worse than a real one and carry chains longer than two
  mis-route; dropping `lam` to `-12` alone takes accuracy from 1.00 to 0.90 and
  it does not come back. Stage 3 sets the contrast and the slope *together* and
  then trains, which is what recovers it.

Stage 4 also carries a saturation term: `relu(1 - dist(z, [0,1]))` on the bank
pre-activations, weight 0.5. It asks the bank for a decisive answer at every
place — it says nothing about *where* the thresholds belong, only that they
should not sit on top of a reachable digit sum. Without it the shipped member
had a bank that straddled the pair (2,7) and could not be certified; with it,
99 of 122 members certify.

**Held-out split.** A 1-in-16 hash bucket of the operand pairs is excluded from
training everywhere (`data.py`), and every accuracy reported during training is
measured on that bucket only.

---

## Certificate: exact on the whole domain, not just on a sample

`certify.py` proves exactness over all 8-digit inputs by enumeration, using the
fact that the block factors the input into three finite stages:

1. **bank** — sees only `x = code[a] + code[b]`, so 100 digit pairs cover it.
   Each must drive both units *hard* into 0 or 1, and the class must agree with
   arithmetic. Worst saturation slack: **+1.0000** (nothing is near a knee).
2. **attention** — keys and values depend on the input only through those three
   classes, so all `3^8 = 6561` class patterns cover it. Each is evaluated in
   float64 against the exact carry recurrence. Worst drift: **6.1e-6**.
3. **read-out** — then depends only on `(a, b, carry-in)` per place, so 100 × 2
   combinations cover it. Worst margin: **0.9462**, against a worst-case
   margin loss from the attention drift of 0.0013 — a **702× safety factor**.

Since the worst margin beats the largest possible drift by 702×, the read-out's
arg-max cannot flip on any input: the model is exact on all 8.1e15 pairs.

## Independent check of the graded file

`verify.py` imports `/workspace/submission.py` by path the way a grader would,
shares no code with the training side, and checks answers against Python's own
arithmetic.

```
imports   ['torch']
params    12   [('code', (9,)), ('knee', (2,)), ('fold', (1,))]
uniform 8-digit x1000000              1000000/1000000  exact
  of those, the held-out bucket only     62254/62254   exact
carry-enriched x200000                 200000/200000   exact
  of those, the held-out bucket only     12321/12321   exact
all 3^8 carry patterns                    6561/6561    exact
hand-picked edge cases                      12/12      exact
add(model, a, b), one call at a time     20000/20000   exact
widths n = 1,2,4,8,12,16,20,24,32        8000/8000     exact each
float32 vs float64                       identical
RESULT: PASS
```

Widths past 8 are head-room, not a requirement — the graded operands are always
8 digits. The block stays exact out to at least 32 places because the key
contrast (400) buys about `400/12 ≈ 33` positions of carry routing.

---

## Why twelve and not eleven

Two obvious ways to reach 11 exist and both were rejected. Setting
`code[d] = d·σ` replaces the learned code table with one scale parameter, and
setting `fold = -10·σ` ties the fold to it. Each removes a parameter by *giving
the model* the fact it is supposed to discover — that the digit codes are a
base-ten ruler and that the read-out wraps at ten. The task is to learn to add;
writing the ruler into the architecture is not making the model smaller, it is
moving the answer out of the model. Twelve is the floor once the base-ten
content stays learned: nine free codes (the tenth is gauge), two thresholds and
one fold, with no gauge freedom left to exploit.

---

## Files

| file | role |
|---|---|
| `submission.py` | **the graded file** — model and inference path only, imports `torch` |
| `run_all.sh` | cold init → graded file, one command |
| `arch.py` | model definition; the region between the SHIPPED markers is copied verbatim into `submission.py` |
| `ens.py`, `data.py`, `train.py` | ensemble forward, sampler with held-out split, cold trainer |
| `retrain.py` | stages 2–4 (`--project`, `--set/--freeze`, `--shipped`) |
| `reduce.py` | the exact rewrites and the certified constant substitution |
| `build.py` | emits `submission.py` from a checkpoint member |
| `certify.py` | whole-domain certificate |
| `select.py` | certifies every member and ranks them |
| `verify.py` | independent end-to-end check of the graded file |
| `logs/run15_*.log` | the actual output of the run that produced the shipped weights |
| `ckpt/run15_*.pt` | that run's checkpoints; member 169 of `run15_ship.pt` is what shipped |
| `ladder.py`, `project.py` | earlier drivers, superseded by `retrain.py`; kept for the record |
| `ckpt/known_good_ship2_m29.py`, `ckpt/probe_submission.py` | earlier certified builds kept as fallbacks — not the graded file |
