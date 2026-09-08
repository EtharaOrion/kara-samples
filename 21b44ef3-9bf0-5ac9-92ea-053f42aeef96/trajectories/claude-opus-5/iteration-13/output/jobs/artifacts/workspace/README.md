# Minimal 8-digit addition transformer — 12 parameters

`/workspace/submission.py` holds a single-block transformer with **12 learned
parameters** that adds two 8-digit numbers. On 2^20 held-out pairs it is exact
(1.000000), and it is *proven* correct on all ~8.1e15 operand pairs in the graded
range — not estimated by sampling. See "Correctness" below.

```
$ python verify.py
summary: {'shape': 'pass', 'heldout': 'pass', 'domain': 'pass',
          'numerics': 'pass', 'width': 'pass', 'attention': 'pass'}
```

## The model

Tokens are digit *pairs*, least-significant place first, with a `(0,0)` pad at each
end, so `P = n + 2` positions. Position `i` predicts the answer digit of place `i-1`;
one forward pass emits all `n+1` answer digits. The residual stream is a **scalar**.

```
embed   u_i = code[a_i] + code[b_i]
bank    g_i = clamp(bw * u_i + bb, 0, 1)                    the only nonlinearity
k / v   k_i = <g_i, kw>,  v_i = <g_i, vw>                   one shared kv stream
attn    logit[h,i,j] = k_j + lam * (i - j),  masked
        head 0 strictly causal (j < i), head 1 inclusively causal (j <= i)
        c[h,i] = sum_j softmax_j(logit[h,i,:]) * v_j
out     z_i = u_i + carry_w * c[0,i] + fold * c[1,i]
        logits[i,d] = -(z_i - code[d])^2                    read-out tied to `code`
```

| tensor | count | kind |
|---|---|---|
| `code_free` | 9 | **learned** — the code of digits 1..9 |
| `bb` | 2 | **learned** — the two clamp knees |
| `fold` | 1 | **learned** — what is subtracted when a carry leaves a place |
| `code_zero`, `carry_w` | 2 | fixed: gauge (origin and scale of the residual stream) |
| `bw`, `kw`, `vw`, `lam` | 8 | fixed: architecture (ramp width, attention sharpness, recency slope, value projection) |

Every buffer is a hand-set constant (`0, 1, ±8, 400`); no trained float hides in one.
`sum(p.numel() for p in model.parameters()) == 12`, independent of the metadata.

## What it learned

Nothing about carry propagation was designed in — the training loss only sees answer
digits. What emerges is **carry-lookahead**. Each place falls into one of three
classes by `s = a_i + b_i`: *absorb* (`s <= 8`, kills an incoming carry), *transparent*
(`s == 9`, passes it through), *generate* (`s >= 10`, creates one). The bank learns
knees that split exactly there, giving `g = (1[s<=8], 1[s>=10])`. The key is
`400*(g0+g1)`, which is high for absorb and generate places and **notched down to zero
at transparent ones**, making them unattendable; the recency term `-8*(i-j)` then makes
each query land on the *nearest earlier non-transparent place*. Its value `1[s>=10]` is
precisely the carry that reaches the query. Head 0 (strictly causal) reads the carry
*in*, head 1 (inclusively causal) the carry *out*, and the read-out subtracts `fold`
when a carry leaves.

That is why the model is exact rather than merely accurate: it resolves a carry chain
of any length in **one** attention step instead of iterating. It also runs unchanged at
other widths — no parameter depends on position:

```
 5 digits: 1.000000     11 digits: 1.000000
 8 digits: 1.000000     16 digits: 1.000000     (trained at 8, 5, 11, 3)
```

The learned code is `0, 0.9674, 1.8759, 2.8069, 3.7405, 4.6702, 5.6031, 6.5351,
7.4557, 8.4017` — near-linear, with irregular steps (0.9674, 0.9085, 0.9309, 0.9337,
0.9297, 0.9329, 0.9320, 0.9205, 0.9460). That irregularity is the signature of gradient
descent finding the code, not of a constructed one. Nothing in the architecture told it
that digits are ordered or that the base is ten; `fold = -9.4889` and the knee positions
are likewise discovered.

## Attention does real work

`verify.py` check 6 ablates the attention four ways, on uniform inputs and on inputs
with many transparent places (where *which* position to attend actually varies):

| | uniform | carry-heavy |
|---|---|---|
| intact | 1.0000 | 1.0000 |
| map frozen at its batch mean | 0.0078 | 0.1298 |
| content term deleted (pure recency — a fixed pattern) | 0.7056 | 0.3635 |
| each input given another input's map | 0.4963 | 0.2379 |

On carry-heavy inputs **52% of queries look past their immediate predecessor**, and
there are 128 distinct head-0 maps over 8192 inputs. A fixed pattern reaches 0.36; the
input-dependent map is what carries it to 1.0.

## Correctness: a whole-domain proof, not a sample

8.1e15 pairs cannot be enumerated, so `certify.py` factors the model instead and checks
each factor exhaustively:

1. **Bank (100 checks).** Every position's `u` is one of 100 digit-pair sums. Both
   clamp pre-activations lie outside `[0,1]` for all 100 — minimum slack **0.457** — so
   the bank output is exactly binary and equals `(1[s<=8], 1[s>=10])`. Each place now
   affects the rest of the network only through its class.
2. **Attention (3^8 = 6561 checks).** With classes fixed, the key vector of a sequence
   is determined by its class pattern. All 6561 patterns are run through the exact
   softmax in float64; the largest deviation of either head from the ideal carry is
   **delta = 3.355e-04**.
3. **Read-out (200 checks).** For all 100 digit pairs and both carry-ins, the noise-free
   `z` decodes correctly, with **margin 0.286** to the nearest decision boundary.

Combining, the attention error can perturb `z` by at most `(|carry_w| + |fold|) * delta
= 3.52e-03`, which is **81x smaller** than the read-out margin. Every input in the
domain is therefore answered correctly. `verify.py` re-runs this on the shipped
literals, not on a checkpoint.

## How it was produced

Training lives entirely outside `submission.py`, which contains only the model and its
inference path and imports nothing but `torch` (verified: it adds zero modules beyond
torch's own, and contains no `exec`/`eval`/`compile`/`open`).

```
core.py     the architecture (functional, with a leading ensemble axis)
data.py     on-GPU sampler; a pair's hash bucket (0..15) is its split, bucket 0 is
            never trained on, so "held out" is by construction
train.py    trains E=1024 independent members at once — parameters never mix across
            the ensemble axis, Adam is elementwise, the gradient clip is per member,
            so each member's trajectory is what it would have been training alone
reduce.py   exact, function-preserving rewrites + diagnostics
stage.py    substitutes the architectural constants
certify.py  the whole-domain certificate above
pick.py     sweeps every trained member through reduce -> stage -> certify
build.py    emits submission.py: model source + weights as plain float literals
verify.py   independent check of the shipped file
```

The pipeline is: **train a wide parent, then rewrite it exactly.**

`train.py` cold-trains a 34-parameter parent (`U=4` clamp units, everything learnable)
to 1.0000 held-out. `reduce.py` then moves it into the shipped coordinates using only
changes of variable that leave the function on the reachable set *identical*:

- **gauges** — translation (`code+t`), scale (`code, e` by `mu`), read-out temperature,
  query/key scale, value affine, key offset. These fix `code[0]=0`, `carry_w=1`,
  `ls=1`, `q=1`, `vw=(0,1)`, `vb=0`, `rb=0` without changing any answer.
- **unit flip** — `clamp(-x+1,0,1) == 1 - clamp(x,0,1)` exactly, for every x.
- **unit merge** — duplicate bank units collapsed, `U=4 -> 2`.
- **ramp narrowing** — `|bw| -> 8` with the knee `-bb/bw` held fixed shrinks each ramp
  around the same knee, so the new ramp is a *subset* of the old one. Exact whenever no
  reachable `u` sat inside the old ramp, which part 1 of the certificate checks.

`reduce.check_exact` confirms the rewritten model reproduces the parent's answers on
120k pairs bit-for-bit before anything else runs. Substituting `kw`, `lam` (attention
sharpness and recency slope) is the one step that is *not* an identity — it is what the
certificate is for, and it is why those tensors are hand-set constants rather than
trained values living in buffers.

`pick.py` puts all 15 members that reached 1.0 through this. Six fail honestly — their
bank is unsaturated, so no exact reduction exists and they are rejected rather than
patched — and three certify. Ranking is by the tightest headroom any check leaves,
since that is what float32 rounding could eat into:

```
headroom 0.28214  slack +0.4575  margin 0.28566  safety x81.19   parentB.pt[2]  <- shipped
headroom 0.14261  slack +0.1426  margin 0.19762  safety x59.39   parent.pt[2]
headroom 0.06710  slack +0.0671  margin 0.30652  safety x89.41   parentB.pt[10]
```

### Why a parent, and not the 12-parameter form directly

`train.py --reduced` is the control: it cold-trains the shipped form itself, with the
constants already frozen and only the 12 numbers (plus `bw`) free. Across **1024 seeds
and 12000 steps the best member reaches 0.0463** held-out, against the parent's 1.0000
by step 3000 (`reduced_control.log`). Two reasons: the clamp knees receive gradient only
while a ramp still straddles some reachable value, and with attention already sharpened
to `kw = 400` the softmax is one-hot from step 1, so the keys get almost no signal. The
wide parent is what makes the mechanism findable; the rewrites are what make it small.

### Dead ends, recorded

- Fine-tuning *after* reduction destroys exactness. Gradient descent parks digit pairs
  inside the clamp ramps (that is the only place `bb` has gradient), so ramp narrowing
  stops being function-preserving. Reduction is therefore the last step, never followed
  by training.
- Annealing `|bw|` upward during training collapses the run: forcing the ramps narrow
  pushes unit 1's knee past `s = 9`, the key notch disappears, and accuracy falls from
  1.0 to 0.03.
- Two reductions below 12 were deliberately **not** taken, because they encode the
  answer instead of learning it: tying `code[d] = d * sigma` (9 parameters -> 1) tells
  the model that digits are linearly ordered in base ten, and tying `fold = -10 * sigma`
  tells it the base. Both are facts the 12-parameter model discovers on its own.

## Why 12 and not fewer

Each remaining number is something the model must learn and cannot derive from the
others without being told the base:

- **9** for the digit code. `code[0]` is free only because the translation gauge is
  spent on it; the other nine are independent learned values.
- **2** for the knees. The key must be high–low–high across absorb / transparent /
  generate, which is not monotone in `u`, so a single clamp cannot produce it. Two
  units, two knees, and they cannot be tied without fixing the transparent class's
  width relative to the (learned) code scale.
- **1** for the fold. It equals `-10 * sigma` only *because* the learned code turned out
  linear — assuming that relation is the base-ten shortcut above. (Here the learned
  `fold = -9.4889` against a mean code step of `0.9291`, i.e. `-10 * sigma = -9.291`:
  the model does not sit at the idealised value, it sits where the loss put it.)

Both remaining gauge freedoms are already spent (translation on `code[0] = 0`, scale on
`carry_w = 1`), so no further parameter can be absorbed by a change of coordinates.
