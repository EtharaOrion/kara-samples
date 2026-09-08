# 8-digit addition — a 12-parameter transformer

`/workspace/submission.py` holds a single transformer block with **12 registered
parameters**. It is exact on every 8-digit input, and that is proved rather than
sampled (see *Certificate*).

    code_free (9)   the digit code, code[1..9]
    knee      (2)   the two clamp thresholds
    fold      (1)   the "subtract ten" write-back
    -------------
    12          registered floating-point parameters

## The model

Tokens are digit *pairs*, LSB-first: for `n` places the sequence is `P = n + 2`
positions — a `(0,0)` pad, the `n` pairs `(a_i, b_i)`, then a trailing `(0,0)`
pad that receives the final carry. Position `p` emits answer digit `p-1`, so the
whole sum comes out of **one forward pass**; `add()` only slices digits in and
digits out.

The residual stream is a single scalar channel. A token embeds as
`code[a] + code[b]` from one 10-entry table that is also used as the read-out
prototypes (tied embedding / read-out; the read-out is negative squared distance
to each prototype).

A two-unit clamp bank reads the residual and produces one key/value stream:

| place                   | gates  | key    | value |
|-------------------------|--------|--------|-------|
| absorb    `a+b <= 8`    | (0,0)  |   0    |  0    |
| transparent `a+b == 9`  | (1,0)  | -400   |  0    |
| generate  `a+b >= 10`   | (1,1)  |   0    |  1    |

Transparent places are *notched out* of the key, so they cannot be attended to;
absorb and generate places are equally attendable and the recency bias
`lam * (p - j)` then picks the **nearest earlier place that settles the carry**.
That is one-step carry lookahead, and it is why a single block suffices for any
width. Two masks read the same stream: strictly causal gives the carry coming
*in* to a place, inclusively causal the carry going *out* of it. The residual
becomes `y = x + carry_w * c_in + fold * c_out`, i.e. add the carry-in and, if
this place carries out, subtract ten.

Nothing about which place is transparent is wired in: the gates are computed
from the learned `code` and the learned `knee`s, so the split is something the
training discovered.

## Is the attention doing real work?

`ablate.py`, run on the shipped weights (`log_final_audit.txt`):

    uniform 8-digit: shipped 1.0000 | key content removed 0.7157 | 134 distinct routings | differs from attend-to-previous on 0.567 of inputs
    carry-heavy    : shipped 1.0000 | key content removed 0.7257 | 136 distinct routings | differs from attend-to-previous on 1.000 of inputs
    mixed          : shipped 1.0000 | key content removed 0.5807 | 256 distinct routings | differs from attend-to-previous on 0.725 of inputs

Deleting the *content* term from the key — leaving the fixed relative-position
bias, i.e. exactly "a fixed pattern dressed up as attention" — collapses the
model. The routing genuinely varies with the input, and on carry-heavy inputs it
differs from the trivial attend-to-your-predecessor pattern on 100% of them.

## Certificate — exact on the whole domain, not just on a sample

`certify.py` proves correctness for **all** 8-digit inputs rather than sampling:

1. **Saturation.** Over all 100 digit pairs and the `(0,0)` pad, every clamp
   pre-activation sits outside `[0,1]` with slack ≥ **1.648**. So both gates are
   exactly 0 or 1 for every reachable input, the key and value are exactly
   constant within a class, and the class map is *checked* (not assumed) to
   coincide with `a+b <= 8` / `== 9` / `>= 10`.
2. **Attention.** The pattern therefore depends only on the class pattern, of
   which there are `3^8 = 6561`. All are enumerated in float64; the two heads
   deviate from the exact integer carry-in/carry-out by at most `6.14e-6`,
   giving a residual drift bound of `6.35e-5`.
3. **Read-out.** All 200 reachable `(digit pair, carry-in)` cases decode to the
   right digit, with a safety radius of **0.3117**.

`0.3117 > 6.35e-5` — a margin of **4906x** — so no 8-digit input can be wrong.
`verify.py` then exercises the graded file the way a grader would, through
`build_model()` / `add()`: 1,000,000 uniform 8-digit pairs, 200,000 carry-heavy
pairs, all `3^8` carry structures, widths 2–16, CPU/CUDA agreement — **0 errors
everywhere**.

## How the weights were produced

All training is outside the graded file. `submission.py` contains the model and
its inference path only, imports nothing but `torch`, and is 4953 bytes.

    two_phase.py   phase 1: 16384 random inits trained on single-place addition
                   (a cold multi-place start does not learn at all — the routing
                   noise stops the digit code from ever forming; diagnosed in
                   log_probe1.txt / log_diag.txt).
                   phase 2: winners replicated, a data-driven re-init lottery on
                   the second bank unit, then trained on places [2,3,5,8] with a
                   transparency ramp.  860 members reach >= 0.999 held out.
    reduce.py      project: the learned 2-channel code comes out numerically
                   rank-1 (singular ratio ~2e-3), so it is projected onto its top
                   singular direction and retrained in 1 channel.
                   ship: spend the one residual-scale gauge on carry_w = 1 and
                   substitute the saturation constants.
    finetune.py    retrain the 12 remaining values against read-out margin and
                   bank saturation.
    export.py      pick the member with the widest *certified* margin; the
                   rewrite is checked bit-exact against the trainer (max logit
                   difference 0.000e+00).
    build.py       emit submission.py: the class source verbatim plus the
                   trained values as literals.

Held-out data is a width-agnostic 1-in-16 hash bucket over the digits, so the
graded pairs are unseen; evaluation corners include `n=11`, a width never
trained on.

The non-parameter buffers are architectural constants and gauge fixings, not
fitted values: `code0 = 0`, `val_b = 0`, `val_w = (0,1)` (forced by the gate
encoding), `carry_w = 1` and `ls = 1` (the residual-scale and read-out-temperature
gauges, which are exact symmetries of the argmax), and `bank_w = 8`,
`key_w = ±400`, `lam = -12` (saturation constants — any sufficiently steep bank,
any key notch deeper than `|lam| * P`, and any `lam` in a wide interval give the
identical function; the knees absorb the rest).

## Why this stops at 12

Twelve is the honest count for this architecture. There is exactly one continuous
gauge freedom (overall residual scale) and it is already spent on `carry_w = 1`;
the 9 code values, 2 knees and the fold are each a real degree of freedom. The
learned code is *not* an arithmetic progression — successive gaps run 0.9159,
0.9240, 0.8996, 0.9240, 0.9511, 0.9100, 0.9248, 0.9259, 0.9142 — which is what
tells you the model fitted the code rather than being handed base-ten linearity.

Every route to 11 that I could find gets there by asserting that linearity: tying
`fold = -(code[9] + code[1])` ("ten in residual units"), or pinning `code[9] = 9`
and tying `carry_w = code[1]`. Those compute part of the answer from an imposed
structure instead of learning it, which is what the task rules out. The scoring
gain would be about +0.0074; that is not worth handing the model the thing it is
supposed to learn.

## Files

    submission.py        the graded file (model + inference only)
    ship_model.py        the shipped class; build.py copies it verbatim
    arch.py              ensemble-batched functional forward used for training
    data.py              sampler + held-out hash split (labels for training only)
    lab.py               loss, exact-match eval, per-member clipping
    two_phase.py         the two-phase curriculum
    reduce.py            C=2 -> C=1 projection, then the shipped form
    finetune.py          final training of the 12 values
    export.py            member selection by certified margin + shipped rewrite
    build.py             emits submission.py
    certify.py           whole-domain proof
    verify.py            independent audit through build_model()/add()
    ablate.py            attention ablation
    probe.py diag.py study2.py train_parent.py     diagnostics and dead ends
    log_*.txt            run logs, including the failures
    *.pt                 checkpoints
