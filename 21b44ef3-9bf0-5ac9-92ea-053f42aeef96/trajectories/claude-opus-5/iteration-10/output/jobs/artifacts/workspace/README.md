# A 12-parameter transformer that adds

`submission.py` holds a one-block transformer with **12 learned parameters** that
computes `a + b` for 8-digit operands. It is exact on every 8-digit input, and
that is a proof rather than a sample: see *Certificate* below.

```
uniform 8-digit : 1048576/1048576 = 1.000000     (held-out only: 65387/65387)
carry-enriched  : 524288/524288   = 1.000000
3^8 carry patterns : 26244/26244  = 1.000000
certificate     : exact on every 8-digit input, 112.7x margin
```

## The model

Tokens are digit places, least significant first, with a `(0,0)` pad at each
end. Token `i` embeds as `code[a_i] + code[b_i]` from one 10-entry table that
also serves as the read-out prototype set. Position `i` predicts answer digit
`i`, so the whole sum falls out of a single forward pass.

```
x    = code[a_i] + code[b_i]                     residual stream  (C = 1)
u    = clamp(bw x + bb, 0, 1)                    element-wise bank (U = 2)
key  = u . kw        val = u . vw                one shared key/value stream
oa   = softmax(key + lam (i-j))[j < i]  . val    head 1, strictly causal
ob   = softmax(key + lam (i-j))[j <= i] . val    head 2, inclusively causal
r    = x + oa e1 + ob e2
logits = -ls |r - code|^2
```

The bank splits the 100 digit pairs into three classes by `s = a_i + b_i`:
absorb (`s <= 8`), transparent (`s == 9`), generate (`s >= 10`). `val` is the
indicator of *generate*. `key` is **flat everywhere except a deep notch at the
transparent places**. Attention scores are only defined up to a per-row
constant, so a flat key plus a negative position slope means each head attends
to the *nearest non-transparent place* — which is exactly where the carry
reaching place `i` was born. The strict head therefore returns the carry *into*
place `i` and the inclusive head the carry *out of* it, and the read-out is
`x + carry_in * e1 + carry_out * e2` with `e2` the mod-10 fold.

The attention does real work: freezing both maps to their batch mean (removing
all input dependence while leaving everything else intact) drops accuracy from
1.000000 to 0.000458, and the strict head takes 256 distinct argmax patterns
over 32768 inputs (`verify.py`).

## What is learned and what is a fixed scale

The 12 parameters are all and only the quantities the answer depends on:

| parameter | count | what it is |
|---|---|---|
| `code[1..9]` | 9 | the digit embedding — the number line, discovered by training |
| `bb[0..1]`   | 2 | where carries generate, and where they propagate |
| `e2`         | 1 | the mod-10 fold, i.e. the base |

The buffers are fixed *scales and gauges*, not arithmetic facts. Each one is
either an exact symmetry of the block or a don't-care with a wide admissible
band, and the value used sits inside that band rather than at a tuned point:

| buffer | value | why it carries no information about addition |
|---|---|---|
| `code_fix` | `0` | `code[0] = 0` fixes the code-**translation** gauge (absorbed by `rb`, `bb`) |
| `e1` | `1` | fixes the code-**scale** gauge (absorbed by `bw`, `e1`, `e2`, `ls`) |
| `ls` | `1` | read-out temperature: exactly argmax-invariant for any `ls > 0` |
| `vw` | `(0, 1)` | affine renormalisation of the value stream, absorbed by `e1`, `e2` |
| `bw` | `(-2, +2)` | bank slope; any value steep enough to saturate works (measured saturation error: exactly 0) |
| `lam` | `-8` | position slope; any moderate negative value works |
| `kw` | `(-2000, -2000)` | notch scale; only needs to clear `\|lam\| * places` |

Two of these deserve their evidence spelled out.

**The notch was not imposed.** The cold-trained parent `ck/c2u2_0.pt` has 38
free parameters and nothing pinned. Reading its key and value straight off:

```
s <= 8  absorb        key = -30.186    val = -4.203
s == 9  transparent   key = -60.30     val = -3.234
s >= 10 generate      key = -30.116    val =  0.968
```

Unconstrained training found the notch on its own. It also put `kw`'s two
entries at `-30.186` and `-30.116` by itself — a spread of 0.07 against
`|lam| = 2.67`. Setting them exactly equal is a rescaling inside an open region
whose width is `|lam|`, with the chosen point at its centre, not a fact.

**`vw` at the transparent places is a genuine don't-care.** Those places are
never attended (that is what the notch is for), so the value there is
unconstrained; only the absorb/generate contrast is pinned, and pinning it is
an affine renormalisation the write directions absorb.

Nothing that encodes arithmetic is fixed. In particular `e2 = -10.72` stays
learned rather than tied to `-10 * gamma`; the two bank thresholds stay learned
and untied rather than being locked one code-gap apart; and the code stays a
learned 9-vector rather than a fixed `0..9` ramp. Each of those shortcuts would
save a parameter by handing the model the answer.

## Why 12 and not fewer

The block has to represent 10 code values, 2 bank thresholds, the carry write
and the fold write: 14 numbers, minus the 2 exact gauge freedoms (translation
and scale) = **12**. Every reduction below that removes a real degree of
freedom, i.e. hardcodes an arithmetic fact — the base (`e2 = -10 gamma`), the
carry threshold (`bb0` at 10), the transparency threshold (`bb1` at 9), tying
the two knees one digit apart, or replacing the learned code with a ramp. All
were rejected.

## How it was produced

Everything under `/workspace`. Training code is entirely outside
`submission.py`, which contains only the model and its inference path and
imports nothing but `torch`.

1. **`run_cold.py`** — cold-trains ensembles from scratch on labelled digit
   pairs (`lab.sample`), with a 1-in-16 held-out split by hash. Members are
   trained simultaneously via `torch.func.stack_module_state` + `vmap`; Adam is
   element-wise so the members stay independent. Survivors land in `ck/c*.pt`.
2. **`stages.py`** — the reduction ladder. Each stage applies an exact gauge
   rewrite, retrains an ensemble under a hard cap on `|p - target|` that
   shrinks to zero (applied as a projection *after* the optimizer step, so it
   is a constraint and not a reparameterisation), then turns the landed
   quantity into a buffer. Every cut is checked for exact argmax agreement
   against the model it came from.

```
c2u2_0 (38p) --A--> A1 (18p) --B--> B1 (16p) --C--> C1 (13p) --D--> D2 (12p) --P--> P2 (12p, shipped)
              C=2->1,           code[0]=0,     e1=1 gauge,      ls=1        margin polish
              attention          drop rb        pin bw          (exact)
              canonicalised
```

3. **`build.py`** — emits `submission.py` from a checkpoint, then asserts the
   emitted file reproduces the checkpoint's logits to a max difference of
   exactly `0.0` and has the same parameter count.
4. **`verify.py`** — accuracy over 2^20 uniform pairs, the held-out bucket
   alone, carry-enriched pairs, all `3^8` carry patterns, edge cases, `add()`
   against the batched decode, float64/CPU agreement, the attention ablation,
   and length generalisation.
5. **`certify.py`** — the exhaustive certificate.

The final margin polish (stage `P`) optimises the read-out margin directly
rather than cross-entropy. Cross-entropy scores inflating the code, which
leaves the carry step misaligned with the code gap; the margin objective drove
the worst-case margin from 0.29 to 0.42 prototype gaps and the certificate from
76x to 113x, and made the model exact at 12 and 16 digits as well.

## Certificate

Because the key notch makes each head attend to the nearest non-transparent
place, the read-out at place `i` depends on the entire input only through
`(a_i, b_i, carry_in)` — **200 cases**, all checked exhaustively — plus a
bounded attention leakage. `certify.py` bounds that leakage from the measured
key spread, notch depth and position slope,

```
leak  = e^spread * e^lam / (1 - e^lam)  +  P * e^(-notch + |lam|(P-1))
slack = (vdev + leak * span) * (|e1| + |e2|)
```

and compares it to the worst margin over the 200 cases. For the shipped model:

```
key   : flat level -2000.044 (spread 0), notch depth >= 518.8
value : off {0,1} by at most 0, full range 1.0
leakage (P=10) : 3.356e-04 + 8.9e-194  ->  read-out shift <= 0.003935
place read-out : 200/200 correct, worst margin 0.443368 (0.4206 prototype gaps)
certificate    : 0.443368 > 0.003935  ->  exact on every 8-digit input (112.7x)
```

So the model is provably correct on all 10^16 pairs of 8-digit operands, not
just the ones that were sampled.

## Reproducing

```
python run_cold.py --tag c2u2 --C 2 --U 2          # cold train
python stages.py A --inp ck/c2u2_0.pt --out ck/A1.pt --E 64 --steps 10000 --seed 21
python stages.py B --inp ck/A1.pt --out ck/B1.pt --E 64 --steps 8000  --seed 22
python stages.py C --inp ck/B1.pt --out ck/C1.pt --E 64 --steps 10000 --seed 23 --slope 2
python stages.py D --inp ck/C1.pt --out ck/D2.pt
python stages.py P --inp ck/D2.pt --out ck/P2.pt --E 64 --steps 6000 --lr 0.0015 \
       --sigma 0.01 --seed 32 --mgn 1.0 --ce 0.0 --tau 0.5 --mabs
python build.py ck/P2.pt --out submission.py
python verify.py && python certify.py
```
