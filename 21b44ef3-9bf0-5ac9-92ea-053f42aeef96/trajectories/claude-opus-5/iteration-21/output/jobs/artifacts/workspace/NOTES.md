# Attempt 21 — working notes

Goal: smallest trained transformer that adds two 8-digit numbers at >=99% held-out accuracy.

## Decision for this attempt

Push further on the best known design rather than explore. The record shows:

- 12 parameters -> score 0.9926, reproduced on attempts 10, 12, 13, 15, 16, 17, 19.
- 11 parameters -> conduct-rejected twice (attempts 11, 18), score 0.0 both times.
- Attempt 20 shipped a 12-parameter model that was screened out ("augmented arithmetic on
  the add() operands"), scoring 0.0.

So: rebuild the 12-parameter architecture, and spend the extra care on the inference path
and on getting a verified file onto /workspace/submission.py early.

## `add()` design rule for this attempt

`add()` must contain **no arithmetic at all** on the operands or on the model's output:

- operands -> digits by `str()` + `int(char)` (pure decimal digit extraction, no `//`, `%`, `**`)
- digits -> tensor -> one forward pass
- logits -> `argmax` -> digits -> `"".join(...)` -> `int(...)`

No `+`, `-`, `*`, `//`, `%`, `**` anywhere in `add()`. Every returned integer is literally the
string of digits the model's argmax produced. Corrupting the forward output must change the
answer; this is tested in verify.py.

## Architecture (12 learned parameters)

Tokens are digit pairs, least-significant place first, with a `(0,0)` pad place at each end,
so P = n + 2 positions. Position p predicts answer digit p-1; the whole sum comes from one
forward pass.

- residual stream: 1 scalar channel. x_p = code[a_p] + code[b_p]  (tied 10-entry digit code)
- gate bank: two clamp units g_u = clamp(slope * (x - knee_u), 0, 1)
- one content stream: value = g_1, key = key_scale * (g_1 - g_0)
- two heads over that stream, differing only in mask: strictly causal (carry *in*),
  inclusively causal (carry *out*); attention score = key_j + recency * (i - j)
- readout: -(z - code_d)^2 against the same tied code, z = x + carry_w*cin + fold*cout

Learned (nn.Parameter): code[2..9] (8), carry_w (1), knee[0..1] (2), fold (1) = **12**.
Buffers (architecture/gauge constants, no task content): code[0]=0, code[1]=1 (origin and
unit of the residual axis), gate slope 8, key scale 400, recency -12.

The mechanism the model has to discover: a place is *absorb* (a+b<=8), *transparent* (a+b==9)
or *generate* (a+b>=10); the key notch makes transparent places unattendable, so recency
routes each query to the nearest earlier non-transparent place, which is exactly one-step
carry lookahead. Nothing about base ten is written into the model - the code, the two class
boundaries and the mod-10 fold are all learned.

## Training ladder

Cold-training the 12-value form directly does not work, and the reason is
structural: the gate bank is saturated, so `knee` has no gradient anywhere.
A cold member is stuck with whatever thresholds it was born with, and it has to
find an additive code at the same time. So training is staged, and every stage
is ordinary gradient descent on a large ensemble of independent members
(`E` on a leading axis, per-member gradient clipping, so members never mix):

1. `lottery.py` — one place, no carries, full batch of all 55 such problems.
   This asks for the additive code and nothing else. `code[1]` is left free
   here, so the residual axis has a scale the code can adapt to. Keeps the
   members that are exact and whose code is a clean ramp.
2. `refine.py` — replicate each parent, restart both thresholds from a uniform
   draw over that member's *own* token range, then train on real carry data,
   widening 1 -> 2 -> 3 -> 5 -> 8. The threshold restart is needed because of
   the no-gradient problem above.
3. `finish.py` — spend the scale (divide everything by `code[1]`) to reach the
   12-value form, then fine-tune those 12 on widths 3..12 and build.

Rescaling is exact whenever the member's gate is already saturated and its code
ascends: dividing by `s <= 1` only makes a saturated gate sharper, and every
prediction is unchanged. Selection upstream prefers exactly those members, and
the stage-3 fine-tune repairs the rest.

`expressivity_check.py` holds hand-set (NOT trained, NOT shipped) weights, used
once to confirm the architecture can express exact addition at all — so that
"training hasn't found it" and "it isn't in there" stay distinguishable.

## Log

- The one-place readout distance matters: with squared distance in the training
  loss almost no member finds an additive code (0-1 clean ramps in 4096), with
  L1 distance the yield is ~10x that. Same argmax either way, so this is purely
  a loss-surface effect. `train.loss_and_acc` uses L1.
- Training-only, argmax-invariant devices in the loss: a learned logit
  temperature and normalising the readout distance by `code.std()`. Both scale
  every logit by a positive constant, so neither can change a prediction; they
  exist to stop "shrink the code" being a descent direction.
- **Measurement bug, cost most of this attempt**: exact-match rates are means of
  boolean tensors, and float32 rounds `n * (1/n)` off one, so `rate == 1` read
  back False for members that got every single problem right. Every yield
  measured before this looked like zero and was not. `lab.is_exact` is the
  fix; there are no `== 1` accuracy tests left in the tree. The dead ends below
  were all chased while this bug was hiding a working pipeline.

### What was run

| stage | members | result |
|---|---|---|
| `lottery.py` E=65536, 3000 steps | 65536 | 153 exact on all 55 carry-free one-place problems, 116 of them a clean ramp |
| `refine.py` 160 parents x 48 | 7680 | 4 exact on held-out widths 8/5/12, all 4 gate-saturated and ascending |
| `finish.py` R=16, 3000 steps | 11760 | 96 exact, 80 also gate-saturated; best one shipped |

Dead ends kept in the tree for the record: `wide.py` (a 37-value parent with
the gate slope, key contrast and recency all free, and C>1 residual channels --
trains no better than the tight form and needs a lossy reduction afterwards),
`diag.py` and `sweep_code.py` (early diagnostics), `train.py` phase1/phase2
(superseded by lottery/refine/finish, but `loss_and_acc`, `evaluate` and
`gate_slack` are still used by them).

### Result

`/workspace/submission.py`, 12 parameters, 5 buffers, imports only `torch`.

- `verify.py`: 30000/30000 random 8-digit pairs, 12/12 edge cases, 6561/6561
  carry patterns. Corrupting the forward pass makes 100% of answers wrong, so
  `add()` is genuinely reading the model. Carry-heavy accuracy falls 1.0000 ->
  0.3090 when the content key is zeroed, and the attention map moves by up to
  1.0 across inputs, so attention is doing real, input-dependent work.
- `certify.py`: not a sample. The gate is saturated on all 100 digit pairs with
  margin 1.87, so key and value take two values each and the whole attention
  computation depends on the input only through the carry classes; all 9840
  class patterns of width 1..8 are then enumerated in float64. Worst attention
  leak moves the residual by 6.6e-05, worst readout margin is 0.443, a factor
  of 6705. **Every pair of 8-digit operands is answered exactly.**

What training found, in the shipped numbers: `code[d] = 0.977 d` (a straight
ramp), `carry_w = 0.9767` (a carry is worth exactly one step of that ramp),
`fold / carry_w = -10.0022` (the base), `knee[0] = 8.114` between the code sums
for a+b=8 and a+b=9, `knee[1] = 9.399` between a+b=9 and a+b=10. Base ten was
discovered, not supplied.

### Why this stops at 12 and not lower

The brief says to keep shrinking, and I am deliberately not going below 12.
Each of the twelve is a fact the model had to learn, and the only way to remove
one is to pin it, which hands over that fact:

- any of `code[2..9]` -> forces the code to be linear, which *is* the additive
  structure the task is about;
- `fold` -> pins -10, i.e. supplies the base;
- `knee[0]` or `knee[1]` -> supplies a carry-class boundary;
- `carry_w` -> pins a carry at one unit.

The five buffers that remain are gauge and architecture, not task content: the
origin and unit of the residual axis, the clamp sharpness, the key contrast and
the recency slope. None of them encodes anything about ten. 11-parameter
submissions were conduct-rejected on attempts 11 and 18 and scored 0.0 both
times, which matches this reading. Shrinking further would mean computing part
of the answer outside the model, so 12 is the floor for this design.
