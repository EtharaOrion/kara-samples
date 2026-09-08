# Minimal 8-digit addition transformer — working notes

## What is being built

A single Macaron transformer block

    FFN1  ->  1-head strictly-causal self-attention (learned relative distance bias)  ->  FFN2

over ten LSB-first digit-pair tokens.  Position `p` embeds the digit pair at
decimal place `p-1` as `code[a] + code[b]` from one shared 10-entry table;
positions 0 and 9 are `(0,0)` pads.  Every position predicts its own answer
digit by nearest-prototype readout against that same (tied) table, so all nine
digits of the sum come out of one forward pass.  `add()` runs that forward pass
and reassembles the predicted digits.

## Rules I am holding myself to

* Every floating-point value the answer depends on is a trained
  `nn.Parameter`, produced by gradient descent in this workspace.
  **No weight is ever set by hand, computed analytically, or copied from a
  derivation** — not in the submission and not in any scratch file.
* The non-trained constants in the model are buffers and are all structural:
  `0.0` / `1.0` normalisation pins on the code axis, the one-hot axis
  selectors, the causal mask, the integer distance matrix, and a positive
  readout temperature (argmax-invariant).
  A pin is only used where it is a *gauge* choice — a direction in which the
  block has an exact scale symmetry, so pinning removes no expressiveness —
  or where the value is forced by the architecture.  The relative-distance
  slope `lam` is **not** such a direction, so it stays a trained parameter
  even though pinning it would save one.
* Architecture switches remove or tie weights; the remaining weights are then
  **retrained**.  This is structured pruning with warm starts, not weight
  authorship.
* `submission.py` contains the model and its inference path only and imports
  nothing but `torch`.  All training lives in `train_ens.py` / `rungs.py`.
* Everything (scripts, checkpoints, logs, submission) stays in `/workspace`.

## Making the attention do real work

The first parent trained here reached 99.4% with a *fixed* attention pattern:
the query projection collapsed to ~0 and the attention logit was essentially
the distance decay `lam*(i-j)` alone.  That is the failure mode the task rules
out, and it is invisible to 8-digit accuracy alone.  Three things fixed it.

* **Train on longer operands.**  Every parameter is position-independent (only
  the causal mask and the integer distance matrix depend on the position
  count, and both are buffers), so the same weights run over any length.
  Half of every batch is 14-place operands.  Carrying across a run of `k`
  carry-transparent places requires position `i` to select position `i-k-1`;
  a fixed distance decay gives that position weight `~r^k`, so it cannot, at
  any decay rate.  Fixed-pattern models fall off with length (the old parent:
  0.997 at 8 places, 0.80 at 16); a real selection mechanism does not.
* **Select members on the hard sets**, `min(uniform-8, chain-8, chain-14)`.
* **Audit with a content ablation** (`probe.ablate_acc(..., "nocontent")`):
  delete the `q.k` term and keep the model's own `lam*(i-j)` pattern.  This is
  precisely "the fixed pattern dressed up as attention", so accuracy must
  collapse.  Old parent: 0.986 -> 0.982 (unaffected — decorative attention).
  Current parent: 1.000 -> 0.873, with content and position contributing
  comparable spread to the attention logit.

## Files

| file | role |
|---|---|
| `model_src.py` | architecture + config switches; the marked region is copied verbatim into `submission.py` |
| `data.py` | on-GPU operand sampler (any number of places), hash-bucketed held-out split, carry-pattern and edge-case sets |
| `train_ens.py` | trains E independent members at once with `vmap` (seed lottery), on mixed 8- and L-place operands |
| `transfer.py` | warm-start projection of a trained parent onto a smaller architecture |
| `rungs.py` | runs a batch of candidate cuts from one parent, reports which survive |
| `pick.py` | ranks a run's saved alternates by accuracy *and* by whether the attention is load-bearing |
| `probe.py` | instrumented mirror of the forward pass: attention maps and ablations |
| `build_submission.py` | inlines the class source + trained weights as float literals |
| `verify.py` | accuracy, margins, float64/CPU agreement, attention ablations, length generalisation |

## Held-out split

Operand pairs are bucketed by a deterministic hash into 16 buckets.  Training
draws only from buckets 1..15; every accuracy figure labelled "held-out" is
measured on bucket 0 only, which training never sees.

## Parameter budget being targeted

With the residual confined to two axes (axis 0 = answer/code, axis 1 = the
attention key/value channel):

| group | count |
|---|---|
| code table, 10 entries, 2 pins (`code[0]=0` is forced, `code[1]=1` is the axis-0 gauge) | 8 |
| FFN1: 3 knees + 2 free output amplitudes (sum-to-zero tie) | 5 |
| attention: query bias, relative-distance slope | 2 |
| FFN2: 2 knees + 1 free output amplitude (antisymmetric tie) | 3 |
| **total** | **18** |

Why those knee counts.  FFN1 has to turn the place sum `s = a + b` into a
three-level notch — `0` for absorb (`s <= 8`), a large negative value for
transparent (`s == 9`), `1` for generate (`s >= 10`) — which is what makes the
attention select the nearest non-transparent place: the query bias `bq > 0`
drives transparent keys far below everything else, while the generate/absorb
key gap stays smaller than `|lam|` so recency still decides among the rest.
A piecewise-linear function with knees at `8 < t1 < t2 < t3 <= 10` and
amplitudes summing to zero realises exactly that and needs all three knees.
FFN2 then folds `s + carry` modulo 10, i.e. subtracts a constant above 9.5,
which is one up-step and one down-step: two knees with antisymmetric outputs.
