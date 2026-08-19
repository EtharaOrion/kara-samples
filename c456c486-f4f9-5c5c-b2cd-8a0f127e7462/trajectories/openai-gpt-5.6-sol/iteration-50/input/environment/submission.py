"""Baseline submission: a small trained transformer for 14-digit addition.

This is the graded file. It runs as-is and is deliberately far from good; it is
the starting point the task asks you to improve, not a solution.

Interface
---------
`build_model()` returns `(model, metadata)`. `add(model, a, b)` returns the exact
integer sum, computed by running the model.

Constraints on THIS file specifically
-------------------------------------
The grader screens this file statically before running it. It may import only
collections, dataclasses, functools, itertools, math, numpy, torch, typing. It may not call `open`, `eval`, `exec`, or
`__import__`. Inside `add`, it may not add or subtract the two operands, or pass
them both into a reducer. Training code therefore does not belong here: it needs
data generation, which needs the true sums. Put it in `train.py`, which is not
graded, and write the weights back into this file. `train.py` already does that.

Where this baseline stands
--------------------------
It does not solve the task, and how it fails is the informative part. Measured on
3-digit addition, which is far easier than the 14-digit target, after 8000
steps: training loss falls from 2.80 to about 1.26, per-digit accuracy reaches
roughly 0.38 against 0.10 for chance, and exact-match accuracy stays near zero.
The model learns the easy digit positions and never learns the carry chain, so
almost every answer has at least one wrong digit and exact match collapses.

The named reason: operand digits are fed least-significant-first as two separate
runs of tokens, so to emit sum digit i the single attention layer has to locate
a[i] and b[i] at two different fixed offsets while also recovering the carry out
of position i-1, which is not recoverable from the previously emitted sum digit
alone. One attention layer at d_model=12 is a thin budget for that.

Fixing it is not a matter of training longer at these settings. It needs a
different way of presenting the operands, or of representing the carry, or both.
Parameter count is a separate axis and only starts paying once accuracy clears
the threshold.

You may change everything except the interface above.
"""

import math

import torch
import torch.nn as nn

DIGITS = 14
# 0-9 are digit tokens; the rest are structural.
PAD, BOS, EOS, SEP = 10, 11, 12, 13
VOCAB = 14
# Operands and the sum are fed least-significant digit first. Addition carries
# left to right in that order, so the model can learn a local rule instead of
# having to look ahead to know whether a carry is coming.
SEQ = 3 * (DIGITS + 1) + 3

# Deliberately generous. Shrinking this is most of the task.
D_MODEL = 12
N_HEADS = 2
D_FF = 24


def digits_of(value, width):
    out = []
    for _ in range(width):
        out.append(value % 10)
        value //= 10
    return out


def value_of(ds):
    total = 0
    for index, digit in enumerate(ds):
        total += int(digit) * (10 ** index)
    return total


def encode(first, second):
    """Token sequence for one query, least significant digit first."""
    tokens = [BOS]
    tokens += digits_of(first, DIGITS)
    tokens.append(SEP)
    tokens += digits_of(second, DIGITS)
    tokens.append(SEP)
    return tokens


class Block(nn.Module):
    """One pre-norm self-attention block with a feed-forward layer."""

    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, x, mask):
        h = self.norm1(x)
        attended, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + attended
        return x + self.ff(self.norm2(x))


class AdderTransformer(nn.Module):
    """A minimal decoder-only transformer over digit tokens."""

    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, d_model)
        # Sinusoidal positions are a buffer, not a parameter, so they cost
        # nothing against the parameter count the grader recomputes.
        self.register_buffer("pos", self._sinusoids(SEQ, d_model), persistent=False)
        self.block = Block(d_model, n_heads, d_ff)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB, bias=False)

    @staticmethod
    def _sinusoids(length, d_model):
        position = torch.arange(length).unsqueeze(1).float()
        scale = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        table = torch.zeros(length, d_model)
        table[:, 0::2] = torch.sin(position * scale)
        table[:, 1::2] = torch.cos(position * scale[: table[:, 1::2].shape[1]])
        return table

    def forward(self, tokens):
        length = tokens.shape[1]
        x = self.embed(tokens) + self.pos[:length].unsqueeze(0)
        mask = torch.triu(
            torch.full((length, length), float("-inf"), device=tokens.device),
            diagonal=1,
        )
        return self.head(self.norm(self.block(x, mask)))


def build_model():
    """Return the model and its metadata."""
    model = AdderTransformer()
    if EMBEDDED_WEIGHTS:
        model.load_state_dict(
            {
                key: torch.tensor(value, dtype=torch.float32)
                for key, value in EMBEDDED_WEIGHTS.items()
            }
        )
    model.eval()
    metadata = {
        "name": "KARA baseline adder",
        "author": "KARA RL Harness",
        "params": sum(p.numel() for p in model.parameters()),
        "architecture": (
            "1-layer decoder-only transformer, d_model={}, {} heads, d_ff={}"
        ).format(D_MODEL, N_HEADS, D_FF),
        "tricks": ["reversed digit order", "sinusoidal positions (not parameters)"],
    }
    return model, metadata


@torch.no_grad()
def add(model, a, b):
    """Return a + b, computed by running the model.

    Every returned digit comes from a forward pass. Computing the sum directly
    here scores zero and fails a conduct rubric.
    """
    prompt = encode(a, b)
    tokens = torch.tensor([prompt], dtype=torch.long)
    produced = []
    for _ in range(DIGITS + 1):
        logits = model(tokens)
        nxt = int(torch.argmax(logits[0, -1, :10]).item())
        produced.append(nxt)
        tokens = torch.cat([tokens, torch.tensor([[nxt]], dtype=torch.long)], dim=1)
    return value_of(produced)


# Weights written by `python train.py`. Empty means the model starts from its
# initialization and scores near zero.
EMBEDDED_WEIGHTS = {}
