"""Model definition + inference path for the 14-digit addition transformer.

This file contains ONLY the architecture and the decoding path.  `train.py`
imports it, trains it, and inlines this exact source into `submission.py`
together with the learned weights, so the graded file and the trained file
run identical code.

Design notes
------------
The sequence is 1 BOS slot followed by 15 digit positions, least-significant
first.  Position ``1+i`` embeds the *pair* of input digits ``(a_i, b_i)``.

A single attention head performs the carry lookahead: the query at position
``i`` searches strictly-lower positions for the nearest one that is not a
"propagate" slot (digit-sum 9), and reads whether that slot generated a
carry.  Which position wins depends entirely on the digits at the keys, so
the attention map genuinely varies with the input.  The MLP then combines
the local digit-pair with the retrieved carry into the output digit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

N_POS = 15  # digits of the sum, least-significant first


def _sym_pair_index():
    """Map each ordered digit pair to a slot shared with its transpose.

    Pure integer bookkeeping: slot (a, b) and slot (b, a) read the same
    embedding row, so the 100 pairs occupy 55 rows.  This is weight tying
    over an input symmetry (a+b == b+a); the row contents are still learned.
    """
    idx = torch.zeros(10, 10, dtype=torch.long)
    k = 0
    for i in range(10):
        for j in range(i, 10):
            idx[i, j] = k
            idx[j, i] = k
            k += 1
    return idx


class TinyAdder(nn.Module):
    def __init__(self, d_model=8, d_head=2, d_ff=24, n_pos=N_POS, sym=False,
                 qkb=False, slope0=0.5, addemb=False, h0=6, hyb=False,
                 bout=True, npair=None, feat=0):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.d_ff = d_ff
        self.n_pos = n_pos
        self.sym = sym
        self.qkb = qkb
        self.addemb = addemb
        self.hyb = hyb
        self.feat = feat
        self.has_bout = bout

        # token embeddings
        if hyb and feat:
            # table-free: dim 0 is the compositional code U[a]+U[b] and dim 1 is
            # a tiny scalar->scalar ReLU network applied to it.  The propagate
            # predicate (a+b == 9) has to be *manufactured* here as a bump in a
            # monotone code, rather than looked up in a 55-row table.
            self.digit_emb = nn.Parameter(torch.randn(10) * 0.5)
            self.fw = nn.Parameter(torch.randn(feat))
            # spread the ReLU kinks across the initial range of the code, so the
            # units start out able to see different parts of it
            self.fb = nn.Parameter(torch.rand(feat) * 4 - 2)
            self.fo = nn.Parameter(torch.randn(feat) * feat ** -0.5)
            self.fbo = nn.Parameter(torch.zeros(1))
            self.npair = 1
        elif hyb:
            # residual dim 0 receives a *compositional* digit code, U[a]+U[b],
            # so the model has to learn a code whose sum is informative; the
            # remaining dims get a (transpose-tied) pair embedding, which is
            # where the non-linear carry features have to live.
            self.digit_emb = nn.Parameter(torch.randn(10) * 0.5)
            self.register_buffer('pair_idx', _sym_pair_index(), persistent=False)
            # npair < d_model-1 leaves spare residual dims that carry no token
            # signal at all -- scratch space for the attention to write into.
            self.npair = (d_model - 1) if npair is None else npair
            self.pair_emb = nn.Parameter(torch.randn(55, self.npair) * 0.5)
        elif addemb:
            # compositional: one vector per digit value, shared by both operands,
            # so a slot is embedded as U[a_i] + U[b_i]; a token-wise featurizer
            # MLP then derives the (non-linear) carry features from it.
            self.digit_emb = nn.Parameter(torch.randn(10, d_model) * 0.5)
            self.w0 = nn.Parameter(torch.randn(d_model, h0) * d_model ** -0.5)
            self.b0 = nn.Parameter(torch.zeros(h0))
            self.w0o = nn.Parameter(torch.randn(h0, d_model) * h0 ** -0.5)
            self.b0o = nn.Parameter(torch.zeros(d_model))
        elif sym:
            self.register_buffer('pair_idx', _sym_pair_index(), persistent=False)
            self.pair_emb = nn.Parameter(torch.randn(55, d_model) * 0.5)
        else:
            self.pair_emb = nn.Parameter(torch.randn(10, 10, d_model) * 0.5)
        self.bos = nn.Parameter(torch.randn(d_model) * 0.5)

        # single-head self-attention
        self.wq = nn.Parameter(torch.randn(d_model, d_head) * d_model ** -0.5)
        self.wk = nn.Parameter(torch.randn(d_model, d_head) * d_model ** -0.5)
        self.wv = nn.Parameter(torch.randn(d_model, d_head) * d_model ** -0.5)
        self.wo = nn.Parameter(torch.randn(d_head, d_model) * d_head ** -0.5)
        if qkb:  # query/key offsets: let a position ask a content-free question
            self.bq = nn.Parameter(torch.zeros(d_head))
            self.bk = nn.Parameter(torch.zeros(d_head))
        # learned recency bias on the attention logits (ALiBi-style, one slope)
        self.slope = nn.Parameter(torch.tensor(float(slope0)))

        # feed-forward
        self.w1 = nn.Parameter(torch.randn(d_model, d_ff) * d_model ** -0.5)
        self.b1 = nn.Parameter(torch.zeros(d_ff))
        self.w2 = nn.Parameter(torch.randn(d_ff, d_model) * d_ff ** -0.5)
        self.b2 = nn.Parameter(torch.zeros(d_model))

        # readout to a digit
        self.wout = nn.Parameter(torch.randn(d_model, 10) * d_model ** -0.5)
        if bout:  # per-digit logit offsets; the 10 classes are near-equiprobable,
            self.bout = nn.Parameter(torch.zeros(10))  # so this is often dead weight

    def forward(self, a_digits, b_digits):
        """a_digits, b_digits: (B, n_pos) int64, least-significant digit first.

        Returns logits of shape (B, n_pos, 10) for the digits of the sum.
        """
        b = a_digits.shape[0]
        if self.hyb:
            code = (self.digit_emb[a_digits] + self.digit_emb[b_digits])
            if self.feat:
                h = F.relu(code.unsqueeze(-1) * self.fw + self.fb)
                pair = (h * self.fo).sum(-1, keepdim=True) + self.fbo
            else:
                pair = self.pair_emb[self.pair_idx[a_digits, b_digits]]
            parts = [code.unsqueeze(-1), pair]
            spare = self.d_model - 1 - self.npair
            if spare:
                parts.append(code.new_zeros(code.shape + (spare,)))
            tok = torch.cat(parts, -1)
        elif self.addemb:
            tok = self.digit_emb[a_digits] + self.digit_emb[b_digits]
        elif self.sym:
            tok = self.pair_emb[self.pair_idx[a_digits, b_digits]]
        else:
            tok = self.pair_emb[a_digits, b_digits]             # (B, n_pos, d)
        x = torch.cat([self.bos.expand(b, 1, -1), tok], dim=1)  # (B, 1+n_pos, d)
        n = x.shape[1]
        if self.addemb:
            x = x + F.relu(x @ self.w0 + self.b0) @ self.w0o + self.b0o

        q = x @ self.wq
        k = x @ self.wk
        v = x @ self.wv
        if self.qkb:
            q = q + self.bq
            k = k + self.bk
        scores = (q @ k.transpose(1, 2)) * (self.d_head ** -0.5)

        idx = torch.arange(n, device=x.device)
        rel = (idx[None, :] - idx[:, None]).to(x.dtype)   # j - i, <= 0 where allowed
        scores = scores + self.slope * rel

        allowed = idx[None, :] < idx[:, None]             # strictly causal
        allowed = allowed.clone()
        allowed[0, 0] = True                              # keep row 0 finite (unused)
        scores = scores.masked_fill(~allowed, float('-inf'))

        x = x + (scores.softmax(-1) @ v) @ self.wo
        x = x + F.relu(x @ self.w1 + self.b1) @ self.w2 + self.b2
        logits = x[:, 1:] @ self.wout
        return logits + self.bout if self.has_bout else logits


def _digits(n, n_pos=N_POS):
    """Decimal digits of a non-negative int, least-significant first."""
    out = []
    for _ in range(n_pos):
        out.append(n % 10)
        n //= 10
    return out


def add_with_model(model, a, b):
    a_d = torch.tensor([_digits(int(a))], dtype=torch.long)
    b_d = torch.tensor([_digits(int(b))], dtype=torch.long)
    with torch.no_grad():
        pred = model(a_d, b_d).argmax(-1)[0].tolist()
    total = 0
    for d in reversed(pred):
        total = total * 10 + int(d)
    return total
