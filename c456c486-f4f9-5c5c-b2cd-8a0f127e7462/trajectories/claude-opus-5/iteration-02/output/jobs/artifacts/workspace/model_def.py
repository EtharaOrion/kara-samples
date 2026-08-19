"""Model definition for the tiny addition transformer.

Contains no training or data-generation code: this exact source is embedded
into submission.py alongside the trained weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

NDIG = 15  # digits of the sum (operands are < 10**14)
SEQ = NDIG + 1  # one leading sentinel slot + one slot per output digit


class TinyAdder(nn.Module):
    """Decoder-only transformer over digit-pair slots (least-significant first).

    Slot 0 is a sentinel; slot i+1 carries the i-th digit of each operand.
    The residual stream is d_model wide: channel 0 receives a compositional
    digit code U[a]+U[b], channel 1 a learned feature of the digit pair, and
    the remaining channels start at zero as scratch space that the attention
    head writes into.  One strictly-causal single-head attention layer plus one
    ReLU feed-forward layer, no normalisation.  The readout is either a linear
    map or the squared distance from channel 0 to ten learned centres.
    """

    def __init__(self, d_model=3, d_ff=4, head="linear", mlp_full_out=True,
                 pair_mode="table", d_feat=3):
        super().__init__()
        self.d_model = d_model
        self.head = head
        self.pair_mode = pair_mode
        d_out = d_model if mlp_full_out else 1

        # --- embeddings -------------------------------------------------
        self.digit_code = nn.Parameter(torch.randn(10) * 0.5)
        if pair_mode == "table":
            self.pair_code = nn.Parameter(torch.randn(55) * 0.5)
            pair_index = torch.zeros(10, 10, dtype=torch.long)
            r = 0
            for i in range(10):
                for j in range(i, 10):
                    pair_index[i, j] = r
                    pair_index[j, i] = r  # (a,b) and (b,a) share a row
                    r += 1
            self.register_buffer("pair_index", pair_index, persistent=False)
        else:
            # channel 1 is a learned ReLU feature of the digit-sum code instead
            self.fw = nn.Parameter(torch.randn(d_feat) * 0.7)
            self.fb = nn.Parameter(torch.randn(d_feat) * 0.7)
            self.fo = nn.Parameter(torch.randn(d_feat) * 0.7)

        # --- attention (one head, d_head = 1) ---------------------------
        self.wq = nn.Parameter(torch.randn(d_model, 1) * 0.5)
        self.bq = nn.Parameter(torch.zeros(1))
        self.wk = nn.Parameter(torch.randn(d_model, 1) * 0.5)
        self.bk = nn.Parameter(torch.zeros(1))
        self.wv = nn.Parameter(torch.randn(d_model, 1) * 0.5)
        self.bv = nn.Parameter(torch.zeros(1))
        self.wo = nn.Parameter(torch.randn(1, d_model) * 0.5)
        self.slope = nn.Parameter(torch.tensor(2.0))

        # --- feed-forward -----------------------------------------------
        self.w1 = nn.Parameter(torch.randn(d_model, d_ff) * 0.7)
        self.b1 = nn.Parameter(torch.zeros(d_ff))
        self.w2 = nn.Parameter(torch.randn(d_ff, d_out) * 0.7)
        self.b2 = nn.Parameter(torch.zeros(d_out))
        self.d_out = d_out

        # --- readout ------------------------------------------------------
        if head == "tied":
            # squared-distance readout against the same learned digit code
            self.tau = nn.Parameter(torch.tensor(1.0))
        elif head == "dist":
            # squared distance to ten learned centres on channel 0
            self.centre = nn.Parameter(torch.randn(10) * 0.5)
            self.tau = nn.Parameter(torch.tensor(1.0))
        else:
            self.wout = nn.Parameter(torch.randn(d_model, 10) * 0.5)
            self.bout = nn.Parameter(torch.zeros(10))

        pos = torch.arange(SEQ)
        rel = (pos[None, :] - pos[:, None]).float()  # j - i, <= 0 where allowed
        self.register_buffer("rel", rel, persistent=False)
        allowed = pos[None, :] < pos[:, None]  # strictly causal
        allowed[0, 0] = True  # sentinel attends to itself so softmax is defined
        self.register_buffer("attn_mask", ~allowed, persistent=False)

    def pair_feature(self, da, db, code):
        """Channel-1 feature of the digit pair at each slot."""
        if self.pair_mode == "table":
            return self.pair_code[self.pair_index[da, db]]
        return F.relu(code[..., None] * self.fw + self.fb) @ self.fo

    def forward(self, da, db):
        """da, db: (B, SEQ) long tensors of digits; returns (B, SEQ, 10) logits."""
        B, S = da.shape
        code = self.digit_code[da] + self.digit_code[db]
        pair = self.pair_feature(da, db, code)
        rest = code.new_zeros(B, S, self.d_model - 2)
        x = torch.cat([code[..., None], pair[..., None], rest], dim=-1)

        q = x @ self.wq + self.bq
        k = x @ self.wk + self.bk
        v = x @ self.wv + self.bv
        scores = q @ k.transpose(1, 2) + self.slope * self.rel[:S, :S]
        scores = scores.masked_fill(self.attn_mask[:S, :S], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        self.attn = attn  # exposed for inspection; not used by the forward result
        x = x + (attn @ v) @ self.wo

        h = F.relu(x @ self.w1 + self.b1)
        upd = h @ self.w2 + self.b2
        if self.d_out == self.d_model:
            x = x + upd
        else:
            x = torch.cat([x[..., :1] + upd, x[..., 1:]], dim=-1)

        if self.head == "tied":
            return -self.tau * (x[..., 0:1] - self.digit_code) ** 2
        if self.head == "dist":
            return -self.tau * (x[..., 0:1] - self.centre) ** 2
        return x @ self.wout + self.bout
