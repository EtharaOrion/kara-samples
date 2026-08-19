"""Minimal transformer that adds two integers in [0, 99_999_999_999_999].

A single-layer, single-head causal transformer with 8 learned parameters.
The operands are tokenised as binary digits (LSB first) and the whole sum is
produced by one forward pass: self-attention performs carry-lookahead, with
each slot attending to the nearest earlier slot whose digit pair does not
propagate a carry, and reading off whether that slot generated one.

All learned values are registered nn.Parameters; the remaining constants are
fixed buffers (a gauge choice for an exactly redundant scale, the sign of the
key feature, and the ALiBi recency slope) that were never trained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

N_BITS = 48                  # 10**14 < 2**47, so the sum needs at most 48 tokens
N_SLOTS = N_BITS + 1         # slot 0 is a phantom (0,0) sentinel


class TinyAdder(nn.Module):
    """1 layer, 1 head. Attention = carry lookahead."""

    def __init__(self):
        super().__init__()
        self.U = nn.Parameter(torch.zeros(2))       # digit-token embedding / readout centres
        self.b = nn.Parameter(torch.zeros(2))       # feature-layer biases
        self.w2 = nn.Parameter(torch.zeros(()))     # key/query feature mixing
        self.bq = nn.Parameter(torch.zeros(()))     # query bias
        self.wo = nn.Parameter(torch.zeros(2))      # attention output projection
        # fixed, never trained
        self.register_buffer('w1', torch.tensor(-1.0))
        self.register_buffer('slope', torch.tensor(4.0))
        j = torch.arange(N_SLOTS)
        self.register_buffer('pos', (j[None, :] - j[:, None]).float())
        causal = j[None, :] < j[:, None]
        causal = causal.clone()
        causal[0, 0] = True                          # sentinel attends to itself
        self.register_buffer('mask_in', torch.where(causal, 0.0, -1e9))
        self.register_buffer('mask_out', torch.where(j[None, :] <= j[:, None], 0.0, -1e9))

    def forward(self, a_tok, b_tok):
        """a_tok, b_tok: (B, N_SLOTS) long tensors of digit tokens. -> (B, N_SLOTS, 2)."""
        z = self.U[a_tok] + self.U[b_tok]                       # token embedding
        h1 = F.relu(z + self.b[0])                              # feed-forward features
        h2 = F.relu(z + self.b[1])
        c1 = self.w1 * h1 + self.w2 * h2                        # key / query feature
        c2 = h1                                                 # value feature
        q = c1 + self.bq
        s = q.unsqueeze(-1) * c1.unsqueeze(-2) + self.slope * self.pos
        c_in = torch.matmul(torch.softmax(s + self.mask_in, -1), c2.unsqueeze(-1)).squeeze(-1)
        c_out = torch.matmul(torch.softmax(s + self.mask_out, -1), c2.unsqueeze(-1)).squeeze(-1)
        y = z + self.wo[0] * c_in + self.wo[1] * c_out          # residual
        return 2.0 * y.unsqueeze(-1) * self.U - self.U * self.U  # tied readout


_WEIGHTS = {
    'U': [1.7415390014648438, 11.102082252502441],
    'b': [0.21342018246650696, -10.708646774291992],
    'w2': 1.6286931037902832,
    'bq': 161.74575805664062,
    'wo': [0.692265510559082, -1.1153851747512817],
}


def build_model():
    model = TinyAdder()
    with torch.no_grad():
        for name, value in _WEIGHTS.items():
            getattr(model, name).copy_(torch.tensor(value))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    meta = {
        'name': 'TinyAdder',
        'n_parameters': n_params,
        'architecture': 'decoder-only transformer, 1 layer, 1 head',
        'tokenization': 'binary digits, least significant first, 49 slots',
        'description': ('self-attention performs carry lookahead: each slot attends to '
                        'the nearest earlier non-propagating slot and reads its carry'),
        'max_operand': 99999999999999,
    }
    return model, meta


def _tokens(n):
    return [0] + [int(ch) for ch in format(n, '0%db' % N_BITS)[::-1]]


@torch.no_grad()
def add(model, a, b):
    """Exact sum of two integers in [0, 99_999_999_999_999], from one forward pass."""
    a_tok = torch.tensor([_tokens(int(a))], dtype=torch.long)
    b_tok = torch.tensor([_tokens(int(b))], dtype=torch.long)
    logits = model(a_tok, b_tok)                      # (1, N_SLOTS, 2)
    digits = logits[0, 1:].argmax(-1).tolist()        # output tokens, LSB first
    return int(''.join(str(d) for d in reversed(digits)), 2)


@torch.no_grad()
def add_batch(model, pairs):
    """Vectorised convenience wrapper; same forward pass, one row per pair."""
    a_tok = torch.tensor([_tokens(int(a)) for a, _ in pairs], dtype=torch.long)
    b_tok = torch.tensor([_tokens(int(b)) for _, b in pairs], dtype=torch.long)
    digits = model(a_tok, b_tok)[:, 1:].argmax(-1).tolist()
    return [int(''.join(str(d) for d in reversed(row)), 2) for row in digits]
