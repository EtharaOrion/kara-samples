"""Ensemble model + data generation for the tiny addition transformer.

Architecture (per model, 26 params before folding):
  slots: 0 = phantom (0,0) sentinel, 1..15 = digit positions 0..14, LSB first.
  embedding  z_i = U[a_i] + U[b_i]                                     (U: 10)
  features   h  = relu(z*W1a + b1a)         width 3                    (6)
             c1 = h.W2a0   (attention key/query feature)               (3)
             c2 = h.W2a1   (attention value  feature)                  (3)
  attention  score(i,j) = (c1_i + bq)*c1_j + slope*(j-i)               (2)
             head A: strictly causal (j<i)   -> c_in  (carry into slot i)
             head B: causal with self (j<=i) -> c_out (carry out of slot i)
             both heads share q,k,v and differ only in the mask.
  output     y = z + wo1*c_in + wo2*c_out                              (2)
  readout    logit_d = 2*y*U[d] - U[d]^2       (tied to U, weightless)
"""
import torch
import torch.nn.functional as F

NSLOT = 16          # 1 sentinel + 15 digit positions
NPOS = 15           # output digits (14-digit operands -> 15-digit sum)
POW10 = torch.tensor([10 ** i for i in range(NPOS)], dtype=torch.long)

# relative position j-i  (<=0 under causal mask)
_ii = torch.arange(NSLOT).view(-1, 1)
_jj = torch.arange(NSLOT).view(1, -1)
REL = (_jj - _ii).float()
MASK_A = (_jj < _ii)                      # strictly causal
MASK_A[0, 0] = True                       # sentinel self-attends (no valid j)
MASK_B = (_jj <= _ii)                     # causal incl. self
NEG = -1e9


class Ens(torch.nn.Module):
    """M independent copies trained in parallel along a leading model dim."""

    def __init__(self, M, seed=0, fold_sign=None, drop=None, width=3, untied=False, fix_slope=None):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        P = torch.nn.Parameter
        r = lambda *s: torch.randn(*s, generator=g)
        self.M = M
        self.W = width
        self.fold_sign = fold_sign          # if set: W1a is fixed to these signs
        self.drop = drop or ()              # names of pruned params
        self.U = P(r(M, 10) * 0.4)
        self.untied = untied
        if untied:                      # separate output centres; folded back onto U later
            self.V = P(r(M, 10) * 0.4)
        if fold_sign is None:
            self.W1a = P(r(M, width) * 1.0)
        else:
            self.register_buffer('W1a_fixed', torch.tensor(fold_sign).float().view(1, len(fold_sign)).repeat(M, 1))
        self.b1a = P(r(M, width) * 1.0)
        self.W2a0 = P(r(M, width) * 0.7)
        if 'W2a1_0' in self.drop:
            self.W2a1p = P(r(M, width - 1) * 0.7)
        else:
            self.W2a1 = P(r(M, width) * 0.7)
        self.bq = P(r(M, 1) * 1.0)
        self.fix_slope = fix_slope
        if fix_slope is None:
            self.slope = P(2.0 + 7.0 * torch.rand(M, 1, generator=g))
        else:   # fixed ALiBi-style decay (not learned, as in the original ALiBi)
            self.register_buffer('slope_fixed', torch.full((M, 1), float(fix_slope)))
        self.wo = P(r(M, 2) * 0.7)
        self.register_buffer('rel', REL.clone())
        self.register_buffer('mA', torch.where(MASK_A, 0.0, NEG))
        self.register_buffer('mB', torch.where(MASK_B, 0.0, NEG))

    def w1a(self):
        return self.W1a if self.fold_sign is None else self.W1a_fixed

    def slope_(self):
        return self.slope if self.fix_slope is None else self.slope_fixed

    def w2a1(self):
        if 'W2a1_0' in self.drop:
            z = torch.zeros(self.M, 1, device=self.W2a1p.device, dtype=self.W2a1p.dtype)
            return torch.cat([z, self.W2a1p], 1)
        return self.W2a1

    def zh(self, oh):
        """embedding and hidden activations, for diagnostics / unit resurrection"""
        M = self.M
        z = torch.einsum('bsd,md->mbs', oh, self.U)
        h = F.relu(z.unsqueeze(-1) * self.w1a().view(M, 1, 1, self.W)
                   + self.b1a.view(M, 1, 1, self.W))
        return z, h

    def forward(self, oh, noise=0.0):
        """oh: (B,NSLOT,10) float = onehot(a)+onehot(b).  returns logits (M,B,NPOS,10)."""
        M = self.M
        z = torch.einsum('bsd,md->mbs', oh, self.U)                       # (M,B,S)
        h = F.relu(z.unsqueeze(-1) * self.w1a().view(M, 1, 1, self.W)
                   + self.b1a.view(M, 1, 1, self.W))                            # (M,B,S,3)
        c1 = (h * self.W2a0.view(M, 1, 1, self.W)).sum(-1)
        c2 = (h * self.w2a1().view(M, 1, 1, self.W)).sum(-1)
        q = c1 + self.bq.view(M, 1, 1)
        sc = q.unsqueeze(-1) * c1.unsqueeze(-2) + self.slope_().view(M, 1, 1, 1) * self.rel
        if noise > 0:
            sc = sc + noise * torch.randn_like(sc)
        v = c2.unsqueeze(-2)                                               # (M,B,1,S)
        c_in = (torch.softmax(sc + self.mA, -1) * v).sum(-1)
        c_out = (torch.softmax(sc + self.mB, -1) * v).sum(-1)
        y = z + self.wo[:, 0].view(M, 1, 1) * c_in + self.wo[:, 1].view(M, 1, 1) * c_out
        V = (self.V if self.untied else self.U).view(M, 1, 1, 10)
        logits = 2.0 * y.unsqueeze(-1) * V - V * V
        return logits[:, :, 1:, :]

    def nparams_single(self):
        n = 0
        for p in self.parameters():
            n += p.numel() // self.M
        return n


# ---------------------------------------------------------------- data
def _digits_uniform(B, dev, g):
    d = torch.randint(0, 10, (B, NPOS), device=dev, generator=g)
    d[:, NPOS - 1] = 0                       # operands are < 10^14
    return d


def sample_batch(B, dev, g):
    """Four regimes, B//4 each: uniform / mixed lengths / 0-9 skewed / forced propagate."""
    q = B // 4
    outs_a, outs_b = [], []

    # 1. uniform 14-digit
    outs_a.append(_digits_uniform(q, dev, g))
    outs_b.append(_digits_uniform(q, dev, g))

    # 2. independent random operand lengths
    def rand_len():
        d = _digits_uniform(q, dev, g)
        L = torch.randint(1, NPOS, (q, 1), device=dev, generator=g)
        return torch.where(torch.arange(NPOS, device=dev).view(1, -1) < L, d, torch.zeros_like(d))
    outs_a.append(rand_len())
    outs_b.append(rand_len())

    # 3. digits skewed toward 0 and 9
    def skew():
        d = _digits_uniform(q, dev, g)
        pick = torch.rand(q, NPOS, device=dev, generator=g)
        d = torch.where(pick < 0.3, torch.zeros_like(d), d)
        d = torch.where(pick > 0.7, torch.full_like(d, 9), d)
        d[:, NPOS - 1] = 0
        return d
    outs_a.append(skew())
    outs_b.append(skew())

    # 4. forced propagate chains: b_i = 9 - a_i with per-sample rate
    a4 = _digits_uniform(q, dev, g)
    b4 = _digits_uniform(q, dev, g)
    rate = 0.3 + 0.7 * torch.rand(q, 1, device=dev, generator=g)
    m = torch.rand(q, NPOS, device=dev, generator=g) < rate
    b4 = torch.where(m, 9 - a4, b4)
    b4[:, NPOS - 1] = 0
    outs_a.append(a4)
    outs_b.append(b4)

    a = torch.cat(outs_a, 0)
    b = torch.cat(outs_b, 0)
    return a, b


def easy_batch(B, dev, g):
    """Carry-free pairs: a uniform, b uniform in [0, 9-a]. Teaches the digit code only."""
    a = _digits_uniform(B, dev, g)
    u = torch.rand(B, NPOS, device=dev, generator=g)
    b = (u * (10 - a).float()).long().clamp(max=9)
    b[:, NPOS - 1] = 0
    return a, b


def curric_batch(B, dev, g, p):
    """Uniform digits with the per-slot *propagate* rate (a_i+b_i==9) pinned to p.
    p small -> carry chains are short, so a near-positional lookahead suffices;
    raising p lengthens the chains the attention has to skip over."""
    a = _digits_uniform(B, dev, g)
    b = _digits_uniform(B, dev, g)
    pm = torch.rand(B, NPOS, device=dev, generator=g) < p
    b = torch.where(pm, 9 - a, b)
    clash = (~pm) & ((a + b) == 9)
    b = torch.where(clash, (b + 1) % 10, b)
    b[:, NPOS - 1] = 0
    return a, b


def stress_batch(B, dev, g, runlen=None):
    """Long forced-propagate chains at random offsets (worst case for carry lookahead)."""
    a = _digits_uniform(B, dev, g)
    b = _digits_uniform(B, dev, g)
    if runlen is None:
        runlen = torch.randint(0, NPOS, (B, 1), device=dev, generator=g)
    else:
        runlen = torch.full((B, 1), runlen, device=dev, dtype=torch.long)
    start = torch.randint(0, NPOS, (B, 1), device=dev, generator=g)
    idx = torch.arange(NPOS, device=dev).view(1, -1)
    m = (idx >= start) & (idx < start + runlen)
    b = torch.where(m, 9 - a, b)
    b[:, NPOS - 1] = 0
    a[:, NPOS - 1] = 0
    return a, b


def targets(a, b, dev):
    p = POW10.to(dev)
    s = (a * p).sum(-1) + (b * p).sum(-1)
    return (s.unsqueeze(-1) // p) % 10          # (B,NPOS)


def to_onehot(a, b, dev):
    B = a.shape[0]
    oh = torch.zeros(B, NSLOT, 10, device=dev)
    idx = torch.arange(NPOS, device=dev).view(1, -1).expand(B, -1)
    oh[:, 1:, :].scatter_add_(2, a.unsqueeze(-1), torch.ones(B, NPOS, 1, device=dev))
    oh[:, 1:, :].scatter_add_(2, b.unsqueeze(-1), torch.ones(B, NPOS, 1, device=dev))
    oh[:, 0, 0] = 2.0                            # sentinel pair (0,0)
    return oh
