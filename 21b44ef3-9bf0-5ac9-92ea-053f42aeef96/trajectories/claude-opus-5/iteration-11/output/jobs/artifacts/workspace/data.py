"""On-GPU sampler for addition problems, with a hash-bucketed held-out split.

Every operand is sampled at full digit width (both most-significant digits are in
1..9), matching how the task is graded.  Places are sampled either uniformly or
from a per-place "carry class" so that carry chains of every length show up often:

    low          a_i + b_i <= 8   kills an incoming carry
    transparent  a_i + b_i == 9   passes an incoming carry through
    high         a_i + b_i >= 10  generates a carry

One in sixteen (a, b) pairs is reserved as held-out by hashing the pair, so
reported accuracy is always on pairs the optimiser never saw.
"""

import torch

LOW, TRANSPARENT, HIGH = 0, 1, 2


def _pair_tables(device):
    """tables[constrained][class] -> int64 [k, 2] of all valid (a, b) digit pairs."""
    tables = {}
    for constrained in (0, 1):
        lo = 1 if constrained else 0
        for cls in (LOW, TRANSPARENT, HIGH):
            pairs = []
            for a in range(lo, 10):
                for b in range(lo, 10):
                    s = a + b
                    if (cls == LOW and s <= 8) or (cls == TRANSPARENT and s == 9) \
                            or (cls == HIGH and s >= 10):
                        pairs.append((a, b))
            tables[(constrained, cls)] = torch.tensor(pairs, dtype=torch.int64,
                                                      device=device)
    return tables


class Sampler:
    def __init__(self, device, holdout_mod=16):
        self.device = device
        self.tab = _pair_tables(device)
        self.holdout_mod = holdout_mod

    # -- held-out split ----------------------------------------------------
    def bucket(self, a, b):
        z = a * 2620678134532503325 + b * 6364136223846793005
        z = z ^ (z >> 30)
        z = z * 6364136223846793005
        z = z ^ (z >> 27)
        z = z * 2620678134532503325
        z = z ^ (z >> 31)
        return z % self.holdout_mod

    # -- sampling ----------------------------------------------------------
    def _from_classes(self, cls):
        """cls: int64 [B, n] of carry classes -> digits a, b [B, n] (LSB first)."""
        B, n = cls.shape
        a = torch.empty(B, n, dtype=torch.int64, device=self.device)
        b = torch.empty(B, n, dtype=torch.int64, device=self.device)
        for constrained in (0, 1):
            # the most significant place (index n-1) must have both digits >= 1
            place = slice(n - 1, n) if constrained else slice(0, n - 1)
            sub = cls[:, place]
            if sub.numel() == 0:
                continue
            for c in (LOW, TRANSPARENT, HIGH):
                tbl = self.tab[(constrained, c)]
                m = sub == c
                k = int(m.sum())
                if k == 0:
                    continue
                pick = torch.randint(len(tbl), (k,), device=self.device)
                chosen = tbl[pick]
                aa = a[:, place]
                bb = b[:, place]
                aa[m] = chosen[:, 0]
                bb[m] = chosen[:, 1]
                a[:, place] = aa
                b[:, place] = bb
        return a, b

    def sample_digits(self, B, n, p_transparent=None):
        """Uniform digits if p_transparent is None, else class-structured."""
        if p_transparent is None:
            a = torch.randint(10, (B, n), device=self.device)
            b = torch.randint(10, (B, n), device=self.device)
            a[:, n - 1] = torch.randint(1, 10, (B,), device=self.device)
            b[:, n - 1] = torch.randint(1, 10, (B,), device=self.device)
            return a, b
        pt = p_transparent
        u = torch.rand(B, n, device=self.device)
        cls = torch.where(u < pt, torch.full_like(u, TRANSPARENT, dtype=torch.int64),
                          torch.where(u < pt + (1 - pt) / 2,
                                      torch.full_like(u, LOW, dtype=torch.int64),
                                      torch.full_like(u, HIGH, dtype=torch.int64)))
        return self._from_classes(cls)

    def batch(self, B, n, mix=(0.35, 0.25, 0.25, 0.15), held_out=False):
        """A batch of B problems with n places.

        mix = weights of (uniform digits, p_t=0.4, p_t=0.7, p_t=0.9).
        Returns ab [B, n+2, 2] int64, target [B, n+2] int64, loss mask [B, n+2].
        """
        specs = [None, 0.4, 0.7, 0.9]
        counts = [int(round(w * B)) for w in mix]
        counts[0] += B - sum(counts)
        parts_a, parts_b = [], []
        for c, spec in zip(counts, specs):
            if c <= 0:
                continue
            aa, bb = self.sample_digits(c, n, spec)
            parts_a.append(aa)
            parts_b.append(bb)
        a = torch.cat(parts_a, 0)
        b = torch.cat(parts_b, 0)
        return self.pack(a, b, held_out=held_out)

    def pack(self, a, b, held_out=None):
        """digits -> padded token/target tensors plus the loss mask."""
        B, n = a.shape
        pw = (10 ** torch.arange(n, device=self.device)).to(torch.int64)
        va = (a * pw).sum(1)
        vb = (b * pw).sum(1)
        vs = va + vb
        P = n + 2
        ab = torch.zeros(B, P, 2, dtype=torch.int64, device=self.device)
        ab[:, 1:n + 1, 0] = a
        ab[:, 1:n + 1, 1] = b
        tgt = torch.zeros(B, P, dtype=torch.int64, device=self.device)
        pw2 = (10 ** torch.arange(n + 1, device=self.device)).to(torch.int64)
        tgt[:, 1:n + 2] = (vs[:, None] // pw2[None, :]) % 10
        mask = torch.zeros(B, P, device=self.device)
        mask[:, 1:n + 2] = 1.0                       # positions 1..n+1 carry answers
        if held_out is not None:
            bk = self.bucket(va, vb)
            keep = (bk == 0) if held_out else (bk != 0)
            mask = mask * keep[:, None].to(mask.dtype)
        return ab, tgt, mask
