"""On-GPU sampler of digit-pair addition problems with a held-out split."""
import torch

# digit pairs grouped by what the place does to a carry
_ABS = [(a, b) for a in range(10) for b in range(10) if a + b <= 8]
_TRA = [(a, b) for a in range(10) for b in range(10) if a + b == 9]
_GEN = [(a, b) for a in range(10) for b in range(10) if a + b >= 10]

HOLD_MOD = 16      # 1-in-16 of pair space is never trained on


class Sampler:
    def __init__(self, device):
        self.dev = device
        self.tab = {
            k: torch.tensor(v, dtype=torch.long, device=device)
            for k, v in (("abs", _ABS), ("tra", _TRA), ("gen", _GEN))
        }
        self.pw = None

    def _hash(self, a, b):
        """Deterministic bucket for a digit-pair problem (length independent)."""
        n = a.shape[1]
        if self.pw is None or self.pw.shape[0] < n:
            w = torch.arange(64, dtype=torch.long, device=self.dev)
            self.pw = (torch.tensor(1000003, device=self.dev) ** (w % 12)) % 1000000007
        w = self.pw[:n]
        h = ((a * 31 + b * 17 + 1) * w).sum(1) % 1000000007
        return (h * 2654435761) % HOLD_MOD

    def _raw(self, B, n, kind, p_tra=0.4):
        d = self.dev
        if kind == "uniform":
            a = torch.randint(0, 10, (B, n), device=d)
            b = torch.randint(0, 10, (B, n), device=d)
        elif kind == "tra":
            a = torch.randint(0, 10, (B, n), device=d)
            b = torch.randint(0, 10, (B, n), device=d)
            t = self.tab["tra"][torch.randint(0, 10, (B, n), device=d)]
            m = torch.rand(B, n, device=d) < p_tra
            a = torch.where(m, t[..., 0], a)
            b = torch.where(m, t[..., 1], b)
        else:  # structured chains: per-place absorb / transparent / generate
            r = torch.rand(B, n, device=d)
            a = torch.empty(B, n, dtype=torch.long, device=d)
            b = torch.empty(B, n, dtype=torch.long, device=d)
            for name, lo, hi in (("abs", 0.0, 0.25), ("tra", 0.25, 0.75), ("gen", 0.75, 1.0)):
                m = (r >= lo) & (r < hi)
                t = self.tab[name]
                pick = t[torch.randint(0, t.shape[0], (B, n), device=d)]
                a = torch.where(m, pick[..., 0], a)
                b = torch.where(m, pick[..., 1], b)
        # every graded operand carries the full digit width
        top = torch.randint(1, 10, (B, 2), device=d)
        a[:, n - 1] = torch.where(a[:, n - 1] == 0, top[:, 0], a[:, n - 1])
        b[:, n - 1] = torch.where(b[:, n - 1] == 0, top[:, 1], b[:, n - 1])
        return a, b

    def batch(self, B, n=8, split="train", mix=(0.35, 0.40, 0.25), p_tra=0.4):
        """Returns (a, b, target) with a,b of shape (B, n+2) and target (B, n+1)."""
        parts_a, parts_b = [], []
        want = [int(B * mix[0]), int(B * mix[1])]
        want.append(B - want[0] - want[1])
        for cnt, kind in zip(want, ("uniform", "tra", "chain")):
            if cnt <= 0:
                continue
            need, ga, gb = cnt, [], []
            for _ in range(8):
                if need <= 0:
                    break
                a, b = self._raw(int(need * 1.3) + 8, n, kind, p_tra)
                keep = (self._hash(a, b) == 0) if split == "val" else (self._hash(a, b) != 0)
                a, b = a[keep][:need], b[keep][:need]
                ga.append(a)
                gb.append(b)
                need -= a.shape[0]
            parts_a.append(torch.cat(ga)[:cnt])
            parts_b.append(torch.cat(gb)[:cnt])
        a = torch.cat(parts_a)
        b = torch.cat(parts_b)
        return pad_and_label(a, b)


def pad_and_label(a, b):
    """Pad with a (0,0) place at each end and compute the answer digits."""
    B, n = a.shape
    z = torch.zeros(B, 1, dtype=torch.long, device=a.device)
    ap = torch.cat([z, a, z], 1)
    bp = torch.cat([z, b, z], 1)
    s = a + b
    carry = torch.zeros(B, dtype=torch.long, device=a.device)
    outs = []
    for i in range(n):
        t = s[:, i] + carry
        outs.append(t % 10)
        carry = t // 10
    outs.append(carry)
    return ap, bp, torch.stack(outs, 1)      # target (B, n+1) for positions 1..n+1
