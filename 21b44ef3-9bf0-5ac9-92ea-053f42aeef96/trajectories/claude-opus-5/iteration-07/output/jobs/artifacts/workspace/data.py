"""On-GPU sampler for decimal addition, with a hash-bucketed held-out split.

The number of decimal places is a parameter.  The model uses only *relative*
positions, so training on longer operands than the eight that are graded is
legitimate and much harder: propagating a carry across a long run of
carry-transparent places forces the attention to actually select a distant
position rather than lean on a fixed decaying pattern.

Operands are always full width: the most significant digit of each operand is
in 1..9.
"""
import torch

NPLACE = 8


def pow10(n, device):
    return torch.tensor([10 ** i for i in range(n)], dtype=torch.long,
                        device=device)


def to_int(d):
    """(..., n) LSB-first digits -> integer tensor (n <= 18)."""
    return (d * pow10(d.shape[-1], d.device)).sum(-1)


def bucket(a, b):
    """Deterministic 4-bit hash of an operand pair (the held-out split key)."""
    m = 0x3FFFFFFFFFFFFFFF
    h = (a * 2654435761 + b * 2246822519) & m
    h = (h ^ (h >> 15)) & m
    h = (h * 2654435761) & m
    h = (h ^ (h >> 13)) & m
    return h & 15


def _fix_msb(ad, bd, gen):
    n, dev = ad.shape[0], ad.device
    ad[:, -1] = torch.randint(1, 10, (n,), device=dev, generator=gen)
    bd[:, -1] = torch.randint(1, 10, (n,), device=dev, generator=gen)
    return ad, bd


def _uniform(n, np_, dev, gen):
    ad = torch.randint(0, 10, (n, np_), device=dev, generator=gen)
    bd = torch.randint(0, 10, (n, np_), device=dev, generator=gen)
    return _fix_msb(ad, bd, gen)


def _transparent(n, np_, dev, gen, p=0.45):
    """Each place independently forced to a+b == 9 (carry-transparent) w.p. p."""
    ad, bd = _uniform(n, np_, dev, gen)
    m = torch.rand(n, np_, device=dev, generator=gen) < p
    bd = torch.where(m, 9 - ad, bd)
    hi = torch.randint(1, 9, (n,), device=dev, generator=gen)
    ad[:, -1] = torch.where(m[:, -1], hi, ad[:, -1])
    bd[:, -1] = torch.where(m[:, -1], 9 - hi, bd[:, -1])
    return ad, bd


def _chain(n, np_, dev, gen):
    """One generating place followed by an unbroken run of transparent places.

    The run reaches the top of the number, so the carry has to travel up to
    ``np_ - 1`` places.
    """
    ad, bd = _uniform(n, np_, dev, gen)
    g = torch.randint(0, np_, (n,), device=dev, generator=gen)
    pos = torch.arange(np_, device=dev)[None, :]
    gcol = g[:, None]

    above = pos > gcol
    bd = torch.where(above, 9 - ad, bd)

    lo = torch.randint(1, 10, (n,), device=dev, generator=gen)
    span = torch.rand(n, device=dev, generator=gen)
    hi = 10 - lo + (span * lo.float()).long().clamp(max=9)
    at = pos == gcol
    ad = torch.where(at, lo[:, None], ad)
    bd = torch.where(at, hi.clamp(0, 9)[:, None], bd)

    # keep both most-significant digits in 1..9
    h1 = torch.randint(1, 9, (n,), device=dev, generator=gen)
    top_t = above[:, -1]
    ad[:, -1] = torch.where(top_t, h1, ad[:, -1])
    bd[:, -1] = torch.where(top_t, 9 - h1, bd[:, -1])
    top_g = at[:, -1]
    ad[:, -1] = torch.where(top_g, ad[:, -1].clamp(min=1), ad[:, -1])
    bd[:, -1] = torch.where(top_g, bd[:, -1].clamp(min=1), bd[:, -1])
    return ad, bd


def _raw(n, np_, dev, gen, mix):
    ns = [int(n * mix[0]), int(n * mix[1])]
    ns.append(n - ns[0] - ns[1])
    parts = []
    for k, fn in zip(ns, (_uniform, _transparent, _chain)):
        if k:
            parts.append(fn(k, np_, dev, gen))
    ad = torch.cat([p[0] for p in parts], 0)
    bd = torch.cat([p[1] for p in parts], 0)
    return ad, bd


def sample(n, dev, gen, mix=(0.3, 0.4, 0.3), split="train", tries=6, nplace=None):
    """Return (tok, target): tok (m, nplace+2, 2), target (m, nplace+2)."""
    np_ = nplace or NPLACE
    ad, bd = _raw(n, np_, dev, gen, mix)
    if split != "all":
        want_zero = (split == "eval")
        if want_zero:
            tries = max(tries, 250)      # bucket 0 is only 1/16 of the space
        for _ in range(tries):           # no host sync: fixed number of re-rolls
            bad = ((bucket(to_int(ad), to_int(bd)) == 0) != want_zero)[:, None]
            nad, nbd = _raw(n, np_, dev, gen, mix)
            ad = torch.where(bad, nad, ad)
            bd = torch.where(bad, nbd, bd)
        keep = (bucket(to_int(ad), to_int(bd)) == 0) == want_zero
        ad, bd = ad[keep], bd[keep]
    return pack(ad, bd)


def pack(ad, bd):
    """Digit arrays -> (tok, target) in the model's position layout."""
    n, np_ = ad.shape
    dev = ad.device
    P = np_ + 2
    tok = torch.zeros(n, P, 2, dtype=torch.long, device=dev)
    tok[:, 1:P - 1, 0] = ad
    tok[:, 1:P - 1, 1] = bd
    tgt = torch.zeros(n, P, dtype=torch.long, device=dev)
    carry = torch.zeros(n, dtype=torch.long, device=dev)
    for i in range(np_):                       # digit-wise: no integer overflow
        s = ad[:, i] + bd[:, i] + carry
        tgt[:, i + 1] = s % 10
        carry = s // 10
    tgt[:, P - 1] = carry
    return tok, tgt


class Stream:
    """Buffered batch source: generates `chunk` rows at a time on the GPU."""

    def __init__(self, dev, gen, bs, mix=(0.3, 0.4, 0.3), split="train",
                 chunk=1 << 18, nplace=None):
        self.dev, self.gen, self.bs, self.mix, self.split = dev, gen, bs, mix, split
        self.nplace = nplace or NPLACE
        self.chunk = max(chunk, bs * 8)
        self.tok = self.tgt = None
        self.i = 0

    def _fill(self):
        self.tok, self.tgt = sample(self.chunk, self.dev, self.gen, mix=self.mix,
                                    split=self.split, tries=3, nplace=self.nplace)
        self.i = 0

    def next(self):
        if self.tok is None or self.i + self.bs > self.tok.shape[0]:
            self._fill()
        s = self.i
        self.i += self.bs
        return self.tok[s:s + self.bs], self.tgt[s:s + self.bs]


def all_carry_patterns(dev, nplace=None):
    """One pair per assignment of {absorb, transparent, generate} to each place."""
    np_ = nplace or NPLACE
    n = 3 ** np_
    idx = torch.arange(n, device=dev)
    ad = torch.zeros(n, np_, dtype=torch.long, device=dev)
    bd = torch.zeros(n, np_, dtype=torch.long, device=dev)
    for i in range(np_):
        kind = (idx // (3 ** i)) % 3
        top = (i == np_ - 1)
        a_ab = torch.full_like(idx, 1 if top else 0)
        b_ab = torch.full_like(idx, 1 if top else 0)
        a_tr = torch.full_like(idx, 4 if top else 0)
        b_tr = torch.full_like(idx, 5 if top else 9)
        a_ge = torch.full_like(idx, 9)
        b_ge = torch.full_like(idx, 9)
        ad[:, i] = torch.where(kind == 0, a_ab, torch.where(kind == 1, a_tr, a_ge))
        bd[:, i] = torch.where(kind == 0, b_ab, torch.where(kind == 1, b_tr, b_ge))
    return pack(ad, bd)


def edge_cases(dev):
    pairs = [
        (10000000, 10000000), (99999999, 99999999), (10000000, 99999999),
        (19999999, 10000001), (12345678, 87654321), (11111111, 88888889),
        (99999999, 10000001), (50000000, 50000000), (10000001, 19999999),
        (99999998, 10000002), (55555555, 44444445), (10000000, 89999999),
        (98765432, 12345678), (11111111, 11111111), (90000000, 19999999),
        (12345678, 12345678), (99999999, 11111111), (10101010, 89898989),
        (45454545, 54545455), (19999998, 10000002), (10000000, 19999999),
    ]
    ad = torch.tensor([[(a // 10 ** i) % 10 for i in range(8)]
                       for a, _ in pairs], device=dev)
    bd = torch.tensor([[(b // 10 ** i) % 10 for i in range(8)]
                       for _, b in pairs], device=dev)
    return pack(ad, bd)
