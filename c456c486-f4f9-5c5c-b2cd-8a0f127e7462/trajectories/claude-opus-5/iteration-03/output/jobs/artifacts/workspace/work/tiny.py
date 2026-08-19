"""Ensemble-batched tiny adder transformer (training-side implementation).

Every parameter carries a leading model dimension M so that M independent
seeds/variants train concurrently in one process.  The tiny kernels are
launch-bound, so M seeds cost ~the same wall clock as one.

Sequence layout: 16 slots.  Slot 0 is a phantom (0,0) digit pair which acts as
the sentinel (it neither generates nor propagates a carry).  Slots 1..15 hold
digit i-1 of each operand, least-significant first; operands are zero padded to
15 digits.  The digit of the sum at position i-1 is read out of slot i.
"""
import math
import torch
import torch.nn.functional as F

NSLOT = 16       # default eval geometry: 1 sentinel + 15 digit positions
NDIG = 15


def pair_row_table(device="cpu"):
    """Unordered-pair token id for each ordered digit pair: (a,b) and (b,a) share a row."""
    a = torch.arange(10, device=device)
    lo = torch.minimum(a[:, None], a[None, :])
    hi = torch.maximum(a[:, None], a[None, :])
    # row index of (lo,hi) in the upper triangle, 55 rows
    return (lo * (21 - lo)) // 2 + (hi - lo)


class Cfg:
    def __init__(self, d_model=3, d_ff=4, head="linear", n_pair=55, use_qk_bias=True,
                 use_k_bias=True, use_b2=True):
        self.d_model = d_model
        self.d_ff = d_ff
        self.head = head            # 'linear' | 'centres' | 'tied'
        self.n_pair = n_pair        # 55 unordered-pair embedding rows (1 channel)
        self.use_qk_bias = use_qk_bias
        # bk is redundant: (q_i+bq)*bk is constant across j and cancels in the softmax row.
        self.use_k_bias = use_k_bias
        self.use_b2 = use_b2

    def __repr__(self):
        return f"Cfg(d={self.d_model},ff={self.d_ff},head={self.head},pair={self.n_pair},qkb={self.use_qk_bias},kb={self.use_k_bias},b2={self.use_b2})"


def param_count(cfg):
    d, f = cfg.d_model, cfg.d_ff
    n = 10 + cfg.n_pair                      # digit code U + pair table P
    n += 3 * d + 1                           # wq, wk, wv, slope
    n += d                                   # wo
    if cfg.use_qk_bias:
        n += 2 if cfg.use_k_bias else 1
    n += d * f + f + f * d                   # MLP
    if cfg.use_b2:
        n += d
    if cfg.head == "linear":
        n += d * 10 + 10
    elif cfg.head == "linear_nb":
        n += d * 10
    elif cfg.head == "centres":
        n += d + 10 + 1
    elif cfg.head == "tied":
        n += d + 1
    return n


def init_params(cfg, M, device, seed=0, dtype=torch.float32):
    g = torch.Generator(device="cpu").manual_seed(seed)

    def r(*shape, s=1.0):
        return (torch.randn(*shape, generator=g) * s).to(device=device, dtype=dtype)

    d, f = cfg.d_model, cfg.d_ff
    p = {}
    # digit code: roughly (d-4.5)/4.5 so that a+b==9 sits at 0, plus noise
    base = (torch.arange(10, dtype=torch.float32) - 4.5) / 4.5
    p["U"] = (base[None, :].repeat(M, 1) + torch.randn(M, 10, generator=g) * 0.05).to(device, dtype)
    p["P"] = r(M, cfg.n_pair, s=0.5)
    p["wq"] = r(M, d, s=0.8)
    p["wk"] = r(M, d, s=0.8)
    p["wv"] = r(M, d, s=0.8)
    p["wo"] = r(M, d, s=0.8)
    p["slope"] = torch.full((M,), 2.0, device=device, dtype=dtype)
    if cfg.use_qk_bias:
        p["bq"] = r(M, s=0.5)
        if cfg.use_k_bias:
            p["bk"] = r(M, s=0.5)
    p["W1"] = r(M, d, f, s=1.0 / math.sqrt(d))
    p["b1"] = r(M, f, s=0.1)
    p["W2"] = r(M, f, d, s=1.0 / math.sqrt(f))
    if cfg.use_b2:
        p["b2"] = r(M, d, s=0.1)
    if cfg.head in ("linear", "linear_nb"):
        p["Wr"] = r(M, d, 10, s=1.0 / math.sqrt(d))
        if cfg.head == "linear":
            p["br"] = r(M, 10, s=0.1)
    elif cfg.head == "centres":
        p["wr"] = r(M, d, s=0.8)
        p["C"] = (base[None, :].repeat(M, 1) + torch.randn(M, 10, generator=g) * 0.05).to(device, dtype)
        p["tau"] = torch.full((M,), 1.5, device=device, dtype=dtype)
    elif cfg.head == "tied":
        p["wr"] = r(M, d, s=0.8)
        p["tau"] = torch.full((M,), 1.5, device=device, dtype=dtype)
    for k in p:
        p[k] = p[k].clone().requires_grad_(True)
    return p


def make_masks(device, dtype=torch.float32, T=NSLOT, strict=True):
    """Strictly causal: the carry into slot i depends only on slots *before* i, so a
    slot must not be able to attend to itself (that would leak its own digit pair
    into its own carry).  Slot 0 self-attends purely to keep its softmax row finite."""
    idx = torch.arange(T, device=device)
    rel = (idx[None, :] - idx[:, None]).to(dtype)        # j - i, <= 0 in causal region
    if strict:
        causal = idx[None, :] < idx[:, None]
        causal[0, 0] = True
    else:
        causal = idx[None, :] <= idx[:, None]
    return rel, causal


def _proj(x, w):
    """[M,B,T,d] . [M,d] -> [M,B,T]  (broadcast+sum beats bmm at these tiny dims)."""
    return (x * w[:, None, None, :]).sum(-1)


def forward(p, cfg, a_dig, b_dig, pair_tab, rel, causal, noise=0.0, hard=False):
    """a_dig,b_dig: int64 [B, NSLOT] digit ids (slot 0 == 0).  Returns logits [M,B,NDIG,10]."""
    d = cfg.d_model
    B, T = a_dig.shape
    dt = p["U"].dtype
    # one-hot matmuls rather than index gathers: the gather backward is a
    # scatter-add onto 10/55 rows, which serialises on atomics and dominates the step.
    pr = pair_tab[a_dig, b_dig]                              # [B,NSLOT]
    oh_d = (F.one_hot(a_dig, 10) + F.one_hot(b_dig, 10)).to(dt).reshape(B * T, 10)
    oh_p = F.one_hot(pr, cfg.n_pair).to(dt).reshape(B * T, cfg.n_pair)
    c = (oh_d @ p["U"].t()).view(B, T, -1).permute(2, 0, 1)  # [M,B,NSLOT] compositional digit code
    e = (oh_p @ p["P"].t()).view(B, T, -1).permute(2, 0, 1)  # [M,B,NSLOT] pair-token embedding
    zeros = torch.zeros_like(c)
    chans = [c, e] + [zeros] * (d - 2)
    x = torch.stack(chans, dim=-1).contiguous()              # [M,B,NSLOT,d]

    q = _proj(x, p["wq"])
    k = _proj(x, p["wk"])
    if cfg.use_qk_bias:
        q = q + p["bq"][:, None, None]
        if cfg.use_k_bias:
            k = k + p["bk"][:, None, None]
    scores = q[:, :, :, None] * k[:, :, None, :]             # [M,B,i,j]
    scores = scores + p["slope"][:, None, None, None] * rel
    if isinstance(noise, torch.Tensor) or noise > 0.0:
        # Anti-analogue regularizer.  A carry-lookahead solution attends nearly
        # one-hot and is unharmed by score jitter; a solution that encodes the
        # carry in the *magnitude* of a diffuse softmax needs ~1e-14 precision in
        # the weights and cannot survive it.
        nz = noise[:, None, None, None] if isinstance(noise, torch.Tensor) else noise
        scores = scores + nz * torch.randn_like(scores)
    scores = scores.masked_fill(~causal, float("-inf"))
    att = torch.softmax(scores, dim=-1)
    if hard:
        # Straight-through top-1 attention.  Forward is exactly one-hot, so the
        # carry cannot be smuggled through the *magnitude* of a diffuse softmax;
        # backward keeps the softmax gradient.
        onehot = torch.zeros_like(att).scatter_(-1, att.argmax(-1, keepdim=True), 1.0)
        att = onehot.detach() + att - att.detach()
    v = _proj(x, p["wv"])
    o = (att * v[:, :, None, :]).sum(-1)
    x = x + o[..., None] * p["wo"][:, None, None, :]

    h = torch.relu((x[..., :, None] * p["W1"][:, None, None, :, :]).sum(-2) + p["b1"][:, None, None, :])
    x = x + (h[..., :, None] * p["W2"][:, None, None, :, :]).sum(-2)
    if cfg.use_b2:
        x = x + p["b2"][:, None, None, :]

    x = x[:, :, 1:, :]                                       # read slots 1..15
    if cfg.head in ("linear", "linear_nb"):
        logits = (x[..., :, None] * p["Wr"][:, None, None, :, :]).sum(-2)
        if cfg.head == "linear":
            logits = logits + p["br"][:, None, None, :]
    else:
        y = _proj(x, p["wr"])
        C = p["U"] if cfg.head == "tied" else p["C"]
        diff = y[..., None] - C[:, None, None, :]
        logits = -(p["tau"][:, None, None, None] ** 2) * diff * diff
    return logits


def ce_loss(logits, tgt):
    """Manual cross entropy: aten::nll_loss is single-block and costs ~50ms at this N."""
    logp = torch.log_softmax(logits, dim=-1)
    picked = logp.gather(-1, tgt[None, :, :, None].expand(logits.shape[0], -1, -1, 1))
    return -picked.mean()


# ----------------------------------------------------------------------------- data
def sample_batch(B, device, gen, regime_p=(0.3, 0.25, 0.2, 0.25), ndig=14):
    """Digit pairs for two ndig-digit operands.

    Returns a_dig,b_dig [B,ndig+2] int64 (slot 0 is the (0,0) sentinel, one
    leading pad slot carries the final carry) and target digits [B,ndig+1].
    """
    r = torch.rand(B, device=device, generator=gen)
    cum = torch.tensor(regime_p, device=device).cumsum(0)
    N = ndig

    a = torch.randint(0, 10, (B, N), device=device, generator=gen)
    b = torch.randint(0, 10, (B, N), device=device, generator=gen)

    # regime 2: independent random operand lengths (leading digits zeroed)
    pos = torch.arange(N, device=device)
    la = torch.randint(1, N + 1, (B, 1), device=device, generator=gen)
    lb = torch.randint(1, N + 1, (B, 1), device=device, generator=gen)
    a2 = torch.where(pos[None, :] < la, a, torch.zeros_like(a))
    b2 = torch.where(pos[None, :] < lb, b, torch.zeros_like(b))

    # regime 3: digits skewed toward 0 and 9
    sk = torch.rand(B, N, device=device, generator=gen)
    hi = torch.randint(0, 10, (B, N), device=device, generator=gen)
    a3 = torch.where(sk < 0.4, torch.zeros_like(a), torch.where(sk < 0.8, torch.full_like(a, 9), hi))
    sk2 = torch.rand(B, N, device=device, generator=gen)
    hi2 = torch.randint(0, 10, (B, N), device=device, generator=gen)
    b3 = torch.where(sk2 < 0.4, torch.zeros_like(b), torch.where(sk2 < 0.8, torch.full_like(b, 9), hi2))

    # regime 4: forced propagate chains  b_i = 9 - a_i at a per-sample rate
    rate = 0.3 + 0.7 * torch.rand(B, 1, device=device, generator=gen)
    m = torch.rand(B, N, device=device, generator=gen) < rate
    b4 = torch.where(m, 9 - a, b)

    sel = torch.zeros(B, dtype=torch.long, device=device)
    sel = torch.where(r >= cum[0], torch.ones_like(sel), sel)
    sel = torch.where(r >= cum[1], torch.full_like(sel, 2), sel)
    sel = torch.where(r >= cum[2], torch.full_like(sel, 3), sel)
    s = sel[:, None]
    A = torch.where(s == 0, a, torch.where(s == 1, a2, torch.where(s == 2, a3, a)))
    Bb = torch.where(s == 0, b, torch.where(s == 1, b2, torch.where(s == 2, b3, b4)))
    return finish(A, Bb, device)


def finish(A, Bb, device):
    """A,Bb: [B,N] digit tensors (LSB first) -> padded inputs and target sum digits."""
    B, N = A.shape
    z = torch.zeros(B, 1, dtype=A.dtype, device=device)
    a_dig = torch.cat([z, A, z], dim=1)          # sentinel + N digits + carry-out slot
    b_dig = torch.cat([z, Bb, z], dim=1)
    tgt = torch.empty(B, N + 1, dtype=torch.long, device=device)
    carry = torch.zeros(B, dtype=torch.long, device=device)
    for i in range(N + 1):
        s = carry + (A[:, i] + Bb[:, i] if i < N else 0)
        tgt[:, i] = s % 10
        carry = s // 10
    return a_dig, b_dig, tgt


class Stream:
    """Chunked sampler: generating one big chunk amortises the ~15ms datagen cost."""

    def __init__(self, bs, device, seed, chunk_mult=32, ndigs=(14,), **kw):
        self.bs, self.device, self.kw = bs, device, kw
        self.ndigs = list(ndigs)
        self.n = chunk_mult
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.i = self.n

    def next(self):
        if self.i >= self.n:
            nd = self.ndigs[int(torch.randint(len(self.ndigs), (1,), generator=self.gen,
                                              device=self.device).item())]
            self.buf = sample_batch(self.bs * self.n, self.device, self.gen, ndig=nd, **self.kw)
            self.i = 0
        s = slice(self.i * self.bs, (self.i + 1) * self.bs)
        self.i += 1
        return self.buf[0][s], self.buf[1][s], self.buf[2][s]
