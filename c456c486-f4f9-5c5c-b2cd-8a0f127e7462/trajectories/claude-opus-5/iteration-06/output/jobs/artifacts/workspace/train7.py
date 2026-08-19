"""Population trainer for the 7-parameter binary carry-lookahead transformer.

Everything that generates labelled data or performs optimisation lives here.
`export()` writes the trained scalars into /workspace/submission.py.

Model (population form: every parameter carries a leading model dimension M so
that M independent random inits train simultaneously):

    z    = U[a_tok] + U[b_tok]
    c1   = W1*z + w2*relu(z + b)  (W1 = -1, fixed gauge; one ReLU feature unit)
    c2   = z                      (value projection = identity on the residual)
    s_ij = (c1_i + bq)*c1_j + SLOPE*(j-i)      (SLOPE = 4.0, fixed buffer)
    cin  = softmax_{j<i}(s) @ c2 ; cout = softmax_{j<=i}(s) @ c2
    y    = z + wo[0]*cin + wo[1]*cout
    logit(d) = 2*y*U[d] - U[d]^2
"""
import argparse, math, os, time
import torch
import torch.nn.functional as F

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
NBITS = 48                 # bit slots (operands < 10**14 < 2**47, sum < 2**48)
L = NBITS + 1              # + phantom sentinel at slot 0
SLOPE = 4.0
W1 = -1.0
MAXOP = 10 ** 14
NEG = -1e9

POW = (1 << torch.arange(NBITS, device=DEV, dtype=torch.int64))
_j = torch.arange(L, device=DEV)
POS = SLOPE * (_j[None, :] - _j[:, None]).float()          # (L,L)  slope*(j-i)
_causal_strict = (_j[None, :] < _j[:, None])
_causal_strict[0, 0] = True                                # slot 0 self-attends
MASK_IN = torch.where(_causal_strict, 0.0, NEG)
MASK_OUT = torch.where(_j[None, :] <= _j[:, None], 0.0, NEG)

PNAMES = ('U', 'b', 'w2', 'bq', 'wo')
PSHAPE = {'U': (2,), 'b': (), 'w2': (), 'bq': (), 'wo': (2,)}


# ---------------------------------------------------------------- model ----
def forward(p, atok, btok, ret_attn=False, pos=None):
    """p: dict of (M,...) tensors. atok/btok: (B,L) int64. -> logits (M,B,L,2)."""
    U, b, w2, bq, wo = p['U'], p['b'], p['w2'], p['bq'], p['wo']
    z = U[:, atok] + U[:, btok]                              # (M,B,L)
    c1 = W1 * z + w2[:, None, None] * F.relu(z + b[:, None, None])
    c2 = z
    q = c1 + bq[:, None, None]
    s = q.unsqueeze(-1) * c1.unsqueeze(-2) + (POS if pos is None else pos)
    a_in = torch.softmax(s + MASK_IN, -1)
    cin = torch.matmul(a_in, c2.unsqueeze(-1)).squeeze(-1)
    cout = torch.matmul(torch.softmax(s + MASK_OUT, -1), c2.unsqueeze(-1)).squeeze(-1)
    y = z + wo[:, 0, None, None] * cin + wo[:, 1, None, None] * cout
    lg = 2.0 * y.unsqueeze(-1) * U[:, None, None, :] - (U * U)[:, None, None, :]
    if ret_attn:
        return lg, a_in
    return lg


def loss_and_acc(p, atok, btok, tgt, ent_w=0.0, pos=None):
    """tgt: (B,NBITS) int64. Returns per-model loss (M,) and exact-match acc (M,)."""
    if ent_w > 0.0:
        lg, att = forward(p, atok, btok, ret_attn=True, pos=pos)
        ent = -(att.clamp_min(1e-9).log() * att).sum(-1)[:, :, 1:].mean(-1).mean(-1)
        lg = lg[:, :, 1:, :]
    else:
        lg = forward(p, atok, btok, pos=pos)[:, :, 1:, :]    # (M,B,NBITS,2)
    M, B = lg.shape[0], lg.shape[1]
    t = tgt.unsqueeze(0).expand(M, B, NBITS)
    ls = F.cross_entropy(lg.reshape(-1, 2), t.reshape(-1), reduction='none').view(M, B, NBITS)
    ok = (lg.argmax(-1) == t).all(-1).float().mean(1)
    out = ls.mean(-1).mean(-1)
    if ent_w > 0.0:
        out = out + ent_w * ent
    return out, ok


# ----------------------------------------------------------------- data ----
def bits_of(x):
    return ((x.unsqueeze(-1) >> torch.arange(NBITS, device=DEV)) & 1)


def to_tokens(a, b):
    """a,b: (B,) int64 -> atok,btok (B,L) int64 with sentinel, and target bits."""
    ab, bb = bits_of(a), bits_of(b)
    zero = torch.zeros_like(ab[:, :1])
    atok = torch.cat([zero, ab], 1)
    btok = torch.cat([zero, bb], 1)
    return atok, btok, bits_of(a + b)


def sample(bs, pmax, gen):
    """Mixed regimes. `pmax` caps the per-slot propagate rate (curriculum)."""
    r = torch.rand((), generator=gen, device=DEV).item()
    if r < 0.25:                                             # true eval distribution
        a = torch.randint(0, MAXOP, (bs,), generator=gen, device=DEV)
        b = torch.randint(0, MAXOP, (bs,), generator=gen, device=DEV)
        if pmax < 1.0:                                       # early curriculum: damp carries
            keep = torch.rand(bs, NBITS, generator=gen, device=DEV) < pmax
            ab, bb = bits_of(a), bits_of(b)
            prop = ab != bb
            bb = torch.where(prop & ~keep, ab, bb)
            a, b = (ab * POW).sum(1), (bb * POW).sum(1)
        return to_tokens(a, b)

    # per-slot propagate rate, per sample
    if r < 0.55:
        p = torch.rand(bs, 1, generator=gen, device=DEV) * pmax
    elif r < 0.8:
        p = torch.full((bs, 1), pmax, device=DEV)
    else:                                                    # forced contiguous runs
        p = torch.rand(bs, 1, generator=gen, device=DEV) * 0.5 * pmax
    prop = torch.rand(bs, NBITS, generator=gen, device=DEV) < p
    if r >= 0.8:
        st = torch.randint(0, NBITS, (bs, 1), generator=gen, device=DEV)
        ln = (torch.rand(bs, 1, generator=gen, device=DEV) * NBITS * pmax).long() + 1
        idx = torch.arange(NBITS, device=DEV)[None, :]
        prop |= (idx >= st) & (idx < st + ln)
    one = torch.randint(0, 2, (bs, NBITS), generator=gen, device=DEV)
    eq = torch.randint(0, 2, (bs, NBITS), generator=gen, device=DEV)
    ab = torch.where(prop, one, eq)
    bb = torch.where(prop, 1 - one, eq)
    lim = torch.randint(0, 2, (bs, 1), generator=gen, device=DEV).bool()   # some short operands
    cut = torch.randint(1, NBITS + 1, (bs, 1), generator=gen, device=DEV)
    idx = torch.arange(NBITS, device=DEV)[None, :]
    keep = (~lim) | (idx < cut)
    ab, bb = ab * keep, bb * keep
    ab[:, NBITS - 1] = 0                                     # operands stay < 2**47
    bb[:, NBITS - 1] = 0
    return to_tokens((ab * POW).sum(1), (bb * POW).sum(1))


def eval_batch(bs, gen, stress=False):
    if not stress:
        a = torch.randint(0, MAXOP, (bs,), generator=gen, device=DEV)
        b = torch.randint(0, MAXOP, (bs,), generator=gen, device=DEV)
        return to_tokens(a, b)
    return hard_batch(bs, gen)


def hard_batch(bs, gen):
    """One batch mixing every stress regime, so the metric is not regime-noisy."""
    n = bs // 4
    ab, bb = [], []
    w = POW
    for p in (0.5, 0.9, 1.0):
        pr = torch.rand(n, NBITS, generator=gen, device=DEV) < p
        one = torch.randint(0, 2, (n, NBITS), generator=gen, device=DEV)
        eq = torch.randint(0, 2, (n, NBITS), generator=gen, device=DEV)
        ab.append(torch.where(pr, one, eq)); bb.append(torch.where(pr, 1 - one, eq))
    # forced runs of a random length at a random offset
    pr = torch.rand(n, NBITS, generator=gen, device=DEV) < 0.5
    one = torch.randint(0, 2, (n, NBITS), generator=gen, device=DEV)
    eq = torch.randint(0, 2, (n, NBITS), generator=gen, device=DEV)
    a4 = torch.where(pr, one, eq); b4 = torch.where(pr, 1 - one, eq)
    st = torch.randint(0, NBITS, (n, 1), generator=gen, device=DEV)
    ln = torch.randint(1, NBITS + 1, (n, 1), generator=gen, device=DEV)
    idx = torch.arange(NBITS, device=DEV)[None, :]
    inrun = (idx >= st) & (idx < st + ln)
    b4 = torch.where(inrun, 1 - a4, b4)
    ab.append(a4); bb.append(b4)
    A = torch.cat(ab, 0); B = torch.cat(bb, 0)
    A[:, NBITS - 1] = 0; B[:, NBITS - 1] = 0
    return to_tokens((A * w).sum(1), (B * w).sum(1))


# ------------------------------------------------------------- training ----
def init_params(M, gen):
    U = torch.randn(M, 2, generator=gen, device=DEV)
    zs = torch.stack([2 * U[:, 0], U[:, 0] + U[:, 1], 2 * U[:, 1]], 1)
    lo, hi = zs.min(1).values, zs.max(1).values
    u = torch.rand(M, generator=gen, device=DEV)
    b = -(lo + u * (hi - lo))                                # keep the ReLU kink alive
    p = {'U': U, 'b': b,
         'w2': torch.randn(M, generator=gen, device=DEV),
         'bq': torch.randn(M, generator=gen, device=DEV).abs() * 4.0 + 1.0,
         'wo': torch.randn(M, 2, generator=gen, device=DEV)}
    return {k: v.contiguous().requires_grad_(True) for k, v in p.items()}


def reinit_cells(p, opt, idx, gen):
    if idx.numel() == 0:
        return
    fresh = init_params(idx.numel(), gen)
    with torch.no_grad():
        for k in PNAMES:
            p[k][idx] = fresh[k].detach()
            st = opt.state.get(p[k])
            if st:
                st['exp_avg'][idx] = 0
                st['exp_avg_sq'][idx] = 0


def resurrect(p, gen, atok, btok):
    """Re-place dead ReLU thresholds (unit always-on or always-off) inside the z range."""
    with torch.no_grad():
        U = p['U']
        z = U[:, atok] + U[:, btok]
        lo = z.amin(dim=(1, 2)); hi = z.amax(dim=(1, 2))
        act = (z + p['b'][:, None, None] > 0).float().mean(dim=(1, 2))
        dead = (act < 1e-4) | (act > 1 - 1e-4)
        if dead.any():
            u = torch.rand(int(dead.sum()), generator=gen, device=DEV)
            p['b'][dead] = -(lo[dead] + u * (hi - lo)[dead])
    return


def train(M=192, steps=30000, bs=192, lr=3e-2, seed=0, warm=None, log_every=500,
          polish=False, jitter=0.02, ramp=0.55, ent=0.0, logjit=0.0, src=0, hop_every=0, hop_frac=0.75, hop_sig=1.5, sjit=1.0,
          snap_path=None):
    snap = {'key': (0, 0.0), 'p': None, 'path': snap_path} if snap_path else None
    gen = torch.Generator(device=DEV); gen.manual_seed(seed)
    p = init_params(M, gen)
    if warm is not None:
        with torch.no_grad():
            if polish:   # basin hopping: replicate the best warm cells, perturb, re-descend
                src = min(warm['U'].shape[0], src or max(1, M // 8))
                idx = torch.arange(M, device=DEV) % src
                for k in PNAMES:
                    w = warm[k][:src].to(DEV)
                    rep = w[idx]
                    sh = rep.shape
                    if logjit > 0:   # broad multiplicative (log-normal) perturbation
                        e = torch.randn(sh, generator=gen, device=DEV) * logjit
                        p[k][:] = rep * torch.exp(e)
                    else:
                        p[k][:] = rep * (1 + torch.randn(sh, generator=gen, device=DEV) * jitter)
                    p[k][:src] = w                      # keep exact copies too
            else:
                for k in PNAMES:
                    n = min(M, warm[k].shape[0])
                    p[k][:n] = warm[k][:n].to(DEV)
    opt = torch.optim.Adam([p[k] for k in PNAMES], lr=lr, betas=(0.9, 0.98))
    if hop_every:
        def _lam(i):
            f = i / steps
            if f < 0.03:
                return f / 0.03
            if f < 0.85:
                return 1.0
            return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (f - 0.85) / 0.15))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lam)
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                    pct_start=0.15, final_div_factor=50.0)
    t0 = time.time()
    for st in range(steps):
        frac = st / steps
        pmax = 1.0 if polish else min(1.0, 0.12 + frac / max(ramp, 1e-9))  # curriculum
        if polish:      # half the eval distribution, half full stress
            a1, b1, t1 = eval_batch(bs // 2, gen)
            a2, b2, t2 = hard_batch(bs - bs // 2, gen)
            atok = torch.cat([a1, a2], 0); btok = torch.cat([b1, b2], 0)
            tgt = torch.cat([t1, t2], 0)
        else:
            atok, btok, tgt = sample(bs, pmax, gen)
        pos = None
        if sjit > 1.0:  # random recency slope: a blurred read cannot calibrate to it
            f = torch.exp((torch.rand((), generator=gen, device=DEV) * 2 - 1) * math.log(sjit))
            pos = POS * f
        ew = ent * (1.0 - frac) if ent > 0 else 0.0      # annealed sharpening pressure
        ls, _ = loss_and_acc(p, atok, btok, tgt, ent_w=ew, pos=pos)
        opt.zero_grad(set_to_none=True)
        ls.sum().backward()
        torch.nn.utils.clip_grad_norm_([p[k] for k in PNAMES], 1.0)
        opt.step(); sched.step()
        if hop_every and st % hop_every == hop_every - 1 and frac < 0.85:
            with torch.no_grad():
                sc = torch.zeros(M, device=DEV)
                for _ in range(3):
                    a2, b2, t2 = hard_batch(768, gen)
                    _, ok = loss_and_acc(p, a2, b2, t2)
                    a2, b2, t2 = eval_batch(768, gen)
                    _, ok2 = loss_and_acc(p, a2, b2, t2)
                    sc += torch.minimum(ok, ok2)
                order = sc.argsort(descending=True)
                nkeep = max(1, int(M * (1.0 - hop_frac)))
                keep, drop = order[:nkeep], order[nkeep:]
                parent = keep[torch.randint(0, nkeep, (drop.numel(),), generator=gen, device=DEV)]
                for k in PNAMES:
                    src_v = p[k][parent]
                    e = torch.randn(src_v.shape, generator=gen, device=DEV) * hop_sig
                    hit = torch.rand(src_v.shape, generator=gen, device=DEV) < 0.30
                    p[k][drop] = src_v * torch.exp(e * hit)
                    stt = opt.state.get(p[k])
                    if stt:
                        stt['exp_avg'][drop] = 0
                        stt['exp_avg_sq'][drop] = 0
                print(f'   hop @ {st+1}: kept {nkeep}, best score {sc.max().item()/3:.4f}', flush=True)
        if st % 200 == 199 and not polish:
            resurrect(p, gen, atok, btok)
        if st % 1000 == 999 and frac < 0.6 and not polish:   # restart hopeless cells
            with torch.no_grad():
                a2, b2, t2 = eval_batch(512, gen, stress=True)
                _, ok = loss_and_acc(p, a2, b2, t2)
                bad = (ok < 0.02).nonzero().flatten()
                if bad.numel() < M:
                    reinit_cells(p, opt, bad, gen)
        if st % log_every == log_every - 1 or st == steps - 1:
            with torch.no_grad():
                a2, b2, t2 = eval_batch(1024, gen)
                _, ok = loss_and_acc(p, a2, b2, t2)
                a3, b3, t3 = eval_batch(1024, gen, stress=True)
                _, ok3 = loss_and_acc(p, a3, b3, t3)
                comb = torch.minimum(ok, ok3)
                bst = comb.argmax().item()
                ngood = (comb > 0.99).sum().item()
                # snapshot the best population ever seen: hopping and the hardening
                # curriculum can both wipe out good cells, and they are what we want.
                key = (ngood, float(comb.max()))
                if snap is not None and key > snap['key']:
                    snap['key'] = key
                    order = comb.argsort(descending=True)
                    snap['p'] = {k: p[k][order].detach().clone() for k in PNAMES}
                    torch.save(snap['p'], snap['path'])
                print(f'step {st+1:6d} pmax {pmax:.2f} loss {ls.min().item():.5f} '
                      f'best unif {ok[bst]:.4f} stress {ok3[bst]:.4f} '
                      f'| #cells>0.99 {ngood} | best-ever {snap["key"][0] if snap else 0} '
                      f'| {time.time()-t0:.0f}s', flush=True)
    return {k: p[k].detach() for k in PNAMES}


@torch.no_grad()
def rank(p, gen, n=40, bs=2048):
    """Score every cell on uniform + stress; returns (score, index) sorted desc."""
    tot = torch.zeros(p['U'].shape[0], device=DEV)
    for i in range(n):
        a, b, t = eval_batch(bs, gen, stress=(i % 2 == 1))
        _, ok = loss_and_acc(p, a, b, t)
        tot += ok
    return tot / n


# ------------------------------------------------------------- exporting ---
TEMPLATE = '''"""Minimal transformer that adds two integers in [0, 99_999_999_999_999].

A single-layer, single-head causal transformer with {NP} learned parameters.
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
        self.b = nn.Parameter(torch.zeros(()))      # feature-layer bias
        self.w2 = nn.Parameter(torch.zeros(()))     # key/query feature mixing
        self.bq = nn.Parameter(torch.zeros(()))     # query bias
        self.wo = nn.Parameter(torch.zeros(2))      # attention output projection
        # fixed, never trained
        self.register_buffer('w1', torch.tensor({W1!r}))
        self.register_buffer('slope', torch.tensor({SLOPE!r}))
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
        c1 = self.w1 * z + self.w2 * F.relu(z + self.b)         # key / query feature
        c2 = z                                                  # value = residual stream
        q = c1 + self.bq
        s = q.unsqueeze(-1) * c1.unsqueeze(-2) + self.slope * self.pos
        c_in = torch.matmul(torch.softmax(s + self.mask_in, -1), c2.unsqueeze(-1)).squeeze(-1)
        c_out = torch.matmul(torch.softmax(s + self.mask_out, -1), c2.unsqueeze(-1)).squeeze(-1)
        y = z + self.wo[0] * c_in + self.wo[1] * c_out          # residual
        return 2.0 * y.unsqueeze(-1) * self.U - self.U * self.U  # tied readout


_WEIGHTS = {WEIGHTS}


def build_model():
    model = TinyAdder()
    with torch.no_grad():
        for name, value in _WEIGHTS.items():
            getattr(model, name).copy_(torch.tensor(value))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    meta = {{
        'name': 'TinyAdder',
        'n_parameters': n_params,
        'architecture': 'decoder-only transformer, 1 layer, 1 head',
        'tokenization': 'binary digits, least significant first, 49 slots',
        'description': ('self-attention performs carry lookahead: each slot attends to '
                        'the nearest earlier non-propagating slot and reads its carry'),
        'max_operand': 99999999999999,
    }}
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
'''


def export(p, idx, path='/workspace/submission.py'):
    w = {k: (p[k][idx].tolist() if p[k].dim() > 1 else float(p[k][idx])) for k in PNAMES}
    body = '{\n' + ''.join(f"    {k!r}: {v!r},\n" for k, v in w.items()) + '}'
    src = TEMPLATE.format(NP=7, W1=W1, SLOPE=SLOPE, WEIGHTS=body)
    with open(path, 'w') as f:
        f.write(src)
    print('wrote', path, w)


# ----------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--M', type=int, default=192)
    ap.add_argument('--bs', type=int, default=192)
    ap.add_argument('--lr', type=float, default=3e-2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--warm', type=str, default=None)
    ap.add_argument('--out', type=str, default='/workspace/ckpt.pt')
    ap.add_argument('--export', action='store_true')
    ap.add_argument('--polish', action='store_true')
    ap.add_argument('--jitter', type=float, default=0.02)
    ap.add_argument('--ramp', type=float, default=0.55)
    ap.add_argument('--ent', type=float, default=0.0)
    ap.add_argument('--logjit', type=float, default=0.0)
    ap.add_argument('--src', type=int, default=0)
    ap.add_argument('--hop_every', type=int, default=0)
    ap.add_argument('--hop_frac', type=float, default=0.75)
    ap.add_argument('--hop_sig', type=float, default=1.5)
    ap.add_argument('--sjit', type=float, default=1.0)
    a = ap.parse_args()
    warm = torch.load(a.warm) if a.warm else None
    p = train(M=a.M, steps=a.steps, bs=a.bs, lr=a.lr, seed=a.seed, warm=warm,
              polish=a.polish, jitter=a.jitter, ramp=a.ramp, ent=a.ent, logjit=a.logjit, src=a.src,
              hop_every=a.hop_every, hop_frac=a.hop_frac, hop_sig=a.hop_sig, sjit=a.sjit,
              snap_path=a.out + '.best')
    gen = torch.Generator(device=DEV); gen.manual_seed(12345)
    sc = rank(p, gen)
    order = sc.argsort(descending=True)
    print('top scores:', [round(float(sc[i]), 5) for i in order[:8]])
    p = {k: p[k][order].contiguous() for k in PNAMES}          # best cell first
    torch.save({k: v.cpu() for k, v in p.items()}, a.out)
    if a.export:
        export(p, 0)


if __name__ == '__main__':
    main()
