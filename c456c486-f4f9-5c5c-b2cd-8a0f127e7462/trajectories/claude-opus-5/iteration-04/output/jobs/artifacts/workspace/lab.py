"""Ensemble trainer for the tiny addition transformer.

Trains M models in parallel (leading `model` dim on every parameter) so a whole
hyper-parameter grid costs about one model's wall clock.  Nothing here is
imported by submission.py -- this file only produces weights.
"""
import argparse, json, math, os, time
import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

NPOS = 15          # digit slots (14-digit operands -> 15-digit sum)
L = NPOS + 1       # + sentinel slot 0
NPAIR = 55         # unordered digit pairs


def pair_tok(a, b):
    lo = torch.minimum(a, b)
    hi = torch.maximum(a, b)
    return (lo * (21 - lo)) // 2 + (hi - lo)


# ---------------------------------------------------------------- data ----
def _pad(d, n):
    """d: (B, NPOS) digits, zero out positions >= n (per row length tensor)."""
    idx = torch.arange(NPOS, device=d.device)[None, :]
    return torch.where(idx < n[:, None], d, torch.zeros_like(d))


def gen_batch(B, dev, g):
    n = B // 4
    parts_a, parts_b = [], []

    # 1. uniform full-width 14-digit operands
    a = torch.randint(0, 10, (n, NPOS), device=dev, generator=g)
    b = torch.randint(0, 10, (n, NPOS), device=dev, generator=g)
    a[:, 14] = 0; b[:, 14] = 0
    parts_a.append(a); parts_b.append(b)

    # 2. independent random operand lengths
    a = torch.randint(0, 10, (n, NPOS), device=dev, generator=g)
    b = torch.randint(0, 10, (n, NPOS), device=dev, generator=g)
    la = torch.randint(1, 15, (n,), device=dev, generator=g)
    lb = torch.randint(1, 15, (n,), device=dev, generator=g)
    parts_a.append(_pad(a, la)); parts_b.append(_pad(b, lb))

    # 3. digits skewed toward 0 / 9 (carry-chain rich)
    a = torch.randint(0, 10, (n, NPOS), device=dev, generator=g)
    b = torch.randint(0, 10, (n, NPOS), device=dev, generator=g)
    ext = torch.randint(0, 2, (n, NPOS), device=dev, generator=g) * 9
    m1 = torch.rand((n, NPOS), device=dev, generator=g) < 0.5
    m2 = torch.rand((n, NPOS), device=dev, generator=g) < 0.5
    a = torch.where(m1, ext, a); b = torch.where(m2, 9 - ext, b)
    a[:, 14] = 0; b[:, 14] = 0
    parts_a.append(a); parts_b.append(b)

    # 4. forced propagate chains: b_i = 9 - a_i with per-sample rate
    a = torch.randint(0, 10, (n, NPOS), device=dev, generator=g)
    b = torch.randint(0, 10, (n, NPOS), device=dev, generator=g)
    rate = 0.3 + 0.7 * torch.rand((n, 1), device=dev, generator=g)
    m = torch.rand((n, NPOS), device=dev, generator=g) < rate
    b = torch.where(m, 9 - a, b)
    a[:, 14] = 0; b[:, 14] = 0
    parts_a.append(a); parts_b.append(b)

    return torch.cat(parts_a), torch.cat(parts_b)


def sum_digits(a_d, b_d):
    s = a_d + b_d
    out = torch.empty_like(s)
    carry = torch.zeros_like(s[:, 0])
    for i in range(NPOS):
        t = s[:, i] + carry
        out[:, i] = t % 10
        carry = (t >= 10).long()
    return out


def encode(a_d, b_d):
    """digits -> model inputs (one-hots incl. sentinel slot)."""
    B = a_d.shape[0]
    z = torch.zeros(B, 1, dtype=a_d.dtype, device=a_d.device)
    a = torch.cat([z, a_d], 1)
    b = torch.cat([z, b_d], 1)
    s1 = F.one_hot(a, 10).float() + F.one_hot(b, 10).float()
    p1 = F.one_hot(pair_tok(a, b), NPAIR).float()
    return s1, p1


# --------------------------------------------------------------- model ----
def init_params(M, dev, cfg, g):
    def r(*shape, s=1.0):
        return (torch.randn(M, *shape, device=dev, generator=g) * s).requires_grad_()
    P = {
        'U': r(10, s=0.8),
        'P': r(NPAIR, s=0.8),
        'wq': r(2, s=0.8), 'bq': r(s=0.5), 'wk': r(2, s=0.8),
        'wv': r(2, s=0.8), 'wo': r(2, s=0.8),
        'W1': r(cfg['d_ff'], 2, s=0.8), 'b1': r(cfg['d_ff'], s=0.3),
        'W2': r(2, cfg['d_ff'], s=0.8),
        'Wout': r(10, 2, s=0.8),
    }
    P['slope'] = cfg['slope_init'].clone().to(dev).requires_grad_()
    return P


def make_bias(dev):
    i = torch.arange(L, device=dev)
    rel = (i[None, :] - i[:, None]).float()          # j - i
    allow = i[None, :] < i[:, None]                  # strictly causal
    allow[0, 0] = True                               # sentinel self-attends
    return rel, ~allow


def forward(P, s1, p1, rel, block, sigma=None, gen=None):
    ch0 = torch.einsum('blk,mk->mbl', s1, P['U'])
    ch1 = torch.einsum('blk,mk->mbl', p1, P['P'])
    x = torch.stack([ch0, ch1], -1)                                  # M,B,L,2

    q = (x * P['wq'][:, None, None, :]).sum(-1) + P['bq'][:, None, None]
    k = (x * P['wk'][:, None, None, :]).sum(-1)
    v = (x * P['wv'][:, None, None, :]).sum(-1)
    sc = q[..., :, None] * k[..., None, :] + P['slope'][:, None, None, None] * rel
    if sigma is not None:
        noise = torch.randn(sc.shape, device=sc.device, generator=gen)
        sc = sc + sigma[:, None, None, None] * noise
    sc = sc.masked_fill(block, float('-inf'))
    A = torch.softmax(sc, -1)
    o = torch.einsum('mbij,mbj->mbi', A, v)
    x = x + o[..., None] * P['wo'][:, None, None, :]

    h = torch.relu(torch.einsum('mbld,mfd->mblf', x, P['W1']) + P['b1'][:, None, None, :])
    x = x + torch.einsum('mblf,mdf->mbld', h, P['W2'])
    return torch.einsum('mbld,mcd->mblc', x, P['Wout'])


def loss_fn(logits, tgt):
    lg = logits[:, :, 1:, :]                                          # M,B,15,10
    lse = torch.logsumexp(lg, -1)
    t = tgt[None, :, :, None].expand(lg.shape[0], -1, -1, 1)
    pick = lg.gather(-1, t).squeeze(-1)
    return (lse - pick).mean(dim=(1, 2))                              # per model


@torch.no_grad()
def evaluate(P, rel, block, dev, g, n=200_000, bs=20000, stress=False):
    M = P['U'].shape[0]
    ok = torch.zeros(M, device=dev)
    tot = 0
    for _ in range(n // bs):
        if stress:
            a_d, b_d = gen_batch(bs, dev, g)
        else:
            a_d = torch.randint(0, 10, (bs, NPOS), device=dev, generator=g)
            b_d = torch.randint(0, 10, (bs, NPOS), device=dev, generator=g)
            a_d[:, 14] = 0; b_d[:, 14] = 0
        tgt = sum_digits(a_d, b_d)
        s1, p1 = encode(a_d, b_d)
        pred = forward(P, s1, p1, rel, block).argmax(-1)[:, :, 1:]
        ok += (pred == tgt[None]).all(-1).float().sum(1)
        tot += bs
    return ok / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=60000)
    ap.add_argument('--bs', type=int, default=2048)
    ap.add_argument('--lr', type=float, default=3.5e-3)
    ap.add_argument('--d_ff', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='/workspace/ckpt_a.pt')
    args = ap.parse_args()

    dev = 'cuda'
    g = torch.Generator(device=dev); g.manual_seed(args.seed)
    slopes = torch.tensor([5.0, 6.0, 7.0, 8.0]).repeat_interleave(4)
    sigmas = torch.tensor([0.0, 0.05, 0.1, 0.2]).repeat(4)
    M = slopes.numel()
    cfg = {'d_ff': args.d_ff, 'slope_init': slopes}
    P = init_params(M, dev, cfg, g)
    sig0 = sigmas.to(dev)
    rel, block = make_bias(dev)

    opt = torch.optim.AdamW(list(P.values()), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)

    t0 = time.time()
    for step in range(args.steps):
        a_d, b_d = gen_batch(args.bs, dev, g)
        tgt = sum_digits(a_d, b_d)
        s1, p1 = encode(a_d, b_d)
        anneal = max(0.0, 1.0 - step / (0.5 * args.steps))
        sigma = sig0 * anneal
        logits = forward(P, s1, p1, rel, block, sigma=sigma, gen=g)
        losses = loss_fn(logits, tgt)
        opt.zero_grad(set_to_none=True)
        losses.sum().backward()
        torch.nn.utils.clip_grad_norm_(list(P.values()), 1.0)
        opt.step(); sched.step()
        if step % 2000 == 0 or step == args.steps - 1:
            print(f'step {step} {time.time()-t0:.0f}s loss ' +
                  ' '.join(f'{l:.4f}' for l in losses.tolist()), flush=True)
        if step and step % 20000 == 0:
            acc = evaluate(P, rel, block, dev, g)
            print('  eval ' + ' '.join(f'{a:.4f}' for a in acc.tolist()), flush=True)

    acc = evaluate(P, rel, block, dev, g, n=1_000_000)
    accs = evaluate(P, rel, block, dev, g, n=1_000_000, stress=True)
    print('final uniform ' + ' '.join(f'{a:.5f}' for a in acc.tolist()), flush=True)
    print('final stress  ' + ' '.join(f'{a:.5f}' for a in accs.tolist()), flush=True)
    best = int(torch.minimum(acc, accs).argmax())
    print('best model', best, 'slope_init', slopes[best].item(), 'sigma', sigmas[best].item(), flush=True)
    torch.save({'params': {k: v.detach().cpu() for k, v in P.items()},
                'acc': acc.cpu(), 'acc_stress': accs.cpu(), 'best': best,
                'slopes': slopes, 'sigmas': sigmas, 'cfg': {'d_ff': args.d_ff}}, args.out)


if __name__ == '__main__':
    main()
