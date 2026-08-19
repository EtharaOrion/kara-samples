"""Arch B: drop the 55-param unordered-pair lookup table.

An additive digit code h(a)+h(b) provably cannot express "a+b==9" with a linear
key, so attempt 3 paid 55 parameters for a pair lookup.  Here a small MLP runs
before attention and turns the additive code into two features:
    c1 ~ propagate indicator  (used as the attention key)
    c2 ~ carry-generate value (used as the attention value)
which is possible because the MLP is nonlinear.  The code h is free to arrange
itself so that all five propagate pairs collapse to one code value.

Params: U(10) + MLP_A(4*dffa) + attn(4) + MLP_B(4*dffb) + head(20 or 1).
"""
import argparse, math, time
import torch
import torch.nn.functional as F

from lab import NPOS, L, gen_batch, sum_digits, make_bias

torch.backends.cuda.matmul.allow_tf32 = True


def encode_b(a_d, b_d):
    B = a_d.shape[0]
    z = torch.zeros(B, 1, dtype=a_d.dtype, device=a_d.device)
    a = torch.cat([z, a_d], 1)
    b = torch.cat([z, b_d], 1)
    return F.one_hot(a, 10).float() + F.one_hot(b, 10).float()


def init_params(M, dev, cfg, g):
    def r(*shape, s=1.0):
        return (torch.randn(M, *shape, device=dev, generator=g) * s).requires_grad_()
    da, db = cfg['d_ffa'], cfg['d_ffb']
    P = {
        'U': r(10, s=0.8),
        'W1a': r(da, s=1.0), 'b1a': r(da, s=1.0),
        'W2a0': r(da, s=1.0), 'W2a1': r(da, s=1.0),
        'wq': r(s=1.0), 'bq': r(s=1.0), 'wv': r(s=1.0),
        'W1b0': r(db, s=1.0), 'W1b1': r(db, s=1.0), 'b1b': r(db, s=1.0),
        'W2b': r(db, s=1.0),
        'wout': r(10, s=0.8), 'bout': r(10, s=0.3),
    }
    P['slope'] = cfg['slope_init'].clone().to(dev).requires_grad_()
    return P


def n_params(cfg):
    return 10 + 4 * cfg['d_ffa'] + 4 + 4 * cfg['d_ffb'] + 20


def forward(P, s1, rel, block, sigma=None, gen=None):
    z = torch.einsum('blk,mk->mbl', s1, P['U'])                       # M,B,L
    hA = torch.relu(z[..., None] * P['W1a'][:, None, None, :] + P['b1a'][:, None, None, :])
    c1 = (hA * P['W2a0'][:, None, None, :]).sum(-1)
    c2 = (hA * P['W2a1'][:, None, None, :]).sum(-1)

    q = c1 * P['wq'][:, None, None] + P['bq'][:, None, None]
    sc = q[..., :, None] * c1[..., None, :] + P['slope'][:, None, None, None] * rel
    if sigma is not None:
        sc = sc + sigma[:, None, None, None] * torch.randn(sc.shape, device=sc.device, generator=gen)
    A = torch.softmax(sc.masked_fill(block, float('-inf')), -1)
    carry = P['wv'][:, None, None] * torch.einsum('mbij,mbj->mbi', A, c2)

    hB = torch.relu(z[..., None] * P['W1b0'][:, None, None, :]
                    + carry[..., None] * P['W1b1'][:, None, None, :]
                    + P['b1b'][:, None, None, :])
    ans = (hB * P['W2b'][:, None, None, :]).sum(-1)
    return ans[..., None] * P['wout'][:, None, None, :] + P['bout'][:, None, None, :]


def loss_fn(logits, tgt):
    lg = logits[:, :, 1:, :]
    lse = torch.logsumexp(lg, -1)
    t = tgt[None, :, :, None].expand(lg.shape[0], -1, -1, 1)
    return (lse - lg.gather(-1, t).squeeze(-1)).mean(dim=(1, 2))


@torch.no_grad()
def evaluate(P, rel, block, dev, g, n=200_000, bs=20000, stress=False):
    M = P['U'].shape[0]
    ok = torch.zeros(M, device=dev); tot = 0
    for _ in range(n // bs):
        if stress:
            a_d, b_d = gen_batch(bs, dev, g)
        else:
            a_d = torch.randint(0, 10, (bs, NPOS), device=dev, generator=g)
            b_d = torch.randint(0, 10, (bs, NPOS), device=dev, generator=g)
            a_d[:, 14] = 0; b_d[:, 14] = 0
        tgt = sum_digits(a_d, b_d)
        pred = forward(P, encode_b(a_d, b_d), rel, block).argmax(-1)[:, :, 1:]
        ok += (pred == tgt[None]).all(-1).float().sum(1); tot += bs
    return ok / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=60000)
    ap.add_argument('--bs', type=int, default=2048)
    ap.add_argument('--lr', type=float, default=3.5e-3)
    ap.add_argument('--d_ffa', type=int, default=5)
    ap.add_argument('--d_ffb', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='/workspace/ckpt_b.pt')
    args = ap.parse_args()

    dev = 'cuda'
    g = torch.Generator(device=dev); g.manual_seed(args.seed)
    slopes = torch.tensor([3.0, 5.0, 7.0, 9.0]).repeat_interleave(4)
    sigmas = torch.tensor([0.0, 0.05, 0.1, 0.2]).repeat(4)
    M = slopes.numel()
    cfg = {'d_ffa': args.d_ffa, 'd_ffb': args.d_ffb, 'slope_init': slopes}
    P = init_params(M, dev, cfg, g)
    sig0 = sigmas.to(dev)
    rel, block = make_bias(dev)
    print('param count', n_params(cfg), flush=True)

    opt = torch.optim.AdamW(list(P.values()), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)

    t0 = time.time()
    for step in range(args.steps):
        a_d, b_d = gen_batch(args.bs, dev, g)
        tgt = sum_digits(a_d, b_d)
        sigma = sig0 * max(0.0, 1.0 - step / (0.5 * args.steps))
        losses = loss_fn(forward(P, encode_b(a_d, b_d), rel, block, sigma, g), tgt)
        opt.zero_grad(set_to_none=True)
        losses.sum().backward()
        torch.nn.utils.clip_grad_norm_(list(P.values()), 1.0)
        opt.step(); sched.step()
        if step % 2000 == 0 or step == args.steps - 1:
            print(f'step {step} {time.time()-t0:.0f}s loss ' +
                  ' '.join(f'{l:.4f}' for l in losses.tolist()), flush=True)
        if step and step % 20000 == 0:
            print('  eval ' + ' '.join(f'{a:.4f}' for a in evaluate(P, rel, block, dev, g).tolist()), flush=True)

    acc = evaluate(P, rel, block, dev, g, n=1_000_000)
    accs = evaluate(P, rel, block, dev, g, n=1_000_000, stress=True)
    print('final uniform ' + ' '.join(f'{a:.5f}' for a in acc.tolist()), flush=True)
    print('final stress  ' + ' '.join(f'{a:.5f}' for a in accs.tolist()), flush=True)
    best = int(torch.minimum(acc, accs).argmax())
    print('best', best, 'slope', slopes[best].item(), 'sigma', sigmas[best].item(),
          'acc', acc[best].item(), accs[best].item(), flush=True)
    torch.save({'params': {k: v.detach().cpu() for k, v in P.items()}, 'acc': acc.cpu(),
                'acc_stress': accs.cpu(), 'best': best, 'slopes': slopes, 'sigmas': sigmas,
                'cfg': {'d_ffa': args.d_ffa, 'd_ffb': args.d_ffb}}, args.out)


if __name__ == '__main__':
    main()
