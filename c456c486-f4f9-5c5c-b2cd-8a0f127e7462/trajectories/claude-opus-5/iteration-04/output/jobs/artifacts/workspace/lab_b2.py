"""Arch B': arch A, but the 55-parameter unordered-pair lookup table is replaced
by a small ReLU MLP that runs before attention and manufactures the two features
attention needs (propagate key, generate value) out of the 10-parameter additive
digit code.

residual x = [z, c] with z = U[a]+U[b] (channel 0) and c = attention output.
  MLP_A : h = relu(z * W1a + b1a);  c1 = h.W2a0 (key), c2 = h.W2a1 (value)
  attn  : score(i,j) = (wq*c1_i + bq) * c1_j + slope*(j-i), strictly causal
  MLP_B : residual ReLU MLP over [z, carry], writes both channels
  head  : bias-free 2 x 10

params = 10 + 4*dffa + 4 + 5*dffb + 20
"""
import argparse, time
import torch
import torch.nn.functional as F

from lab import NPOS, L, gen_batch, sum_digits, make_bias
from lab_b import encode_b

torch.backends.cuda.matmul.allow_tf32 = True


def n_params(cfg):
    return 10 + 4 * cfg['d_ffa'] + 4 + 5 * cfg['d_ffb'] + 20


def init_params(M, dev, cfg, g):
    def r(*shape, s=1.0):
        return (torch.randn(M, *shape, device=dev, generator=g) * s).requires_grad_()
    da, db = cfg['d_ffa'], cfg['d_ffb']
    P = {
        'U': r(10, s=0.8),
        'W1a': r(da, s=1.0), 'b1a': r(da, s=1.0), 'W2a0': r(da, s=1.0), 'W2a1': r(da, s=1.0),
        'wq': r(s=1.0), 'bq': r(s=1.0), 'wv': r(s=1.0),
        'W1b0': r(db, s=1.0), 'W1b1': r(db, s=1.0), 'b1b': r(db, s=0.5), 'W2b': r(2, db, s=0.8),
        'Wout': r(10, 2, s=0.8),
    }
    P['slope'] = cfg['slope_init'].clone().to(dev).requires_grad_()
    return P


def forward(P, s1, rel, block, sigma=None, gen=None):
    z = torch.einsum('blk,mk->mbl', s1, P['U'])
    ha = torch.relu(z[..., None] * P['W1a'][:, None, None, :] + P['b1a'][:, None, None, :])
    c1 = (ha * P['W2a0'][:, None, None, :]).sum(-1)
    c2 = (ha * P['W2a1'][:, None, None, :]).sum(-1)

    q = c1 * P['wq'][:, None, None] + P['bq'][:, None, None]
    sc = q[..., :, None] * c1[..., None, :] + P['slope'][:, None, None, None] * rel
    if sigma is not None:
        sc = sc + sigma[:, None, None, None] * torch.randn(sc.shape, device=sc.device, generator=gen)
    A = torch.softmax(sc.masked_fill(block, float('-inf')), -1)
    carry = P['wv'][:, None, None] * torch.einsum('mbij,mbj->mbi', A, c2)

    x = torch.stack([z, carry], -1)                                   # M,B,L,2
    hb = torch.relu(z[..., None] * P['W1b0'][:, None, None, :]
                    + carry[..., None] * P['W1b1'][:, None, None, :]
                    + P['b1b'][:, None, None, :])
    x = x + torch.einsum('mblf,mdf->mbld', hb, P['W2b'])
    return torch.einsum('mbld,mcd->mblc', x, P['Wout'])


def loss_fn(logits, tgt):
    lg = logits[:, :, 1:, :]
    lse = torch.logsumexp(lg, -1)
    t = tgt[None, :, :, None].expand(lg.shape[0], -1, -1, 1)
    return (lse - lg.gather(-1, t).squeeze(-1)).mean(dim=(1, 2))


@torch.no_grad()
def evaluate(P, rel, block, dev, g, n=200_000, bs=20000, stress=False, fwd=forward):
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
        pred = fwd(P, encode_b(a_d, b_d), rel, block).argmax(-1)[:, :, 1:]
        ok += (pred == tgt[None]).all(-1).float().sum(1); tot += bs
    return ok / tot


def run(args, cfg, slopes, sigmas, fwd=forward, initf=init_params, npf=n_params):
    dev = 'cuda'
    g = torch.Generator(device=dev); g.manual_seed(args.seed)
    M = slopes.numel()
    cfg['slope_init'] = slopes
    P = initf(M, dev, cfg, g)
    sig0 = sigmas.to(dev)
    rel, block = make_bias(dev)
    print('param count', npf(cfg), flush=True)

    opt = torch.optim.AdamW(list(P.values()), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    t0 = time.time()
    for step in range(args.steps):
        a_d, b_d = gen_batch(args.bs, dev, g)
        tgt = sum_digits(a_d, b_d)
        sigma = sig0 * max(0.0, 1.0 - step / (0.5 * args.steps))
        losses = loss_fn(fwd(P, encode_b(a_d, b_d), rel, block, sigma, g), tgt)
        opt.zero_grad(set_to_none=True)
        losses.sum().backward()
        torch.nn.utils.clip_grad_norm_(list(P.values()), 1.0)
        opt.step(); sched.step()
        if step % 2000 == 0 or step == args.steps - 1:
            print(f'step {step} {time.time()-t0:.0f}s loss ' +
                  ' '.join(f'{l:.4f}' for l in losses.tolist()), flush=True)
        if step and step % 20000 == 0:
            print('  eval ' + ' '.join(f'{a:.4f}' for a in
                  evaluate(P, rel, block, dev, g, fwd=fwd).tolist()), flush=True)

    acc = evaluate(P, rel, block, dev, g, n=1_000_000, fwd=fwd)
    accs = evaluate(P, rel, block, dev, g, n=1_000_000, stress=True, fwd=fwd)
    print('final uniform ' + ' '.join(f'{a:.5f}' for a in acc.tolist()), flush=True)
    print('final stress  ' + ' '.join(f'{a:.5f}' for a in accs.tolist()), flush=True)
    best = int(torch.minimum(acc, accs).argmax())
    print('best', best, 'slope', slopes[best].item(), 'sigma', sigmas[best].item(),
          'acc', acc[best].item(), accs[best].item(), flush=True)
    save_cfg = {k: v for k, v in cfg.items() if k != 'slope_init'}
    torch.save({'params': {k: v.detach().cpu() for k, v in P.items()}, 'acc': acc.cpu(),
                'acc_stress': accs.cpu(), 'best': best, 'slopes': slopes, 'sigmas': sigmas,
                'cfg': save_cfg}, args.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=80000)
    ap.add_argument('--bs', type=int, default=2048)
    ap.add_argument('--lr', type=float, default=3.5e-3)
    ap.add_argument('--d_ffa', type=int, default=5)
    ap.add_argument('--d_ffb', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='/workspace/ckpt_b2.pt')
    ap.add_argument('--slopes', default='3,5,7,9')
    ap.add_argument('--sigmas', default='0,0.05,0.1,0.2')
    args = ap.parse_args()
    slopes = torch.tensor([float(s) for s in args.slopes.split(',')]).repeat_interleave(4)
    sigmas = torch.tensor([float(s) for s in args.sigmas.split(',')]).repeat(4)
    run(args, {'d_ffa': args.d_ffa, 'd_ffb': args.d_ffb}, slopes, sigmas)


if __name__ == '__main__':
    main()
