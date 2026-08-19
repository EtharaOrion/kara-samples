"""Population runner: same ensemble training, but grid cells that are clearly
stuck get re-initialised (weights + Adam state) instead of burning the full
schedule.  Roughly 1 in 8 random inits finds the carry-lookahead basin, so this
multiplies the number of converged models per wall-clock hour.
"""
import argparse, time
import torch

import lab_b2, lab_c, lab_d
from lab import gen_batch, sum_digits, make_bias, NPOS
from lab_b2 import encode_b, loss_fn, evaluate

torch.backends.cuda.matmul.allow_tf32 = True

ARCH = {'b': (lab_b2.forward, lab_b2.init_params, lab_b2.n_params),
        'c': (lab_c.forward, lab_c.init_params, lab_c.n_params),
        'd': (lab_d.forward, lab_d.init_params, lab_d.n_params)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='c')
    ap.add_argument('--steps', type=int, default=80000)
    ap.add_argument('--bs', type=int, default=2048)
    ap.add_argument('--lr', type=float, default=3.5e-3)
    ap.add_argument('--d_ffa', type=int, default=5)
    ap.add_argument('--d_ffb', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='/workspace/ckpt_r.pt')
    ap.add_argument('--slopes', default='2,3,4,5')
    ap.add_argument('--freqs', default='0.3,0.7,1.5,3.0')
    ap.add_argument('--sigmas', default='0.05')
    ap.add_argument('--restart_every', type=int, default=4000)
    ap.add_argument('--M', type=int, default=16)
    ap.add_argument('--min_age', type=int, default=12000)
    ap.add_argument('--restart_until', type=float, default=0.7)
    ap.add_argument('--restart_thresh', type=float, default=0.40)
    args = ap.parse_args()

    dev = 'cuda'
    fwd, initf, npf = ARCH[args.arch]
    g = torch.Generator(device=dev); g.manual_seed(args.seed)
    M = args.M
    sl = [float(s) for s in args.slopes.split(',')]
    fr = [float(s) for s in args.freqs.split(',')]
    sig = [float(s) for s in args.sigmas.split(',')]
    slopes = torch.tensor(sl).repeat_interleave(M // len(sl))[:M]
    freqs = torch.tensor((fr * M)[:M])
    sigmas = torch.tensor((sig * M)[:M])
    cfg = {'d_ffa': args.d_ffa, 'd_ffb': args.d_ffb, 'freq': freqs, 'slope_init': slopes}
    P = initf(M, dev, cfg, g)
    sig0 = sigmas.to(dev)
    rel, block = make_bias(dev)
    print('arch', args.arch, 'param count', npf(cfg), flush=True)

    opt = torch.optim.AdamW(list(P.values()), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    ema = torch.full((M,), 2.3, device=dev)
    age = torch.zeros(M, device=dev)
    best = {'loss': torch.full((M,), 9.9), 'params': {k: v.detach().cpu().clone() for k, v in P.items()}}
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
        ema.mul_(0.995).add_(losses.detach(), alpha=0.005)
        age += 1

        if step % 2000 == 0 or step == args.steps - 1:
            print(f'step {step} {time.time()-t0:.0f}s ema ' +
                  ' '.join(f'{l:.4f}' for l in ema.tolist()), flush=True)

        # keep the best weights each cell has ever reached (restarts destroy them)
        if step % 2000 == 1999:
            e = ema.cpu()
            better = (e < best['loss']).nonzero().flatten()
            for i in better.tolist():
                for k, v in P.items():
                    best['params'][k][i] = v[i].detach().cpu()
                best['loss'][i] = e[i]

        if (step and step % args.restart_every == 0
                and step < args.restart_until * args.steps):
            # give every cell a fair trial, then recycle it: finding the
            # carry-lookahead basin is an init lottery, so trials/hour is what
            # matters.  A cell that has not converged after min_age steps is
            # re-drawn.
            dead = ((ema > args.restart_thresh) & (age >= args.min_age)).nonzero().flatten()
            if len(dead):
                sub = dict(cfg)
                sub['freq'] = cfg['freq'][dead.cpu()]
                sub['slope_init'] = cfg['slope_init'][dead.cpu()]
                fresh = initf(len(dead), dev, sub, g)
                with torch.no_grad():
                    for k in P:
                        P[k][dead] = fresh[k].detach()
                        st = opt.state.get(P[k])
                        if st:
                            st['exp_avg'][dead] = 0
                            st['exp_avg_sq'][dead] = 0
                ema[dead] = 2.3
                age[dead] = 0
                print(f'  restart {len(dead)} cells: {dead.tolist()}', flush=True)

    # restore each cell's best-ever weights before the final evaluation
    with torch.no_grad():
        for k, v in P.items():
            keep = (best['loss'] < ema.cpu()).to(dev)
            v.copy_(torch.where(keep.view(-1, *([1] * (v.dim() - 1))),
                                best['params'][k].to(dev), v))
    acc = evaluate(P, rel, block, dev, g, n=1_000_000, fwd=fwd)
    accs = evaluate(P, rel, block, dev, g, n=1_000_000, stress=True, fwd=fwd)
    print('final uniform ' + ' '.join(f'{a:.5f}' for a in acc.tolist()), flush=True)
    print('final stress  ' + ' '.join(f'{a:.5f}' for a in accs.tolist()), flush=True)
    b = int(torch.minimum(acc, accs).argmax())
    print('best', b, 'slope', slopes[b].item(), 'freq', freqs[b].item(),
          'acc', acc[b].item(), accs[b].item(), flush=True)
    torch.save({'params': {k: v.detach().cpu() for k, v in P.items()}, 'acc': acc.cpu(),
                'acc_stress': accs.cpu(), 'best': b, 'slopes': slopes, 'sigmas': sigmas,
                'cfg': {'d_ffa': args.d_ffa, 'd_ffb': args.d_ffb}}, args.out)


if __name__ == '__main__':
    main()
