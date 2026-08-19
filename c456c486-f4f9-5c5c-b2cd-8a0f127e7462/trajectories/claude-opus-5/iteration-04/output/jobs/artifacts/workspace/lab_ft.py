"""Shrink by warm start: keep the trained carry-lookahead front end, apply one
structural cut (tie the readout to the digit code, or drop hidden units), and
re-train the whole model under the new constraint.

Finding the carry-lookahead circuit is an initialisation lottery -- roughly one
random init in tens gets there -- but *given* a working front end, relearning
the output fold is easy.  So instead of re-running the lottery for every smaller
configuration, this trains a population of jittered copies of a converged model.
Everything is still learned by gradient descent; the warm start only decides
where training starts.
"""
import argparse, time
import torch

import lab_b2, lab_d, lab_e, lab_f, lab_g, lab_h, lab_i
from lab import gen_batch, sum_digits, make_bias
from lab_b2 import encode_b, loss_fn, evaluate

torch.backends.cuda.matmul.allow_tf32 = True

FRONT = ['U', 'W1a', 'b1a', 'W2a0', 'W2a1', 'wq', 'bq', 'wv', 'slope']
MOD = {'b': lab_b2, 'd': lab_d, 'e': lab_e}


def build(src, M, cfg, dev, g, args):
    """M jittered copies of src, cut down to cfg's width and head."""
    da, db = cfg['d_ffa'], cfg['d_ffb']
    # rank hidden units by how much they drive their layer's output, keep the top
    a_rank = (src['W2a0'].abs() + src['W2a1'].abs()).argsort(descending=True)
    w2b = src['W2b']
    b_rank = (w2b.abs() if w2b.dim() == 1 else w2b.norm(dim=0)).argsort(descending=True)
    base = {k: v.clone() for k, v in src.items()}
    for k in ('W1a', 'b1a', 'W2a0', 'W2a1'):
        base[k] = base[k][a_rank[:da]]
    for k in ('W1b0', 'W1b1', 'b1b'):
        if k in base:
            base[k] = base[k][b_rank[:db]]
    base['W2b'] = base['W2b'][..., b_rank[:db]]
    grow = db - base['W1b0'].numel()
    if grow > 0:
        # widen the fold MLP.  Cutting elsewhere can be worth paying for a unit
        # here; the new units start near zero so the warm start still applies.
        for k in ('W1b0', 'W1b1', 'b1b'):
            if k in base:
                base[k] = torch.cat([base[k], 0.2 * torch.randn(grow, device=base[k].device,
                                                                generator=g)])
        w = base['W2b']
        base['W2b'] = torch.cat([w, 0.02 * torch.randn((*w.shape[:-1], grow), device=w.device,
                                                       generator=g)], -1)
    if cfg['tied'] and 'Wout' in base:                 # tie readout column 0 to U
        base['Wout1'] = base.pop('Wout')[:, 1].clone()
    if args.arch in ('e', 'f') and 'wq' in base:       # exact re-parametrisation
        base = lab_e.fold(base, cfg['tied'])
    if args.arch == 'f' and base['W2b'].dim() == 2:    # keep one residual write row
        base['W2b'] = base['W2b'][cfg['mlp_to']].clone()
    if args.arch in ('h', 'i'):                       # distance readout: no Wout1, no k2
        base = lab_h.fold(base)
    if args.arch == 'i':                              # attention output projection
        base = lab_i.fold(base)
    if args.arch == 'g' and 'k2' not in base:         # drop the constant feature unit
        base = lab_g.fold(base, unit=int(base['W1a'].abs().argmin()))
    if args.drop_unit >= 0:          # drop a named feature unit, not the lowest-ranked
        keep = [j for j in range(base['W1a'].numel()) if j != args.drop_unit]
        for k in ('W1a', 'b1a', 'W2a0', 'W2a1'):
            base[k] = base[k][keep]
    if args.no_b1b:
        base.pop('b1b', None)
    if args.no_wout1:
        base.pop('Wout1', None); base.pop('Wout', None)

    P = {k: v[None].repeat(M, *([1] * v.dim())).to(dev) for k, v in base.items()}
    jit = torch.tensor((args.jitters * M)[:M], device=dev)
    for k in P:
        noise = torch.randn(P[k].shape, device=dev, generator=g)
        P[k] = P[k] * (1 + jit.view(-1, *([1] * (P[k].dim() - 1))) * noise)
    if args.reinit_out:
        # half the cells relearn the output block from scratch -- after a
        # structural cut the old output weights can be the wrong starting point
        mod = {'e': lab_e, 'f': lab_f, 'g': lab_g, 'h': lab_h, 'i': lab_i}.get(args.arch) or (lab_d if cfg['tied'] else lab_b2)
        fresh = mod.init_params(M, dev, cfg, g)
        for k in P:
            if k not in FRONT:
                P[k][M // 2:] = fresh[k][M // 2:].detach()
    return {k: v.detach().requires_grad_() for k, v in P.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='/workspace/ckpt_b2.pt')
    ap.add_argument('--index', type=int, default=-1)
    ap.add_argument('--arch', default='b', choices=['b', 'e', 'f', 'g', 'h', 'i'])  # e: no wq/wv, f: 1 residual write
    ap.add_argument('--mlp_to', type=int, default=0)
    ap.add_argument('--drop_unit', type=int, default=-1)
    ap.add_argument('--no_b1b', action='store_true')   # fold MLP without its bias
    ap.add_argument('--no_wout1', action='store_true')  # readout without its carry column
    ap.add_argument('--tied', action='store_true')
    ap.add_argument('--d_ffa', type=int, default=5)
    ap.add_argument('--d_ffb', type=int, default=4)
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--bs', type=int, default=2048)
    ap.add_argument('--lr', type=float, default=1.5e-3)
    ap.add_argument('--M', type=int, default=16)
    ap.add_argument('--sigma', type=float, default=0.02)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--reinit_out', action='store_true')
    ap.add_argument('--jitters', default='0,0.02,0.05,0.12')
    ap.add_argument('--out', default='/workspace/ckpt_ft.pt')
    args = ap.parse_args()
    args.jitters = [float(x) for x in args.jitters.split(',')]

    dev = 'cuda'
    g = torch.Generator(device=dev); g.manual_seed(args.seed)
    ck = torch.load(args.src, map_location='cpu')
    i = ck['best'] if args.index < 0 else args.index
    src = {k: v[i].to(dev) for k, v in ck['params'].items()}
    M = args.M
    cfg = {'d_ffa': args.d_ffa, 'd_ffb': args.d_ffb, 'tied': args.tied,
           'mlp_to': args.mlp_to, 'slope_init': torch.zeros(M)}
    P = build(src, M, cfg, dev, g, args)
    fwd = (lab_i.forward if args.arch == 'i' else
           lab_h.forward if args.arch == 'h' else
           lab_g.make_forward(args.mlp_to, not args.no_wout1) if args.arch == 'g' else
           lab_f.make_forward(args.mlp_to) if args.arch == 'f' else
           lab_e.forward if args.arch == 'e' else
           lab_d.forward if args.tied else lab_b2.forward)
    npar = sum(v[0].numel() for v in P.values())
    print(f'warm start from {args.src}[{i}] -> arch {args.arch} d_ffa {args.d_ffa} '
          f'd_ffb {args.d_ffb} tied {args.tied}: {npar} parameters, lr {args.lr}', flush=True)

    rel, block = make_bias(dev)
    opt = torch.optim.AdamW(list(P.values()), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps,
                                                pct_start=0.1)
    sig0 = torch.full((M,), args.sigma, device=dev)
    ema = torch.full((M,), 2.3, device=dev)
    best = {'loss': torch.full((M,), 9.9),
            'params': {k: v.detach().cpu().clone() for k, v in P.items()}}

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

        if step % 2000 == 0 or step == args.steps - 1:
            print(f'step {step} {time.time()-t0:.0f}s ema ' +
                  ' '.join(f'{l:.4f}' for l in ema.tolist()), flush=True)
        if step % 2000 == 1999:                       # keep best-ever weights
            e = ema.cpu()
            for c in (e < best['loss']).nonzero().flatten().tolist():
                for k, v in P.items():
                    best['params'][k][c] = v[c].detach().cpu()
                best['loss'][c] = e[c]

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
    print('best', b, 'params', npar, 'acc', acc[b].item(), accs[b].item(), flush=True)
    torch.save({'params': {k: v.detach().cpu() for k, v in P.items()}, 'acc': acc.cpu(),
                'acc_stress': accs.cpu(), 'best': b,
                'cfg': {'d_ffa': args.d_ffa, 'd_ffb': args.d_ffb, 'tied': args.tied,
                        'arch': args.arch, 'mlp_to': args.mlp_to}}, args.out)


if __name__ == '__main__':
    main()
