"""Stage 1: cold-train a wide 'parent' (all 25 values per member free) as an
ensemble of E independent members on a shared batch.

Usage:  python train_parent.py --E 1024 --steps 8000 --out runs/p0.pt
"""
import argparse, json, os, time
import torch
import core


def init_params(E, U, device, seed):
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    def rn(*s):
        return torch.randn(*s, generator=g, device=device)

    def ru(*s):
        return torch.rand(*s, generator=g, device=device)

    par = {
        'code':    rn(E, 10) * 0.8,
        'bank_w':  torch.where(ru(E, U) < 0.5, -1.0, 1.0) * (0.5 + 2.5 * ru(E, U)),
        'knee':    rn(E, U) * 1.5,
        'key_w':   rn(E, U) * 2.0,
        'val_w':   rn(E, U) * 1.0,
        'val_b':   rn(E) * 0.1,
        'q':       torch.ones(E, device=device),
        'lam':     -(0.3 + 2.5 * ru(E)),
        'carry_w': rn(E) * 0.8,
        'fold':    rn(E) * 0.8,
        'res_b':   torch.zeros(E, device=device),
        'ls':      torch.ones(E, device=device),
    }
    for v in par.values():
        v.requires_grad_(True)
    return par


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--E', type=int, default=1024)
    ap.add_argument('--U', type=int, default=2)
    ap.add_argument('--B', type=int, default=256)
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--lr', type=float, default=0.012)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--warm_n1', type=float, default=0.25, help='fraction of steps on n=1 only')
    ap.add_argument('--places', type=str, default='1,2,3,5,8')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eval_every', type=int, default=500)
    ap.add_argument('--eval_B', type=int, default=4096)
    ap.add_argument('--ls_loss', type=float, default=1.0)
    ap.add_argument('--out', type=str, required=True)
    a = ap.parse_args()

    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    places = [int(x) for x in a.places.split(',')]
    par = init_params(a.E, a.U, dev, a.seed)
    keys = list(par.keys())
    tensors = [par[k] for k in keys]

    opt = torch.optim.AdamW(tensors, lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps,
                                                pct_start=0.15)
    g = torch.Generator(device=dev); g.manual_seed(a.seed + 12345)
    ge = torch.Generator(device=dev); ge.manual_seed(999)

    # cached masks per P
    mk = {n: core.masks(n + 2, dev) for n in set(places) | {1}}
    ev = {n: core.held_set(a.eval_B, n, ge, dev) for n in places}

    best = {k: par[k].detach().clone() for k in keys}
    best_score = torch.zeros(a.E, device=dev)
    t0 = time.time()
    n1_steps = int(a.warm_n1 * a.steps)
    log = []

    for step in range(a.steps):
        n = 1 if step < n1_steps else places[step % len(places)]
        da, db, tgt = core.sample(a.B, n, g, dev)
        d, ms, mi = mk[n]
        loss, _ = core.loss_and_acc(par, da, db, tgt, d, ms, mi, ls_loss=a.ls_loss)
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        with torch.no_grad():
            sq = sum((p.grad.reshape(a.E, -1) ** 2).sum(1) for p in tensors)
            sc = (a.clip / (sq.sqrt() + 1e-12)).clamp(max=1.0)
            for p in tensors:
                p.grad.mul_(sc.view(-1, *([1] * (p.dim() - 1))))
        opt.step()
        sched.step()

        if (step + 1) % a.eval_every == 0 or step == a.steps - 1:
            with torch.no_grad():
                accs = []
                for n_ in places:
                    d_, ms_, mi_ = mk[n_]
                    lg = core.forward(par, ev[n_][0], ev[n_][1], d_, ms_, mi_)
                    ok = (lg[:, :, 1:, :].argmax(-1) == ev[n_][2][None, :, 1:]).all(-1).float().mean(-1)
                    accs.append(ok)
                A = torch.stack(accs)                       # [len(places), E]
                score = A.mean(0) * 0.5 + A.min(0).values * 0.5
                imp = score > best_score
                best_score = torch.where(imp, score, best_score)
                for k in keys:
                    m = imp.view(-1, *([1] * (par[k].dim() - 1)))
                    best[k] = torch.where(m, par[k].detach(), best[k])
                nperf = int((A.min(0).values >= 0.9999).sum())
                rec = dict(step=step + 1, loss=float(loss.mean()),
                           acc={str(n_): float(A[i].max()) for i, n_ in enumerate(places)},
                           n_perfect=nperf, best_max=float(best_score.max()),
                           secs=round(time.time() - t0, 1))
                log.append(rec)
                print(json.dumps(rec), flush=True)

    # final scoring of the snapshots
    with torch.no_grad():
        accs = []
        for n_ in places:
            d_, ms_, mi_ = mk[n_]
            lg = core.forward(best, ev[n_][0], ev[n_][1], d_, ms_, mi_)
            ok = (lg[:, :, 1:, :].argmax(-1) == ev[n_][2][None, :, 1:]).all(-1).float().mean(-1)
            accs.append(ok)
        A = torch.stack(accs)
        minacc = A.min(0).values
    order = torch.argsort(minacc, descending=True)
    print(f'top10 min-acc: {[round(float(minacc[i]),5) for i in order[:10]]}')
    print(f'members >= 0.9999 on all places: {int((minacc >= 0.9999).sum())}')
    torch.save(dict(par={k: best[k].cpu() for k in keys}, minacc=minacc.cpu(),
                    cfg=vars(a), places=places, log=log), a.out)
    print('saved', a.out)


if __name__ == '__main__':
    main()
