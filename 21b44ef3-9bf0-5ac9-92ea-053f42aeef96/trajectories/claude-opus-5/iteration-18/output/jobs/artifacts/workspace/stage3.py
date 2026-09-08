"""Stage 3: collapse the two bank thresholds into one shared learned threshold.

The stage-2 model spends two parameters on the clamp bank: one threshold that
separates "this place can pass a carry through" from "it cannot", and a second
that separates "this place makes a carry" from "it does not".  Those two
thresholds sit one code step apart, which is a *width*, not an independent
quantity -- so the bank can keep one learned threshold k and get the second
boundary from a unit whose slope is gentler by a fixed factor:

    u0 = clamp(  8 * (x - k), 0, 1)      sharp   -> saturates just above k
    u1 = clamp( W1 * (x - k), 0, 1)      gentle  -> saturates 1/W1 above k

    value = u0                 (does this place stand at or above the boundary)
    key   = 400 * (u1 - u0)    (notched wherever the two units disagree)

The model must still learn where k goes and how wide a code step is, because
the notch only lands on the pass-through class if the learned codes arrange
x(a+b=9) inside the ramp of u1 and x(a+b>=10) beyond its end.  W1 is a slope
constant with a wide admissible band (see the band study), in the same family
as the sharpness 8, the key contrast 400 and the recency bias -12 that stage 2
already carried as buffers.  Learned: 9 codes, 1 threshold, 1 fold = 11.

The structural penalty below never mentions which digit pairs belong to which
class.  It asks two class-agnostic things of every one of the 100 pairs: that
the sharp unit be saturated, and that the two units either agree or disagree
decisively.
"""
import argparse, json, time
import torch
import core
from stage2 import margin_loss


def ship_buf(W1):
    return dict(bank_w=(8.0, float(W1)), key_w=(-400.0, 400.0), val_w=(1.0, 0.0),
                val_b=0.0, q=1.0, lam=-12.0, carry_w=1.0, res_b=0.0, ls=1.0)


def ship_par(code_free, knee, fold, W1, dev):
    """knee is [E,1]; both bank units read the same learned threshold."""
    E = code_free.shape[0]
    z = torch.zeros(E, 1, device=dev)
    par = dict(code=torch.cat([z, code_free], 1), knee=knee.expand(E, 2), fold=fold)
    for k, v in ship_buf(W1).items():
        t = torch.tensor(v, device=dev, dtype=torch.float32)
        par[k] = t.expand(E, *t.shape).contiguous() if t.dim() else t.expand(E).contiguous()
    return par


def struct_penalty(par, t0=1.5, tol=0.02, dmin=0.35):
    """Class-agnostic decisiveness penalty over all 100 digit pairs."""
    code = par['code']                                        # [E,10]
    x = (code[:, :, None] + code[:, None, :]).reshape(code.shape[0], -1)
    k = par['knee'][:, :1]
    e0 = par['bank_w'][:, 0:1] * (x - k)
    slack = torch.maximum(-e0, e0 - 1.0)                      # distance to saturation
    pen_sat = torch.relu(t0 - slack).mean(1)
    g = torch.clamp(par['bank_w'][:, 1:2] * (x - k), 0., 1.) - torch.clamp(e0, 0., 1.)
    pen_gap = torch.minimum(torch.relu(-g - tol), torch.relu(g + dmin)).mean(1)
    return pen_sat + pen_gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=str, required=True)
    ap.add_argument('--W1', type=float, default=1.0)
    ap.add_argument('--topk', type=int, default=32)
    ap.add_argument('--rep', type=int, default=16)
    ap.add_argument('--noise', type=float, default=0.02)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--B', type=int, default=256)
    ap.add_argument('--lr', type=float, default=0.003)
    ap.add_argument('--places', type=str, default='1,2,3,5,8')
    ap.add_argument('--pen_w', type=float, default=0.5)
    ap.add_argument('--want', type=float, default=0.45)
    ap.add_argument('--eval_every', type=int, default=500)
    ap.add_argument('--eval_B', type=int, default=4096)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--out', type=str, required=True)
    a = ap.parse_args()
    dev = 'cuda'
    ck = torch.load(a.src, map_location=dev, weights_only=False)
    acc0, mg0 = ck['acc'].to(dev), ck['margin'].to(dev)
    sel = torch.nonzero(acc0 >= 0.99999).flatten()
    sel = sel[torch.argsort(mg0[sel], descending=True)][:a.topk]
    print(f'selected {len(sel)} stage-2 members, margin range '
          f'{float(mg0[sel].min()):.4f}..{float(mg0[sel].max()):.4f}')
    code = ck['code'].to(dev)[sel]                    # [S,9] free codes
    knee2 = ck['knee'].to(dev)[sel]                   # [S,2]
    fold = ck['fold'].to(dev)[sel]
    # one shared threshold, initialised halfway between the two it replaces
    knee = knee2.mean(1, keepdim=True)

    idx = torch.arange(len(sel), device=dev).repeat_interleave(a.rep)
    code, knee, fold = code[idx], knee[idx], fold[idx]
    E = code.shape[0]
    g = torch.Generator(device=dev); g.manual_seed(a.seed)
    m = (torch.arange(E, device=dev) % a.rep > 0).float()[:, None]
    code = code + m * a.noise * torch.randn(code.shape, generator=g, device=dev)
    knee = knee + m * a.noise * torch.randn(knee.shape, generator=g, device=dev)
    fold = fold + m[:, 0] * a.noise * torch.randn(fold.shape, generator=g, device=dev)

    cf = code.clone().requires_grad_(True)
    kn = knee.clone().requires_grad_(True)
    fo = fold.clone().requires_grad_(True)
    tensors = [cf, kn, fo]
    opt = torch.optim.AdamW(tensors, lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    steps = max(a.steps, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps,
                                                pct_start=0.1)
    places = [int(x) for x in a.places.split(',')]
    mk = {n: core.masks(n + 2, dev) for n in places}
    gd = torch.Generator(device=dev); gd.manual_seed(a.seed + 1)
    ge = torch.Generator(device=dev); ge.manual_seed(4242)
    ev = {n: core.held_set(a.eval_B, n, ge, dev) for n in places}

    def score_now(c=None, k_=None, f=None):
        with torch.no_grad():
            p = ship_par(cf if c is None else c, kn if k_ is None else k_,
                         fo if f is None else f, a.W1, dev)
            accs = []
            for n_ in places:
                d_, ms_, mi_ = mk[n_]
                lg = core.forward(p, ev[n_][0], ev[n_][1], d_, ms_, mi_)
                accs.append((lg[:, :, 1:, :].argmax(-1) == ev[n_][2][None, :, 1:])
                            .all(-1).float().mean(-1))
            A = torch.stack(accs).min(0).values
            d8, ms8, mi8 = mk[places[-1]]
            _, mg = margin_loss(p, *ev[places[-1]], d8, ms8, mi8, want=a.want)
            return A, mg, struct_penalty(p)

    acc, mg, pen = score_now()
    print(f'after substitution (W1={a.W1}): #=1 {int((acc>=0.99999).sum())}/{E}, '
          f'max acc {float(acc.max()):.4f}, best margin '
          f'{float(mg[acc>=0.99999].max()) if int((acc>=0.99999).sum()) else -1:.4f}')

    best = dict(code=cf.detach().clone(), knee=kn.detach().clone(), fold=fo.detach().clone())
    best_s = torch.full((E,), -1e9, device=dev)
    t0 = time.time()
    for step in range(a.steps):
        n = places[step % len(places)]
        da, db, tgt = core.sample(a.B, n, gd, dev)
        d_, ms_, mi_ = mk[n]
        p = ship_par(cf, kn, fo, a.W1, dev)
        ml, _ = margin_loss(p, da, db, tgt, d_, ms_, mi_, want=a.want)
        loss = ml + a.pen_w * struct_penalty(p)
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        with torch.no_grad():
            sq = sum((t.grad.reshape(E, -1) ** 2).sum(1) for t in tensors)
            s = (1.0 / (sq.sqrt() + 1e-12)).clamp(max=1.0)
            for t in tensors:
                t.grad.mul_(s.view(-1, *([1] * (t.dim() - 1))))
        opt.step()
        sched.step()

        if (step + 1) % a.eval_every == 0 or step == a.steps - 1:
            acc, mg, pen = score_now()
            s = acc * 10.0 + torch.clamp(mg, max=a.want) - torch.clamp(pen, max=1.0)
            imp = s > best_s
            best_s = torch.where(imp, s, best_s)
            best['code'] = torch.where(imp[:, None], cf.detach(), best['code'])
            best['knee'] = torch.where(imp[:, None], kn.detach(), best['knee'])
            best['fold'] = torch.where(imp, fo.detach(), best['fold'])
            ne = int((acc >= 0.99999).sum())
            print(json.dumps(dict(step=step + 1, loss=round(float(loss.mean()), 5),
                                  n_exact=ne, max_acc=round(float(acc.max()), 5),
                                  best_margin=round(float(mg[acc >= 0.99999].max()), 4) if ne else -1,
                                  secs=round(time.time() - t0, 1))), flush=True)

    acc, mg, pen = score_now(best['code'], best['knee'], best['fold'])
    ne = int((acc >= 0.99999).sum())
    print(f'FINAL: n_exact {ne}/{E}  best margin '
          f'{float(mg[acc>=0.99999].max()) if ne else -1:.4f}')
    torch.save(dict(code=best['code'].cpu(), knee=best['knee'].cpu(),
                    fold=best['fold'].cpu(), acc=acc.cpu(), margin=mg.cpu(),
                    pen=pen.cpu(), cfg=vars(a), buf=ship_buf(a.W1), stage3=True), a.out)
    print('saved', a.out)


if __name__ == '__main__':
    main()
