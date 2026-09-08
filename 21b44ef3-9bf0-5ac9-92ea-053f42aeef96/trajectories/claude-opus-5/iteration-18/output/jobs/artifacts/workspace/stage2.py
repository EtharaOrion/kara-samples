"""Stage 2: canonicalise trained parents into the shipped form and fine-tune
the remaining free values (code[1..9], knee[0..1], fold) as an ensemble.

The shipped form keeps as buffers only quantities that are either exact gauge
freedoms of the model (the code origin, the residual scale) or sharpness
constants with a wide admissible band (bank slope, key contrast, recency).
Everything that carries arithmetic content -- the ten digit codes, the bank
thresholds, the mod-10 fold -- stays a learned parameter.
"""
import argparse, json, math, os, time
import torch
import core

SHIP_BUF = dict(bank_w=(8.0, 8.0), key_w=(-400.0, 400.0), val_w=(0.0, 1.0),
                val_b=0.0, q=1.0, lam=-12.0, carry_w=1.0, res_b=0.0, ls=1.0)


def canonicalise(p):
    """p: dict of [E,...] parent tensors -> (code [E,10], knee [E,U], fold [E])."""
    p = {k: v.clone().double() for k, v in p.items()}
    E, U = p['bank_w'].shape

    # 1. fold the key scale q into key_w
    p['key_w'] = p['key_w'] * p['q'][:, None]

    # 2. make every bank slope positive:  clamp(-t,0,1) = 1 - clamp(t+1,0,1)
    neg = p['bank_w'] < 0
    w = p['bank_w'].abs().clamp(min=1e-9)
    knee = torch.where(neg, p['knee'] - 1.0 / w, p['knee'])
    p['val_b'] = p['val_b'] + (p['val_w'] * neg).sum(1)
    p['key_w'] = torch.where(neg, -p['key_w'], p['key_w'])
    p['val_w'] = torch.where(neg, -p['val_w'], p['val_w'])
    p['bank_w'], p['knee'] = w, knee

    # 3. order units by threshold
    order = torch.argsort(p['knee'], dim=1)
    for k in ('bank_w', 'knee', 'key_w', 'val_w'):
        p[k] = torch.gather(p[k], 1, order)

    # 4. renormalise the value stream so absorb -> 0, generate -> 1.  Which end
    # of the code ramp is "absorb" depends on the sign the member learned, so
    # read the value off the two extreme tokens (0,0) and (9,9) rather than
    # assuming an orientation.
    def val_at(x):
        u = torch.clamp(p['bank_w'] * (x[:, None] - p['knee']), 0.0, 1.0)
        return (p['val_w'] * u).sum(1) + p['val_b']

    v_lo = val_at(2 * p['code'][:, 0])            # a+b = 0  -> absorb
    v_hi = val_at(2 * p['code'][:, 9])            # a+b = 18 -> generate
    V = v_hi - v_lo
    res_b = p['res_b'] + (p['carry_w'] + p['fold']) * v_lo
    carry_w = p['carry_w'] * V
    fold = p['fold'] * V

    # 5. residual-scale gauge: make carry_w == 1
    beta = 1.0 / carry_w
    code = p['code'] * beta[:, None]
    knee = p['knee'] * beta[:, None]
    fold = fold * beta
    res_b = res_b * beta
    # beta<0 flips the code, so re-flip the bank to keep its slope positive
    negb = beta < 0
    knee = torch.where(negb[:, None], knee - 1.0 / p['bank_w'], knee)
    knee = torch.gather(knee, 1, torch.argsort(knee, dim=1))

    # 6. translation gauge: code[0] == 0 (knees live in x-space, so shift by 2t)
    t = code[:, :1].clone()
    code = code - t
    knee = knee - 2.0 * t
    res_b = res_b + t[:, 0]
    return code.float(), knee.float(), fold.float(), res_b.float()


def ship_par(code_free, knee, fold, dev, code_zero=None):
    E = code_free.shape[0]
    z = torch.zeros(E, 1, device=dev) if code_zero is None else code_zero
    par = dict(code=torch.cat([z, code_free], 1), knee=knee, fold=fold)
    for k, v in SHIP_BUF.items():
        t = torch.tensor(v, device=dev, dtype=torch.float32)
        par[k] = t.expand(E, *t.shape).contiguous() if t.dim() else t.expand(E).contiguous()
    return par


def sat_penalty(par, target=1.5):
    """Every one of the 100 digit pairs must drive every bank unit into
    saturation (gate exactly 0 or exactly 1).  Class-agnostic: it says 'be
    decisive', not which way."""
    code = par['code']                                     # [E,10]
    x = code[:, :, None] + code[:, None, :]                # [E,10,10]
    x = x.reshape(code.shape[0], -1)                       # [E,100]
    e = par['bank_w'][:, None, :] * (x[:, :, None] - par['knee'][:, None, :])
    slack = torch.maximum(-e, e - 1.0)
    return torch.relu(target - slack).mean(dim=(1, 2))


def margin_loss(par, da, db, tgt, dist, ms, mi, want=0.45):
    w = {}
    core.forward(par, da, db, dist, ms, mi, want=w)
    r = w['r'][:, :, 1:]                                   # [E,B,P-1]
    code = par['code'][:, None, None, :]
    d = torch.sqrt((r[..., None] - code) ** 2 + 1e-12)     # [E,B,P-1,10]
    t = tgt[None, :, 1:, None].expand(d.shape[0], -1, -1, 1)
    dt = d.gather(-1, t).squeeze(-1)
    dw = d.scatter(-1, t, float('inf')).min(-1).values
    m = dw - dt
    return torch.relu(want - m).mean(dim=(1, 2)), m.reshape(m.shape[0], -1).min(-1).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=str, required=True)
    ap.add_argument('--topk', type=int, default=64)
    ap.add_argument('--rep', type=int, default=8)
    ap.add_argument('--noise', type=float, default=0.02)
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--B', type=int, default=256)
    ap.add_argument('--lr', type=float, default=0.004)
    ap.add_argument('--places', type=str, default='1,2,3,5,8')
    ap.add_argument('--sat_w', type=float, default=0.5)
    ap.add_argument('--sat_t', type=float, default=1.5)
    ap.add_argument('--want', type=float, default=0.45)
    ap.add_argument('--eval_every', type=int, default=500)
    ap.add_argument('--eval_B', type=int, default=4096)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', type=str, required=True)
    a = ap.parse_args()
    dev = 'cuda'
    ck = torch.load(a.src, map_location=dev, weights_only=False)
    par0 = {k: v.to(dev) for k, v in ck['par'].items()}
    minacc = ck['minacc'].to(dev)
    sel = torch.argsort(minacc, descending=True)[:a.topk]
    print(f'selected {len(sel)} parents, acc range '
          f'{float(minacc[sel].min()):.4f}..{float(minacc[sel].max()):.4f}')
    par0 = {k: v[sel] for k, v in par0.items()}
    code, knee, fold, res_b = canonicalise(par0)
    print(f'residual offset after canonicalisation: |res_b| median '
          f'{float(res_b.abs().median()):.4f} max {float(res_b.abs().max()):.4f}')

    idx = torch.arange(len(sel), device=dev).repeat_interleave(a.rep)
    code, knee, fold = code[idx], knee[idx], fold[idx]
    E = code.shape[0]
    g = torch.Generator(device=dev); g.manual_seed(a.seed)
    jit = lambda t, s: t + s * torch.randn(t.shape, generator=g, device=dev)
    rep_i = torch.arange(E, device=dev) % a.rep
    m = (rep_i > 0).float()[:, None]
    code = code + m * a.noise * torch.randn(code.shape, generator=g, device=dev)
    knee = knee + m * a.noise * torch.randn(knee.shape, generator=g, device=dev)
    fold = fold + m[:, 0] * a.noise * torch.randn(fold.shape, generator=g, device=dev)

    cf = code[:, 1:].clone().requires_grad_(True)
    kn = knee.clone().requires_grad_(True)
    fo = fold.clone().requires_grad_(True)
    tensors = [cf, kn, fo]
    opt = torch.optim.AdamW(tensors, lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps,
                                                pct_start=0.1)
    places = [int(x) for x in a.places.split(',')]
    mk = {n: core.masks(n + 2, dev) for n in places}
    gd = torch.Generator(device=dev); gd.manual_seed(a.seed + 1)
    ge = torch.Generator(device=dev); ge.manual_seed(4242)
    ev = {n: core.held_set(a.eval_B, n, ge, dev) for n in places}

    def score_now():
        with torch.no_grad():
            p = ship_par(cf, kn, fo, dev)
            accs = []
            for n_ in places:
                d_, ms_, mi_ = mk[n_]
                lg = core.forward(p, ev[n_][0], ev[n_][1], d_, ms_, mi_)
                accs.append((lg[:, :, 1:, :].argmax(-1) == ev[n_][2][None, :, 1:]).all(-1).float().mean(-1))
            A = torch.stack(accs)
            d8, ms8, mi8 = mk[places[-1]]
            _, mg = margin_loss(p, *ev[places[-1]], d8, ms8, mi8, want=a.want)
            sl = 1.0 - sat_penalty(p, target=a.sat_t) / max(a.sat_t, 1e-9)
            return A.min(0).values, mg, sl

    acc0, mg0, sl0 = score_now()
    print(f'after substitution: max acc {float(acc0.max()):.4f}, '
          f'#>0.5 {int((acc0>0.5).sum())}, #=1 {int((acc0>=0.99999).sum())}')

    best = dict(code=cf.detach().clone(), knee=kn.detach().clone(), fold=fo.detach().clone())
    best_s = torch.full((E,), -1e9, device=dev)
    t0 = time.time()
    for step in range(a.steps):
        n = places[step % len(places)]
        da, db, tgt = core.sample(a.B, n, gd, dev)
        d_, ms_, mi_ = mk[n]
        p = ship_par(cf, kn, fo, dev)
        ml, _ = margin_loss(p, da, db, tgt, d_, ms_, mi_, want=a.want)
        sp = sat_penalty(p, target=a.sat_t)
        loss = ml + a.sat_w * sp
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
            acc, mg, sl = score_now()
            s = acc * 10.0 + torch.clamp(mg, max=a.want) + torch.clamp(sl, max=0.9)
            imp = s > best_s
            best_s = torch.where(imp, s, best_s)
            best['code'] = torch.where(imp[:, None], cf.detach(), best['code'])
            best['knee'] = torch.where(imp[:, None], kn.detach(), best['knee'])
            best['fold'] = torch.where(imp, fo.detach(), best['fold'])
            print(json.dumps(dict(step=step + 1, loss=float(loss.mean()),
                                  n_exact=int((acc >= 0.99999).sum()),
                                  max_acc=round(float(acc.max()), 5),
                                  best_margin=round(float(mg[acc >= 0.99999].max()) if int((acc >= 0.99999).sum()) else -1, 4),
                                  secs=round(time.time() - t0, 1))), flush=True)

    p = ship_par(best['code'], best['knee'], best['fold'], dev)
    with torch.no_grad():
        accs = []
        for n_ in places:
            d_, ms_, mi_ = mk[n_]
            lg = core.forward(p, ev[n_][0], ev[n_][1], d_, ms_, mi_)
            accs.append((lg[:, :, 1:, :].argmax(-1) == ev[n_][2][None, :, 1:]).all(-1).float().mean(-1))
        acc = torch.stack(accs).min(0).values
        d8, ms8, mi8 = mk[places[-1]]
        _, mg = margin_loss(p, *ev[places[-1]], d8, ms8, mi8, want=a.want)
        sl = 1.0 - sat_penalty(p, target=a.sat_t) / max(a.sat_t, 1e-9)
    print(f'FINAL: n_exact {int((acc>=0.99999).sum())}/{E}  best margin '
          f'{float(mg[acc>=0.99999].max()) if int((acc>=0.99999).sum()) else -1:.4f}')
    torch.save(dict(code=best['code'].cpu(), knee=best['knee'].cpu(), fold=best['fold'].cpu(),
                    acc=acc.cpu(), margin=mg.cpu(), sat=sl.cpu(), cfg=vars(a),
                    buf=SHIP_BUF), a.out)
    print('saved', a.out)


if __name__ == '__main__':
    main()
