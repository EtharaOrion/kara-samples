import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from core import Ens, sample_batch, easy_batch, curric_batch, stress_batch, targets, to_onehot, NPOS

p = argparse.ArgumentParser()
p.add_argument('--M', type=int, default=64)
p.add_argument('--B', type=int, default=1024)
p.add_argument('--steps', type=int, default=60000)
p.add_argument('--lr', type=float, default=3.5e-3)
p.add_argument('--warm', type=float, default=0.05)
p.add_argument('--noise', type=float, default=0.2)
p.add_argument('--noise_end', type=float, default=0.6)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--restart_until', type=float, default=0.35)
p.add_argument('--restart_every', type=int, default=2000)
p.add_argument('--restart_frac', type=float, default=0.25)
p.add_argument('--out', type=str, default='/workspace/lab/ckpt.pt')
p.add_argument('--init', type=str, default='')
p.add_argument('--fold_sign', type=str, default='')
p.add_argument('--drop', type=str, default='')
p.add_argument('--fix_slope', type=float, default=-1.0)
p.add_argument('--width', type=int, default=3)
p.add_argument('--untied', type=int, default=0)
p.add_argument('--easy', type=float, default=0.06)
p.add_argument('--cur_end', type=float, default=0.40)
p.add_argument('--pmax', type=float, default=0.35)
p.add_argument('--res_every', type=int, default=500)
p.add_argument('--res_until', type=float, default=0.7)
p.add_argument('--jitter', type=float, default=0.0)
p.add_argument('--tie_lambda', type=float, default=0.0)
p.add_argument('--tag', type=str, default='run')
a = p.parse_args()

dev = 'cuda'
torch.backends.cuda.matmul.allow_tf32 = False
fold = [float(x) for x in a.fold_sign.split(',')] if a.fold_sign else None
drop = tuple(x for x in a.drop.split(',') if x)
fixsl = None if a.fix_slope < 0 else a.fix_slope
model = Ens(a.M, seed=a.seed, fold_sign=fold, drop=drop, width=a.width, untied=bool(a.untied), fix_slope=fixsl).to(dev)
NP = model.nparams_single()
print(f'[{a.tag}] params/model = {NP}  M={a.M}', flush=True)

if a.init:
    st = torch.load(a.init, map_location=dev)
    src = st['params']
    for k, v in model.named_parameters():
        if k in src:
            w = src[k].to(dev)
            if w.shape[0] == 1:
                w = w.repeat(a.M, *([1] * (w.dim() - 1)))
            with torch.no_grad():
                v.copy_(w[:a.M] if w.shape[0] >= a.M else w.repeat(a.M // w.shape[0] + 1, 1)[:a.M])
    if a.jitter > 0:
        with torch.no_grad():
            for k, v in model.named_parameters():
                n = torch.randn_like(v) * a.jitter
                n[0] = 0                       # cell 0 keeps the exact warm start
                v.mul_(1 + n)
    print(f'[{a.tag}] warm-started from {a.init}', flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.98), weight_decay=0.0)
g = torch.Generator(device=dev).manual_seed(a.seed + 12345)
geval = torch.Generator(device=dev).manual_seed(999)

# fixed eval sets
ea, eb = sample_batch(4096, dev, geval)
et = targets(ea, eb, dev); eoh = to_onehot(ea, eb, dev)
sa, sb = stress_batch(4096, dev, geval)
st_ = targets(sa, sb, dev); soh = to_onehot(sa, sb, dev)


def lr_at(i):
    f = i / a.steps
    if f < a.warm:
        return a.lr * (0.05 + 0.95 * f / a.warm)
    t = (f - a.warm) / (1 - a.warm)
    return a.lr * (0.5 * (1 + math.cos(math.pi * t)) * 0.98 + 0.02)


def noise_at(i):
    f = i / a.steps
    return max(0.0, a.noise * (1 - f / a.noise_end)) if f < a.noise_end else 0.0


@torch.no_grad()
def evaluate():
    res = {}
    for nm, (oh, tt) in (('uni', (eoh, et)), ('str', (soh, st_))):
        lg = model(oh)
        pr = lg.argmax(-1)
        ok = (pr == tt.unsqueeze(0))
        res[nm] = ok.all(-1).float().mean(-1)          # (M,) exact match
        res[nm + 'd'] = ok.float().mean(dim=(1, 2))    # (M,) per-digit
    return res


def reinit(idx, seed):
    fresh = Ens(len(idx), seed=seed, fold_sign=fold, drop=drop, width=a.width, untied=bool(a.untied), fix_slope=fixsl).to(dev)
    with torch.no_grad():
        d = dict(fresh.named_parameters())
        for k, v in model.named_parameters():
            v[idx] = d[k].data
            stt = opt.state.get(v, None)
            if stt:
                stt['exp_avg'][idx] = 0
                stt['exp_avg_sq'][idx] = 0


@torch.no_grad()
def resurrect(oh):
    """A relu unit that is always-on or always-off across the batch contributes no
    threshold. Re-place its threshold at a random point inside the observed range of z."""
    z, h = model.zh(oh)
    act = (h > 0).float().mean(dim=(1, 2))                       # (M,W)
    dead = (act < 0.02) | (act > 0.98)
    if not dead.any():
        return 0
    zmin = z.amin(dim=(1, 2)).view(-1, 1); zmax = z.amax(dim=(1, 2)).view(-1, 1)
    t = zmin + (zmax - zmin) * torch.rand_like(model.b1a)
    newb = -model.w1a() * t
    model.b1a.data = torch.where(dead, newb, model.b1a.data)
    stt = opt.state.get(model.b1a, None)
    if stt:
        stt['exp_avg'] = torch.where(dead, torch.zeros_like(stt['exp_avg']), stt['exp_avg'])
        stt['exp_avg_sq'] = torch.where(dead, torch.zeros_like(stt['exp_avg_sq']), stt['exp_avg_sq'])
    return int(dead.sum())


ema = None
nres = 0
t0 = time.time()
best = {'acc': -1.0}
for it in range(a.steps + 1):
    for gp in opt.param_groups:
        gp['lr'] = lr_at(it)
    # curriculum: carry-free -> short carry chains -> long chains -> full stress mix
    f = it / a.steps
    w_easy = max(0.0, 0.9 * (1 - f / a.easy)) if a.easy > 0 else 0.0
    w_full = min(1.0, max(0.0, (f - a.cur_end) / 0.20))
    w_cur = max(0.0, 1.0 - w_easy - w_full)
    pprop = a.pmax * min(1.0, max(0.0, (f - a.easy) / max(1e-9, a.cur_end - a.easy)))
    n_e = int(a.B * w_easy); n_c = int(a.B * w_cur); n_f = a.B - n_e - n_c
    parts_a, parts_b = [], []
    if n_e: 
        x, y_ = easy_batch(n_e, dev, g); parts_a.append(x); parts_b.append(y_)
    if n_c:
        x, y_ = curric_batch(n_c, dev, g, pprop); parts_a.append(x); parts_b.append(y_)
    if n_f:
        x, y_ = sample_batch(max(4, (n_f // 4) * 4), dev, g); parts_a.append(x[:n_f]); parts_b.append(y_[:n_f])
    da = torch.cat(parts_a, 0); db = torch.cat(parts_b, 0)
    tt = targets(da, db, dev)
    oh = to_onehot(da, db, dev)
    logits = model(oh, noise=noise_at(it))
    lp = logits - logits.logsumexp(-1, keepdim=True)
    ll = torch.gather(lp, 3, tt.view(1, -1, NPOS, 1).expand(model.M, -1, -1, -1)).squeeze(-1)
    per = -ll.mean(dim=(1, 2))                       # (M,)
    loss = per.sum()
    if a.tie_lambda > 0 and model.untied:
        lam = a.tie_lambda * min(1.0, it / (0.6 * a.steps)) ** 2
        loss = loss + lam * ((model.U - model.V) ** 2).mean(-1).sum()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    ema = per.detach() if ema is None else 0.99 * ema + 0.01 * per.detach()

    if a.res_every and it % a.res_every == 0 and it < a.res_until * a.steps:
        nres += resurrect(oh)

    if it % a.restart_every == 0 and it > 0:
        # evolutionary restarts: the worst cells by EMA loss are almost always stuck
        # in the analogue-carry basin; replace them with fresh inits.
        if it < a.restart_until * a.steps and not a.init:
            nbad = max(1, int(a.M * a.restart_frac))
            bad = torch.argsort(ema, descending=True)[:nbad]
            reinit(bad, seed=a.seed + 1000 + it)
            print(f'[{a.tag}] step {it}: restarted {nbad}/{a.M} cells '
                  f'(ema loss {ema.min().item():.4f}..{ema.max().item():.4f})', flush=True)
            with torch.no_grad():
                ema[bad] = ema.median()

    if it % 2000 == 0:
        acc = evaluate()
        sc = acc['unid'] + acc['strd']
        k = int(sc.argmax())
        cur = min(acc['uni'][k].item(), acc['str'][k].item())
        print(f'[{a.tag}] {it:6d} lr {lr_at(it):.2e} loss {per.min().item():.4f} '
              f'best[{k}] uni {acc["uni"][k]:.4f}/{acc["unid"][k]:.4f} '
              f'str {acc["str"][k]:.4f}/{acc["strd"][k]:.4f} '
              f'n>.99 {(torch.minimum(acc["uni"], acc["str"]) > 0.99).sum().item()} '
              f'res {nres} {(time.time()-t0)/max(it,1)*1000:.1f}ms/it', flush=True)
        if cur > best['acc']:
            best = {'acc': cur, 'it': it, 'idx': k,
                    'uni': acc['uni'][k].item(), 'str': acc['str'][k].item()}
            torch.save({'params': {kk: v.detach()[k:k + 1].cpu() for kk, v in model.named_parameters()},
                        'meta': best, 'fold_sign': fold, 'drop': drop, 'fix_slope': fixsl, 'nparams': NP},
                       a.out)
    if it % 10000 == 0 and it > 0:
        torch.save({'params': {kk: v.detach().cpu() for kk, v in model.named_parameters()},
                    'fold_sign': fold, 'drop': drop, 'fix_slope': fixsl, 'nparams': NP}, a.out + '.all')

torch.save({'params': {kk: v.detach().cpu() for kk, v in model.named_parameters()},
            'fold_sign': fold, 'drop': drop, 'fix_slope': fixsl, 'nparams': NP}, a.out + '.all')
print(f'[{a.tag}] DONE best={json.dumps(best)}', flush=True)
