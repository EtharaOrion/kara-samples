"""Train E independent copies of the same tiny architecture at once.

At these sizes seed variance dominates: the same config can land at 89% or
99.99% depending on init.  So instead of running one model at a time we vmap E
members over a shared batch and keep the winners.  Adam is elementwise, so the
members stay completely independent (gradients are clipped per member).

  python train_ens.py --cfg '{"d":2,"u1":4,"u2":4}' --E 256 --steps 30000 \
      --out work/run1.pt [--init_from work/run0.pt --init_noise 0.6]
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

import data
import model_src


# ----------------------------------------------------------------------
def stack_init(cfg, E, device, seed):
    outs = {}
    for e in range(E):
        torch.manual_seed(seed + 7919 * e)
        m = model_src.Adder(cfg)
        for k, v in m.named_parameters():
            outs.setdefault(k, []).append(v.detach())
    return {k: torch.stack(v).to(device) for k, v in outs.items()}


def unit_keep(parent, child_u, prefix, pmap=None, cmap=None):
    """Pick which FFN units to carry over: the ones with the largest output.

    When the child pins each unit to one residual axis, units are matched axis
    by axis, so a child slot that writes axis 1 inherits a parent unit that was
    already writing axis 1.
    """
    o = parent[f'{prefix}_o']                       # (E, u, dout) or (E, u)
    E, pu = o.shape[0], o.shape[1]
    if cmap is None:
        score = o.abs().sum(-1) if o.dim() == 3 else o.abs()
        idx = score.argsort(-1, descending=True)[:, :child_u]
        return idx.sort(-1).values                  # (E, child_u)

    out = o.new_zeros(E, child_u, dtype=torch.long)
    for a in sorted(set(cmap)):
        slots = [i for i, b in enumerate(cmap) if b == a]
        if o.dim() == 3:
            score = o[..., a].abs() if a < o.shape[-1] else o.abs().sum(-1)
        else:
            score = o.abs()
        if pmap is not None:                        # only same-axis parent units
            bad = [j for j in range(pu) if pmap[j] != a]
            if len(bad) < pu:
                score = score.clone()
                score[:, bad] = -1.0
        idx = score.argsort(-1, descending=True)[:, :len(slots)]
        idx = idx.sort(-1).values
        for c, s in enumerate(slots):
            out[:, s] = idx[:, c]
    return out


def full_code(params, cfg):
    """Parent code rows in digit order, (E, 10, cd)."""
    c = params['code_p']
    nfix = cfg['code_fix']
    if not nfix:
        return c
    cd = c.shape[-1]
    pin = torch.zeros(c.shape[0], max(nfix, 1), cd, device=c.device, dtype=c.dtype)
    for i, row in enumerate(model_src.PIN_ROWS[:nfix]):
        if row == 1:
            pin[:, i, 0] = 1.0
    allc = torch.cat([pin[:, :nfix], c], 1)
    order = torch.zeros(10, dtype=torch.long, device=c.device)
    free = [i for i in range(10) if i not in model_src.PIN_ROWS[:nfix]]
    for j, row in enumerate(model_src.PIN_ROWS[:nfix]):
        order[row] = j
    for j, row in enumerate(free):
        order[row] = nfix + j
    return allc.index_select(1, order)


def remap(parent_params, parent_cfg, child_cfg, child_shapes, device):
    """Express parent members on the child architecture (best effort)."""
    E = next(iter(parent_params.values())).shape[0]
    out = {}
    keep = {}
    for pre, cu in (('f1', child_cfg['u1']), ('f2', child_cfg['u2'])):
        pu = parent_cfg['u1'] if pre == 'f1' else parent_cfg['u2']
        cmap, pmap = child_cfg[f'{pre}_out_map'], parent_cfg[f'{pre}_out_map']
        keep[pre] = (unit_keep(parent_params, cu, pre, pmap, cmap)
                     if (cu < pu or (cmap is not None and cmap != pmap)) else None)
    # the code table is re-indexed through digit order, so pinning a different
    # set of rows does not shift every digit by one
    code10 = full_code(parent_params, parent_cfg)
    cfree = [i for i in range(10) if i not in model_src.PIN_ROWS[:child_cfg['code_fix']]]
    parent_params = dict(parent_params)
    parent_params['code_p'] = code10[:, cfree]

    for name, shape in child_shapes.items():
        p = parent_params.get(name)
        if p is None:
            out[name] = None
            continue
        pre = name[:2]
        if pre in ('f1', 'f2') and keep[pre] is not None:
            idx = keep[pre]
            if p.dim() == 2:                          # (E, u): biases or gains
                p = torch.gather(p, 1, idx)
            elif name.endswith('_o'):                 # (E, u, dout)
                p = torch.gather(p, 1, idx[..., None].expand(-1, -1, p.shape[-1]))
            elif name.endswith('_w'):                 # (E, d, u)
                p = torch.gather(p, 2, idx[:, None, :].expand(-1, p.shape[1], -1))
        if name.endswith('_o') and p.dim() == 3 and len(shape) == 1:
            # child gives each unit one gain on a fixed axis: take that column
            amap = child_cfg[f'{pre}_out_map']
            col = torch.tensor([min(a, p.shape[-1] - 1) for a in amap],
                               device=p.device)
            p = p.gather(2, col.view(1, -1, 1).expand(p.shape[0], -1, 1)).squeeze(-1)
        elif name.endswith('_o') and p.dim() == 2 and len(shape) == 2:
            amap = parent_cfg[f'{pre}_out_map']
            q = p.new_zeros((E, p.shape[1], shape[-1]))
            for j, a in enumerate(amap):
                if a < shape[-1]:
                    q[:, j, a] = p[:, j]
            p = q
        # trim / pad remaining dims
        cur = list(p.shape[1:])
        tgt = list(shape)
        if cur != tgt:
            sl = [slice(0, min(c, t)) for c, t in zip(cur, tgt)]
            q = p.new_zeros((E, *tgt))
            q[(slice(None), *sl)] = p[(slice(None), *sl)]
            p = q
        out[name] = p
    return out


def load_init(path, child_cfg, E, device, noise, top):
    ck = torch.load(path, map_location=device)
    pcfg = model_src.default_cfg(**ck['cfg'])       # fill keys added since
    pparams, acc = ck['params'], ck['acc']
    order = acc.argsort(descending=True)[:top]
    base = model_src.Adder(child_cfg)
    shapes = {k: tuple(v.shape) for k, v in base.named_parameters()}
    sel = {k: v[order] for k, v in pparams.items()}
    mapped = remap(sel, pcfg, child_cfg, shapes, device)
    rand = stack_init(child_cfg, E, device, seed=1234)
    out = {}
    reps = math.ceil(E / len(order))
    for name, shape in shapes.items():
        m = mapped.get(name)
        if m is None:
            out[name] = rand[name]
            continue
        tiled = m.repeat(reps, *([1] * (m.dim() - 1)))[:E].clone()
        scale = tiled.abs().mean().clamp(min=0.05)
        n = torch.randn_like(tiled) * (noise * scale)
        n[:len(order)] = 0.0                     # keep the parents intact
        out[name] = tiled + n
    print(f'init from {path}: parents {[round(float(acc[i]), 5) for i in order[:8]]}')
    return out


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='{}')
    ap.add_argument('--E', type=int, default=256)
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=0.012)
    ap.add_argument('--temp', type=float, default=1.0)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--init_from', default=None)
    ap.add_argument('--init_noise', type=float, default=0.6)
    ap.add_argument('--init_top', type=int, default=8)
    ap.add_argument('--eval_every', type=int, default=2000)
    ap.add_argument('--eval_n', type=int, default=8192)
    ap.add_argument('--mix', default='0.35,0.40,0.25')
    ap.add_argument('--aux', type=float, default=0.0,
                    help='initial weight of the annealed attention-supervision term')
    ap.add_argument('--aux_frac', type=float, default=0.4,
                    help='fraction of training over which --aux decays to zero')
    ap.add_argument('--anneal_norm', type=float, default=0.0,
                    help='fraction of training over which prenorm blends to 0')
    ap.add_argument('--anneal_sub', default='',
                    help="sublayers whose norm blends to 0 ('1', '2', 'a')")
    ap.add_argument('--anneal_frac', type=float, default=0.4,
                    help='fraction of training over which --anneal_sub decays')
    args = ap.parse_args()

    device = 'cuda'
    cfg = model_src.default_cfg(**json.loads(args.cfg))
    mix = tuple(float(x) for x in args.mix.split(','))
    base = model_src.Adder(cfg).to(device)
    buffers = {k: v.to(device) for k, v in base.named_buffers()}
    n_par = base.n_params()
    print(f'cfg={cfg}\nparams/member={n_par}  E={args.E}')

    if args.init_from:
        params = load_init(args.init_from, cfg, args.E, device,
                           args.init_noise, args.init_top)
    else:
        params = stack_init(cfg, args.E, device, args.seed)
    params = {k: v.contiguous() for k, v in params.items()}

    def carry_target(da, db):
        """For each position, the nearest earlier place that is not transparent.

        That is the place a correct carry lookup has to land on; position 0 is
        the (0,0) slot, which is never transparent, so a target always exists.
        Used only as an annealed training signal.
        """
        s = da + db
        idx = torch.arange(model_src.N_POS, device=da.device).expand_as(s)
        valid = torch.where(s != 9, idx, torch.zeros_like(idx))
        pref = torch.cummax(valid, dim=1).values          # (B, T)
        tgt = torch.zeros_like(pref)
        tgt[:, 1:] = pref[:, :-1]
        return tgt

    def loss_fn(p, da, db, y, tgt, w_aux):
        logits, att = functional_call(base, (p, buffers), (da, db),
                                      {'want_attn': True})
        loss = F.cross_entropy(logits[:, 1:].reshape(-1, 10) * args.temp,
                               y[:, 1:].reshape(-1))
        if args.aux > 0:
            lp = torch.log(att[:, 1:].clamp_min(1e-9))
            aux = -lp.gather(-1, tgt[:, 1:, None]).mean()
            loss = loss + w_aux * aux
        return loss

    def acc_fn(p, da, db, y):
        logits = functional_call(base, (p, buffers), (da, db))
        ok = (logits[:, 1:].argmax(-1) == y[:, 1:]).all(-1)
        return ok.float().mean()

    gfn = vmap(grad(loss_fn), in_dims=(0, None, None, None, None, None))
    afn = vmap(acc_fn, in_dims=(0, None, None, None))

    m = {k: torch.zeros_like(v) for k, v in params.items()}
    v2 = {k: torch.zeros_like(v) for k, v in params.items()}
    b1, b2, eps = 0.9, 0.99, 1e-8
    g = torch.Generator(device=device).manual_seed(args.seed + 99)
    ge = torch.Generator(device=device).manual_seed(12345)

    warm = max(1, int(0.05 * args.steps))
    t0 = time.time()
    best = None
    for step in range(1, args.steps + 1):
        if step <= warm:
            lr = args.lr * step / warm
        else:
            f = (step - warm) / max(1, args.steps - warm)
            lr = args.lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * f)))
        if args.anneal_norm > 0:
            f = max(0.0, 1.0 - step / (args.anneal_norm * args.steps))
            base.cfg['prenorm'] = f * float(cfg['prenorm'])
        if args.anneal_sub:
            f = max(0.0, 1.0 - step / (args.anneal_frac * args.steps))
            base.cfg['norm_t'] = {c: f for c in args.anneal_sub}
        da, db, y = data.sample(args.batch, device, g, mix, holdout=False)
        w_aux = args.aux * max(0.0, 1.0 - step / (args.aux_frac * args.steps))
        tgt = carry_target(da, db) if args.aux > 0 else y
        grads = gfn(params, da, db, y, tgt, w_aux)
        # per-member gradient clipping
        sq = sum(gr.flatten(1).pow(2).sum(1) for gr in grads.values())
        scale = (args.clip / (sq.sqrt() + 1e-12)).clamp(max=1.0)
        for k in params:
            gr = grads[k] * scale.view(-1, *([1] * (grads[k].dim() - 1)))
            m[k].mul_(b1).add_(gr, alpha=1 - b1)
            v2[k].mul_(b2).addcmul_(gr, gr, value=1 - b2)
            mh = m[k] / (1 - b1 ** step)
            vh = v2[k] / (1 - b2 ** step)
            params[k] = params[k] - lr * mh / (vh.sqrt() + eps)

        if step % args.eval_every == 0 or step == args.steps:
            with torch.no_grad():
                ea, eb, ey = data.sample_uniform_pairs(args.eval_n, device, ge)
                acc = afn(params, ea, eb, ey)
                ca, cb, cy = data.sample(args.eval_n, device, ge,
                                         (0.0, 0.4, 0.6), holdout=True)
                cacc = afn(params, ca, cb, cy)
            comb = torch.minimum(acc, cacc)
            top = comb.argsort(descending=True)[:5]
            print(f'{step:7d} lr {lr:.4f} {time.time()-t0:6.0f}s | '
                  f'best uni {acc.max():.5f} chain {cacc.max():.5f} | '
                  f'top5 comb {[round(float(comb[i]),5) for i in top]} | '
                  f'>=0.999: {(comb>=0.999).sum().item()}', flush=True)
            best = comb

    cfg['prenorm'] = base.cfg['prenorm']      # record the annealed end state
    if args.anneal_sub:                       # ended at blend 0 == a hard skip
        cfg['norm_skip'] = ''.join(sorted(set(cfg['norm_skip']) | set(args.anneal_sub)))
        cfg['norm_t'] = None
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with torch.no_grad():
        accs = []
        for _ in range(8):
            ea, eb, ey = data.sample_uniform_pairs(16384, device, ge)
            a1 = afn(params, ea, eb, ey)
            ca, cb, cy = data.sample(16384, device, ge, (0.0, 0.4, 0.6), holdout=True)
            a2 = afn(params, ca, cb, cy)
            accs.append(torch.minimum(a1, a2))
        acc = torch.stack(accs).mean(0)
    torch.save({'cfg': cfg, 'params': {k: v.cpu() for k, v in params.items()},
                'acc': acc.cpu(), 'n_par': n_par}, args.out)
    top = acc.argsort(descending=True)[:10]
    print('final top10:', [round(float(acc[i]), 5) for i in top])
    print('saved', args.out, 'params/member', n_par)


if __name__ == '__main__':
    main()
