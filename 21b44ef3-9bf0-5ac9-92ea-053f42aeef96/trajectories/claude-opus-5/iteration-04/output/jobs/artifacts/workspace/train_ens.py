"""Train E independent copies of the model at once ("seed lottery").

At these parameter counts the outcome is dominated by initialisation luck, so
we vmap E members over a shared batch and keep the best one.  Adam is
elementwise and gradients are clipped per member, so the members are
genuinely independent runs -- just packed into single kernels.
"""
import argparse, json, math, os, time
import torch
import torch.nn.functional as F
from torch.func import functional_call, vmap

import data
import warm
from model_src import Adder, BASE, count_params

DEV = "cuda"


def stack_init(cfg, E, seed, init=None, sigma=0.0):
    """E independently initialised copies of the model, stacked on dim 0.

    With `init` (a parent checkpoint, already remapped onto this cfg) member 0
    is the parent exactly and the rest are the parent plus noise, so a rung of
    the cascade can never do worse than the rung above it.  The noise level is
    ramped across members from a twentieth of `sigma` up to twice it: how far a
    child has to be kicked before it escapes the parent's basin is not
    something we can guess, so the ensemble spends its width on finding out.

    The learning rate is ramped with it, and for the same reason.  A member that
    was not kicked starts at a solution and only wants polishing -- run it at
    the exploration rate and the first few steps throw the solution away -- while
    a member kicked past the basin needs the full rate to find its way back.
    Returns the per-member multiplier alongside the stacked parameters.
    """
    base = Adder(cfg)
    out = {}
    if E > 1:
        lo, hi = math.log(sigma / 20 + 1e-12), math.log(sigma * 2 + 1e-12)
        ramp = torch.exp(torch.linspace(lo, hi, E - 1)) if sigma > 0 \
            else torch.zeros(E - 1)
        ramp = torch.cat([torch.zeros(1), ramp])
    else:
        ramp = torch.zeros(1)
    for name, _ in base.named_parameters():
        gs = torch.Generator().manual_seed(seed * 100003 + hash(name) % 99991)
        vs = []
        for e in range(E):
            m = Adder(cfg)
            torch.manual_seed(seed * 7919 + e * 104729 + hash(name) % 7717)
            vs.append(dict(m.named_parameters())[name].data.clone())
        t = torch.stack(vs)
        if init is not None and name in init and init[name].numel() == t[0].numel():
            w0 = init[name].to(t.dtype).reshape(t.shape[1:])
            sc = ramp.reshape((E,) + (1,) * (t.dim() - 1)).to(t.dtype)
            t = w0.unsqueeze(0) + torch.randn(t.shape, generator=gs) * sc
        out[name] = t.to(DEV).requires_grad_(True)
    lrs = torch.ones(E)
    if init is not None and sigma > 0:
        lrs = (ramp / (2 * sigma)).clamp(min=0.01, max=1.0)
    return out, base, lrs.to(DEV)


def prune_plan(cfg, ffn, E, dev):
    """Which unit each ensemble member is going to do without, and the schedule
    that takes it away.

    Dropping a ReLU from a trained bank in one step lands the child far from any
    solution: the survivors were fitted alongside the one that left.  So take it
    away slowly instead, scaling its output row down to nothing over the first
    two thirds of training while the accuracy term keeps pulling the rest of the
    bank into shape around the loss.  Which unit should go is not obvious
    either, so the ensemble does not guess: member e gives up unit e mod f, and
    every unit is tried in parallel."""
    f = cfg[ffn]
    idx = torch.arange(E, device=dev) % f
    return idx, torch.nn.functional.one_hot(idx, f).to(dev)


def prune_scale(frac, hold=0.05, done=0.7):
    """1 until `hold` of the way through, 0 from `done` on, cosine between."""
    if frac <= hold:
        return 1.0
    if frac >= done:
        return 0.0
    return 0.5 * (1 + math.cos(math.pi * (frac - hold) / (done - hold)))


def _dead(o):
    """How close the least-used unit of an output matrix is to writing nothing.

    Each column is normalised by its own mean square first, so a member cannot
    satisfy the measure by inflating the columns whose scale the rest of the
    model can absorb -- only by actually taking a unit's work away from it."""
    o2 = o.reshape(o.shape[0], o.shape[1], -1).pow(2)
    sq = (o2 / (o2.mean(1, keepdim=True) + 1e-12)).sum(-1)
    return sq.min(-1).values / (sq.mean(-1) + 1e-12)


def evaluate(base, params, dig_sets, chunk=2048):
    """Exact-match accuracy per member for each held-out set."""
    accs = []
    f = lambda p, xx: functional_call(base, p, (xx,))
    for dig in dig_sets:
        tot = 0
        ok = None
        for i in range(0, dig.shape[0], chunk):
            d = dig[i:i + chunk]
            x, y = data.tokens(d), data.targets(d)
            with torch.no_grad():
                lg = vmap(f, in_dims=(0, None))(params, x)
                pred = lg[:, :, 1:, :].argmax(-1)
                c = (pred == y.unsqueeze(0)).all(-1).sum(1)
            ok = c if ok is None else ok + c
            tot += d.shape[0]
        accs.append((ok.float() / tot))
    return accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="{}")
    ap.add_argument("--E", type=int, default=256)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.012)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--init_from", type=str, default="")
    ap.add_argument("--sigma", type=float, default=0.6)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--eval_n", type=int, default=8192)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--pen_ffn", type=str, default="")
    ap.add_argument("--pen_lam", type=float, default=0.0)
    ap.add_argument("--pen_tol", type=float, default=0.02)
    ap.add_argument("--prune_ffn", type=str, default="")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--compile", type=int, default=0)
    a = ap.parse_args()

    cfg = dict(BASE)
    cfg.update(json.loads(a.cfg))
    init = None
    if a.init_from:
        ck = torch.load(a.init_from, map_location="cpu", weights_only=False)
        init = warm.remap(ck["cfg"], ck["params"], cfg)
        agree, miss = warm.check(ck["cfg"], ck["params"], cfg, init)
        print(f"[{a.tag}] warm start from {a.init_from}: parent acc {ck['acc']:.4f}, "
              f"remapped child agrees with parent on {agree:.4f} of sequences"
              + (f", fresh: {miss}" if miss else ""), flush=True)
    params, base, lrs = stack_init(cfg, a.E, a.seed, init, a.sigma)
    base = base.to(DEV)
    n_par = count_params(base)
    print(f"[{a.tag}] params={n_par} E={a.E} steps={a.steps} cfg={json.dumps(cfg)}",
          flush=True)

    # The prototype readout measures a squared distance, so the margin between
    # two digits goes as the square of the gap between their codes -- and that
    # gap has no fixed size, because naming the answer axis' unit (`code_fix`)
    # rescales the whole table.  A temperature tuned for a gap of one would then
    # mean something different on every rung, so scale it by the gap actually
    # present at initialisation and let --temp keep its meaning.
    if a.temp != 1.0 and not cfg.get("logit_scale", True) \
            and cfg.get("out_mode") == "proto" and cfg["code_dim"] == 1:
        with torch.no_grad():
            probe = Adder(cfg)                       # member 0's table, not base's
            probe.code_p.data = params["code_p"][0].detach().cpu()
            gap = probe._code().sort()[0].diff().abs().median()
        a.temp = a.temp / max(float(gap), 1e-6) ** 2
        print(f"[{a.tag}] code gap {float(gap):.4f} -> temp {a.temp:.4g}", flush=True)

    ev_u = data.eval_set(a.eval_n, DEV, 1234, data.UNIFORM)
    ev_h = data.eval_set(a.eval_n, DEV, 5678, data.HARD)

    pkey = f"{a.prune_ffn}_o" if a.prune_ffn else None
    if pkey and pkey in params:
        _, pmask = prune_plan(cfg, a.prune_ffn, a.E, DEV)
        pmask = pmask.reshape(pmask.shape + (1,) * (params[pkey].dim() - 2))
        lrs = torch.ones_like(lrs)   # every member is losing a unit, so none
    else:                            # of them is merely polishing the parent
        pkey = None

    plist = list(params.values())
    keys = list(params.keys())
    opt = torch.optim.AdamW(plist, lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    g = torch.Generator(device=DEV).manual_seed(a.seed + 999)
    f = lambda p, xx: functional_call(base, p, (xx,))

    def loss_fn(p, x, y):
        # `temp` sharpens the loss, not the model: it is a constant, it is not
        # shipped, and scaling every logit at a position by the same positive
        # number cannot move an argmax, which is all the prediction reads.  It
        # matters once the model has no learned logit scale of its own -- the
        # prototype gap is then fixed near 1 and plain cross-entropy on a
        # margin of 1 has almost no gradient to give.
        lg = vmap(f, in_dims=(0, None))(p, x)          # (E,B,T,10)
        lg = lg[:, :, 1:, :] * a.temp
        ce = F.cross_entropy(lg.reshape(-1, 10), y.repeat(lg.shape[0], 1).reshape(-1),
                             reduction="none").view(lg.shape[0], -1).mean(1)
        if a.pen_lam and f"{a.pen_ffn}_o" in p:
            # Make one unit of the bank droppable rather than hoping one will
            # be.  A trained bank spreads its job over all its units, so the
            # cheapest unit to remove is still expensive; this asks for the
            # cheapest to become free, by pushing the smallest row of the output
            # matrix down relative to the rest.
            ce = ce + a.pen_lam * _dead(p[f"{a.pen_ffn}_o"])
        return ce

    def _step(pl, x, y):
        """loss + per-member clipped grads, in one compiled graph."""
        per = loss_fn(dict(zip(keys, pl)), x, y)
        grads = torch.autograd.grad(per.sum(), pl)
        sq = sum(gr.pow(2).reshape(gr.shape[0], -1).sum(1) for gr in grads)
        sc = (a.clip / (sq.sqrt() + 1e-12)).clamp(max=1.0)
        return per.detach(), [gr * sc.view(-1, *([1] * (gr.dim() - 1))) for gr in grads]

    step_fn = torch.compile(_step, dynamic=False) if a.compile else _step
    per_lr = bool((lrs != 1.0).any())

    best = {"acc": -1.0, "rank": -9.0}
    cur_s = 1.0
    t0 = time.time()

    def snap(step):
        """Record the best member so far.  Called at step 0 too, so a rung that
        starts from an exact gauge cut keeps that solution even if every
        member's training then wanders away from it.

        Under a sparsity penalty accuracy alone is the wrong thing to keep: the
        member that scores best may be the one that ignored the penalty.  A
        member that gave a unit up outranks one that did not, whatever it
        scores, but the ranking stays total so there is always something to
        save."""
        au, ah = evaluate(base, params, [ev_u, ev_h])
        comb = torch.minimum(au, ah)
        # While a unit is still being retired the model is not yet the model we
        # are training towards, so those snapshots rank below every later one.
        rank = comb - 5.0 * (pkey is not None and cur_s > 0)
        if a.pen_lam and f"{a.pen_ffn}_o" in params:
            rank = comb - 2.0 * (_dead(params[f"{a.pen_ffn}_o"]) >= a.pen_tol)
        i = int(rank.argmax())
        if float(rank[i]) > best["rank"]:
            best.update(rank=float(rank[i]), acc=float(comb[i]),
                        u=float(au[i]), h=float(ah[i]),
                        member=i, step=step,
                        params={k: v[i].detach().cpu().clone()
                                for k, v in params.items()})
        return comb, i, int((rank > 0.99).sum()), au, ah

    snap(0)
    for step in range(a.steps):
        frac = step / max(1, a.steps - 1)
        lr = a.lr * (0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, (frac - 0.15) / 0.85))))
                     * 0.999 + 0.001)
        if frac < 0.15:
            lr = a.lr * (0.05 + 0.95 * frac / 0.15)
        for gp in opt.param_groups:
            gp["lr"] = lr

        dig = data.train_batch(a.batch, DEV, g)
        x, y = data.tokens(dig), data.targets(dig)
        per, grads = step_fn(plist, x, y)
        for p, gr in zip(plist, grads):
            p.grad = gr
        with torch.no_grad():
            pre = [p.detach().clone() for p in plist] if per_lr else None
        opt.step()
        if per_lr:                       # per-member learning rate: Adam's step
            with torch.no_grad():        # is linear in lr, so scale the update
                for p, q in zip(plist, pre):
                    p.add_((p - q).mul_(lrs.view(-1, *([1] * (p.dim() - 1))) - 1))
        if pkey is not None:             # shrink the row being retired, so the
            nxt = prune_scale((step + 1) / max(1, a.steps - 1))   # rest of the
            with torch.no_grad():        # bank keeps adapting around it
                d = 0.0 if cur_s <= 0 else nxt / cur_s
                params[pkey].mul_(1 - pmask + pmask * d)
            cur_s = nxt

        if (step + 1) % a.eval_every == 0 or step == a.steps - 1:
            comb, i, n_good, au, ah = snap(step + 1)
            print(f"[{a.tag}] step {step+1} loss {per.mean():.4f} "
                  f"best_comb {float(comb[i]):.5f} (u {float(au[i]):.5f} h {float(ah[i]):.5f}) "
                  f"n>0.99: {n_good}  ever {best['acc']:.5f}  {time.time()-t0:.0f}s", flush=True)

    torch.save({"cfg": cfg, "params": best["params"], "n_params": n_par,
                "acc": best["acc"], "u": best["u"], "h": best["h"]}, a.out)
    print(f"[{a.tag}] DONE params={n_par} best_comb={best['acc']:.5f} "
          f"u={best['u']:.5f} h={best['h']:.5f} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
