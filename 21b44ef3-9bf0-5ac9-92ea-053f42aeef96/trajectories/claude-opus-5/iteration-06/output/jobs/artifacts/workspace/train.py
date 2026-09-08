"""Seed-lottery trainer: E independent copies of the block, trained in lockstep.

At these sizes the outcome depends heavily on the initial seed, so we train a
whole population at once (vmap over stacked parameters, one shared batch) and
keep the best member.  Nothing here ends up in the graded file.
"""

import argparse, json, math, os, time

import torch
from torch.func import functional_call, stack_module_state, vmap

import data
from model_src import DigitPairAdder, default_cfg

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------- population ---

def make_pop(cfg, E, seed, init_sd=None, sigma=0.0):
    models = []
    for e in range(E):
        torch.manual_seed(seed * 100003 + e)
        m = DigitPairAdder(cfg).to(DEV)
        if init_sd is not None:
            with torch.no_grad():
                for k, v in m.named_parameters():
                    if k in init_sd:
                        src = init_sd[k].to(DEV).reshape(v.shape)
                        if e == 0 or sigma == 0.0:
                            v.copy_(src)
                        else:
                            scale = sigma * (src.abs().mean() + 0.05)
                            v.copy_(src + scale * torch.randn_like(src))
        models.append(m)
    base = DigitPairAdder(cfg).to(DEV).to("meta")
    params, buffers = stack_module_state(models)
    return base, params, buffers


def fwd_fn(base):
    def f(p, b, da, db):
        return functional_call(base, (p, b), (da, db))
    return vmap(f, in_dims=(0, 0, None, None))


# -------------------------------------------------------------- evaluation --

@torch.no_grad()
def evaluate(fwd, params, buffers, sets):
    out = {}
    for name, (da, db, y) in sets.items():
        ok = None
        for i in range(0, da.shape[0], 4096):
            lg = fwd(params, buffers, da[i:i + 4096], db[i:i + 4096])
            pred = lg.argmax(-1)[:, :, 1:]
            hit = (pred == y[None, i:i + 4096]).all(-1)
            ok = hit.sum(1) if ok is None else ok + hit.sum(1)
        out[name] = ok.float() / da.shape[0]
    return out


def eval_sets(n, seed=1234):
    g = torch.Generator(device=DEV).manual_seed(seed)
    sets = {
        "u": data.batch(n, DEV, g, train=False, mix=((1.0, None, None),)),
        "h": data.batch(n, DEV, g, train=False, mix=((0.4, 0.30, 0.40),
                                                     (0.3, 0.10, 0.80),
                                                     (0.3, 0.02, 0.96))),
    }
    return sets


# ---------------------------------------------------------------- training --

def train(cfg, E=256, steps=20000, bs=1024, lr=0.012, seed=0, init=None,
          sigma=0.0, clip=1.0, log_every=1000, tag="run", warmup=0.15):
    init_sd = None
    if init:
        ck = torch.load(init, map_location=DEV, weights_only=False)
        init_sd = ck["sd"]
    base, params, buffers = make_pop(cfg, E, seed, init_sd, sigma)
    fwd = fwd_fn(base)
    plist = list(params.values())
    nparam = sum(v[0].numel() for v in plist)
    opt = torch.optim.AdamW(plist, lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=warmup, div_factor=10.0,
                                                final_div_factor=100.0)
    g = torch.Generator(device=DEV).manual_seed(seed + 777)
    sets = eval_sets(8192)
    best = (-1.0, None, -1)
    t0 = time.time()
    for step in range(steps):
        da, db, y = data.batch(bs, DEV, g, train=True)
        logits = fwd(params, buffers, da, db)              # (E, B, P, 10)
        logp = torch.log_softmax(logits[:, :, 1:], -1)
        nll = -logp.gather(-1, y[None, :, :, None].expand(E, -1, -1, -1)).squeeze(-1)
        per_member = nll.mean((1, 2))
        loss = per_member.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():                              # per-member grad clip
            sq = sum((p.grad.reshape(E, -1) ** 2).sum(1) for p in plist)
            scale = (clip / (sq.sqrt() + 1e-12)).clamp(max=1.0)
            for p in plist:
                p.grad.mul_(scale.view(-1, *([1] * (p.dim() - 1))))
        opt.step()
        sched.step()
        if (step + 1) % log_every == 0 or step == steps - 1:
            ev = evaluate(fwd, params, buffers, sets)
            score = torch.minimum(ev["u"], ev["h"])
            v, i = score.max(0)
            if v.item() > best[0]:
                best = (v.item(), {k: p[i].detach().clone() for k, p in params.items()}, i.item())
            print(f"[{tag}] step {step+1:6d} loss {per_member.min().item():.4f} "
                  f"best_u {ev['u'].max().item():.5f} best_h {ev['h'].max().item():.5f} "
                  f"min {v.item():.5f} (member {i.item()}) "
                  f"n>=0.999 {(score>=0.999).sum().item():3d} "
                  f"{time.time()-t0:.0f}s", flush=True)
    return best, nparam, params, buffers, fwd, sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="{}")
    ap.add_argument("--E", type=int, default=256)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.012)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", type=str, default=None)
    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--out", type=str, default="ck.pt")
    ap.add_argument("--tag", type=str, default="run")
    ap.add_argument("--log_every", type=int, default=1000)
    a = ap.parse_args()

    cfg = default_cfg(**json.loads(a.cfg))
    best, nparam, *_ = train(cfg, a.E, a.steps, a.bs, a.lr, a.seed, a.init, a.sigma,
                             tag=a.tag, log_every=a.log_every)
    print(f"[{a.tag}] BEST {best[0]:.5f} params {nparam} -> {a.out}", flush=True)
    torch.save({"cfg": cfg, "sd": {k: v.cpu() for k, v in best[1].items()},
                "acc": best[0], "nparam": nparam}, a.out)


if __name__ == "__main__":
    main()
