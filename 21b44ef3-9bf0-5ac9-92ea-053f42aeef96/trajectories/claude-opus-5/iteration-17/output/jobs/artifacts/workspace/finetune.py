"""Stage 3b -- re-train the 12 free values in the shipped form.

Cross-entropy is the wrong objective here: with the read-out temperature pinned
to 1 the logits are flat, so CE pushes the code scale up, which the residual
scale gauge (carry_w = 1) forbids.  Instead train directly on the geometry that
decides the answer:

  * read-out margin -- how far the residual has to move before the nearest
    prototype changes;
  * bank saturation -- how far every clamp pre-activation sits outside [0,1],
    which is what makes the key and value exactly binary.
"""
import argparse
import time

import torch

import arch
import data
import lab

DEV = "cuda"


def bank_slack(params, cfg):
    """How far every reachable clamp pre-activation sits outside [0, 1].

    A pure function of the weights -- no data, no labels: the 100 digit pairs
    and the (0,0) pad are simply every input the bank can ever see.  Driving
    this positive is what makes the key and value exactly binary, which is the
    premise the whole-domain certificate rests on.
    """
    w = arch.effective(params, cfg, DEV)
    code = w["code"][:, :, 0]                               # (E,10)
    x = code[:, :, None] + code[:, None, :]                 # (E,10,10)
    x = torch.cat([x.reshape(x.shape[0], -1),
                   torch.zeros_like(x[:, 0, :1])], dim=1)   # (E,101)
    e = w["Wb"][:, None, :, 0] * x[..., None] + w["bb"][:, None, :]
    return torch.maximum(-e, e - 1.0)                       # (E,101,U)


def geo_loss(params, cfg, tok, tgt, m_target=0.45, sat_target=1.5, sat_w=0.5):
    _, parts = arch.forward(params, cfg, tok, return_parts=True)
    y = parts["y"][..., 0]                                  # (E,B,P)
    code = parts["w"]["code"][:, :, 0]                      # (E,10)
    t = tgt.clamp(min=0)
    valid = (tgt >= 0)[None]

    d = (y[..., None] - code[:, None, None, :]).abs()       # (E,B,P,10)
    oh = torch.nn.functional.one_hot(t, 10)[None].bool()
    d_c = d.masked_fill(~oh, float("inf")).amin(-1)
    d_o = d.masked_fill(oh, float("inf")).amin(-1)
    margin = d_o - d_c
    l_m = (torch.relu(m_target - margin) * valid).sum((1, 2)) / valid.sum()

    sat = bank_slack(params, cfg)                           # (E,101,U)
    l_s = torch.relu(sat_target - sat).mean((1, 2))
    return l_m + sat_w * l_s, margin, sat


@torch.no_grad()
def score(params, cfg, corners):
    """exact match, plus worst margin / worst saturation as tiebreaks."""
    acc = lab.eval_exact(params, cfg, corners)
    wm = torch.full_like(acc, float("inf"))
    E = acc.shape[0]
    ws = bank_slack(params, cfg).amin((1, 2))      # exhaustive, not sampled
    for tok, tgt in corners:
        ch = max(16, min(tok.shape[0], 2 ** 25 // max(1, E * tok.shape[1] * 10)))
        for i in range(0, tok.shape[0], ch):
            _, mg, _ = geo_loss(params, cfg, tok[i:i + ch], tgt[i:i + ch])
            v = (tgt[i:i + ch] >= 0)[None]
            wm = torch.minimum(wm, mg.masked_fill(~v, float("inf")).amin((1, 2)))
    return acc, wm, ws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="ship.pt")
    ap.add_argument("--out", default="ship_ft.pt")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--rep", type=int, default=4)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ck = torch.load(a.inp, map_location=DEV, weights_only=False)
    cfg, p = ck["cfg"], {k: v.to(DEV) for k, v in ck["params"].items()}
    E0 = p["code"].shape[0]
    g = torch.Generator(device=DEV).manual_seed(a.seed)
    p = {k: v.repeat(a.rep, *([1] * (v.dim() - 1))).contiguous()
         for k, v in p.items()}
    keys = arch.trainable_keys(cfg)
    print("free scalars / member =", arch.n_free_values(cfg), "keys:", keys)
    for k in keys:                       # jitter the copies so they diverge
        p[k] = p[k] + torch.randn(p[k].shape, device=DEV, generator=g) * a.noise
        p[k] = p[k].contiguous().requires_grad_(True)
    E = p["code"].shape[0]

    corners = lab.make_eval([8, 5, 11, 3], DEV, per_n=1024)
    opt = torch.optim.AdamW([p[k] for k in keys], lr=a.lr, weight_decay=0.0,
                            betas=(0.9, 0.99))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr,
                                              total_steps=a.steps, pct_start=0.05)
    gen = torch.Generator(device=DEV).manual_seed(a.seed + 5)
    best = torch.full((E,), -1e9, device=DEV)
    best_w = {k: p[k].detach().clone() for k in p}
    places = [2, 3, 5, 8, 11]
    t0 = time.time()
    for step in range(a.steps):
        n = places[step % len(places)]
        da, db = data.sample(a.batch, n, DEV, gen=gen)
        tok, tgt = data.tokens(da, db), data.targets(da, db)
        loss, _, _ = geo_loss(p, cfg, tok, tgt)
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        lab.clip_per_member(p, keys, 1.0)
        opt.step()
        sch.step()
        if (step + 1) % 250 == 0 or step == a.steps - 1:
            acc, wm, ws = score(p, cfg, corners)
            # exact match dominates; margin and saturation break ties so the
            # search keeps improving after everything is already exact
            sc = acc + 0.02 * wm.clamp(-1, 0.5) + 0.02 * ws.clamp(-1, 1.0)
            imp = sc > best
            if imp.any():
                for k in p:
                    best_w[k] = torch.where(
                        imp.view(-1, *([1] * (p[k].dim() - 1))),
                        p[k].detach(), best_w[k])
                best = torch.maximum(best, sc)
            j = int(sc.argmax())
            print(f"step {step+1:5d} loss {loss.mean().item():.4f} "
                  f"#exact {(acc>=1.0).sum().item():4d}/{E} "
                  f"best-score {best.max().item():.4f} "
                  f"(acc {acc[j]:.5f} margin {wm[j]:.3f} sat {ws[j]:.3f}) "
                  f"{time.time()-t0:.0f}s", flush=True)

    acc, wm, ws = score(best_w, cfg, corners)
    ok = ((acc >= 1.0) & (ws > 0)).nonzero().flatten()
    print(f"final: {ok.numel()}/{E} members exact on all corners AND fully "
          f"saturated (from {E0} originals x{a.rep})", flush=True)
    order = ok[torch.argsort(wm[ok], descending=True)]
    print("top margins:", [round(float(wm[i]), 4) for i in order[:10]])
    print("their slacks:", [round(float(ws[i]), 4) for i in order[:10]])
    torch.save(dict(params={k: v[order].cpu() for k, v in best_w.items()},
                    cfg=cfg, acc=acc[order].cpu(), margin=wm[order].cpu(),
                    sat=ws[order].cpu()), a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
