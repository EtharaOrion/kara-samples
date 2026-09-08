"""Stage 1 -- cold-train a wide parent as an E-member seed lottery.

C=2 residual channels, U=2 clamp-bank units, every constant learned.  Roughly
38 free scalars per member.  A handful of the E members find the carry-routing
mechanism; the rest plateau on the "attend to the previous place" shortcut.
"""
import argparse
import time

import torch

import arch
import data
import lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.012)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--places", type=int, nargs="+", default=[8, 5, 11, 3])
    ap.add_argument("--out", type=str, default="parent.pt")
    a = ap.parse_args()

    dev = "cuda"
    torch.manual_seed(a.seed)
    cfg = arch.default_cfg(C=2, U=2)
    params = arch.init_params(a.E, cfg, dev, seed=a.seed)
    keys = arch.trainable_keys(cfg)
    for k in keys:
        params[k].requires_grad_(True)
    print(f"free scalars / member = {arch.n_free_values(cfg)}", flush=True)

    opt = torch.optim.AdamW([params[k] for k in keys], lr=a.lr, weight_decay=0.0,
                            betas=(0.9, 0.99))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr,
                                              total_steps=a.steps, pct_start=0.1)
    gen = torch.Generator(device=dev).manual_seed(a.seed + 999)
    corners = lab.make_eval(a.places, dev, per_n=2048)

    best = torch.zeros(a.E, device=dev)
    best_w = {k: params[k].detach().clone() for k in params}
    t0 = time.time()
    for step in range(a.steps):
        n = a.places[step % len(a.places)]
        da, db = data.sample(a.batch, n, dev, gen=gen, held_out=False)
        tok, tgt = data.tokens(da, db), data.targets(da, db)
        logits = arch.forward(params, cfg, tok)
        loss = lab.ce_loss(logits, tgt)
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        lab.clip_per_member(params, keys, 1.0)
        opt.step()
        sch.step()

        if (step + 1) % 500 == 0 or step == a.steps - 1:
            acc = lab.eval_exact(params, cfg, corners)
            imp = acc > best
            if imp.any():
                for k in params:
                    best_w[k] = torch.where(
                        imp.view(-1, *([1] * (params[k].dim() - 1))),
                        params[k].detach(), best_w[k])
                best = torch.maximum(best, acc)
            print(f"step {step+1:6d}  loss {loss.mean().item():.4f}  "
                  f"best-acc max {best.max().item():.4f}  "
                  f"#>=0.999 {(best >= 0.999).sum().item():4d}  "
                  f"{time.time()-t0:.0f}s", flush=True)

    order = torch.argsort(best, descending=True)
    print("top-10 held-out exact:", [round(float(best[i]), 5) for i in order[:10]])
    torch.save(dict(params={k: v.cpu() for k, v in best_w.items()},
                    cfg=cfg, acc=best.cpu(), order=order.cpu()), a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
