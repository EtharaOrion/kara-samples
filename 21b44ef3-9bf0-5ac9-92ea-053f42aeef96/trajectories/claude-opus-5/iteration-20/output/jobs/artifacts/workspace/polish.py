"""Final training stage: take exact members and make them robustly exact.

Adds two terms to the classification loss:

* a relative read-out margin hinge -- push the winning prototype further from
  the runner-up, measured in units of the code's own step so the objective
  cannot be gamed by inflating the code scale;
* a gate-saturation hinge computed in closed form over all 100 digit pairs
  (sampling it lets a rare pair sit mid-transition while held-out accuracy
  still reads 1.0, which silently breaks the whole-domain certificate).
"""

import argparse
import os

import torch

import data
import ens
import train


def code_of(params):
    dev = params["code_free"].device
    e = params["code_free"].shape[0]
    head = torch.tensor([ens.CONST["code0"], ens.CONST["code1"]], device=dev).expand(e, 2)
    return torch.cat([head, params["code_free"]], dim=1)


def sat_penalty(params, target):
    """Closed-form over every digit pair, not over sampled tokens."""
    code = code_of(params)
    x = code[:, :, None] + code[:, None, :]
    t = ens.CONST["bank_w"] * (x[..., None] - params["knee"][:, None, None, :])
    slack = torch.maximum(-t, t - 1.0)
    return torch.relu(target - slack).mean(dim=(1, 2, 3))


def losses(params, tokens, targets, ls, margin_target, sat_target, w_margin, w_sat):
    stream, code = ens.stream_ens(params, tokens)
    dist = (stream[:, :, 1:, None] - code[:, None, None, :]).abs()
    step = code.diff(dim=-1).abs().mean(dim=-1).clamp(min=1e-4)
    logits = -ls * dist / step[:, None, None, None]
    logp = torch.log_softmax(logits, dim=-1)
    tgt = targets[None, :, :, None].expand(logp.shape[0], -1, -1, 1)
    ce = -logp.gather(-1, tgt).squeeze(-1).mean(dim=(1, 2))

    good = dist.gather(-1, tgt)
    other = dist.scatter(-1, tgt, float("inf")).amin(-1, keepdim=True)
    rel = ((other - good).squeeze(-1) / step[:, None, None])
    hinge = torch.relu(margin_target - rel).mean(dim=(1, 2))
    return ce + w_margin * hinge + w_sat * sat_penalty(params, sat_target), ce


def run(a):
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    blob = torch.load(a.ckpt, map_location=dev)
    src = {k: v.to(dev) for k, v in blob["params"].items()}
    picks = torch.tensor([int(i) for i in a.members.split(",")], device=dev)
    rep = a.replicas
    params = {k: v[picks].repeat_interleave(rep, dim=0).clone() for k, v in src.items()}
    e = params["code_free"].shape[0]
    keep = torch.arange(e, device=dev) % rep == 0
    for k, v in params.items():
        noise = torch.randn(v.shape, device=dev, generator=gen) * a.jitter
        v.add_(torch.where(keep.reshape(-1, *([1] * (v.dim() - 1))), 0.0, noise))
    for v in params.values():
        v.requires_grad_(True)

    opt = torch.optim.Adam(list(params.values()), lr=a.lr, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps,
                                                pct_start=0.1)
    places = [int(p) for p in a.places.split(",")]
    eval_sets = train.make_eval([int(p) for p in a.eval_places.split(",")],
                                a.eval_batch, dev, a.seed + 77)
    best_score = torch.full((e,), -1e9, device=dev)
    best = {k: v.detach().clone() for k, v in params.items()}

    for step in range(a.steps):
        n = places[step % len(places)]
        da, db = data.sample(a.batch, n, dev, gen)
        tokens, targets = data.tokens_and_targets(da, db)
        loss, _ = losses(params, tokens, targets, a.ls, a.margin, a.sat,
                         a.w_margin, a.w_sat)
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        train.per_member_clip(params, 1.0)
        opt.step()
        sched.step()
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            det = {k: v.detach() for k, v in params.items()}
            exact, margin = train.evaluate(det, eval_sets)
            sat = train.saturation(det)
            code = code_of(det)
            rel = margin / code.diff(dim=-1).abs().mean(dim=-1).clamp(min=1e-4)
            score = ((exact >= 1.0).float() * 100 + (sat > 0).float() * 10
                     + rel.clamp(-1, 0.6) + 0.2 * sat.clamp(-1, 2.0))
            better = score > best_score
            best_score = torch.where(better, score, best_score)
            for k in best:
                m = better.reshape(-1, *([1] * (best[k].dim() - 1)))
                best[k] = torch.where(m, det[k], best[k])
            print("step %6d exact1.0 %5d  exact+sat %5d  best_rel_margin %.4f"
                  % (step + 1, int((exact >= 1.0).sum()),
                     int(((exact >= 1.0) & (sat > 0)).sum()),
                     float(rel[(exact >= 1.0) & (sat > 0)].max())
                     if int(((exact >= 1.0) & (sat > 0)).sum()) else -1.0), flush=True)

    out = os.path.join(train.RUNS, a.out)
    torch.save({"params": {k: v.detach().cpu() for k, v in best.items()},
                "args": vars(a)}, out)
    print("saved", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--members", required=True)
    p.add_argument("--replicas", type=int, default=64)
    p.add_argument("--jitter", type=float, default=0.01)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--ls", type=float, default=8.0)
    p.add_argument("--margin", type=float, default=0.45)
    p.add_argument("--sat", type=float, default=1.5)
    p.add_argument("--w_margin", type=float, default=2.0)
    p.add_argument("--w_sat", type=float, default=2.0)
    p.add_argument("--places", type=str, default="2,3,5,8,12")
    p.add_argument("--eval_places", type=str, default="3,8,12")
    p.add_argument("--eval_batch", type=int, default=4096)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=str, default="polish.pt")
    run(p.parse_args())
