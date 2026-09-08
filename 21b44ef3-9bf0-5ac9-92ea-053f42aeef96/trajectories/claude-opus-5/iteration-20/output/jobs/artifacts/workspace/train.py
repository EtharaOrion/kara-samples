"""Random-restart lottery trainer for the digit-pair adder.

Phase 1 trains E independent members on single-place problems.  That fixes the
digit code, the mod-10 fold and the carry write weight, but leaves the
transparency knee (the gate that decides which places a carry may travel
through) unconstrained -- a one-place problem has nothing to travel through.

Phase 2 replicates the phase-1 winners, redraws that knee, and trains on
multi-place problems, where a carry must be routed past runs of a+b==9 places.
"""

import argparse
import json
import os
import time

import torch

import data
import ens

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def all_pairs_batch(device):
    d = torch.arange(10, device=device)
    da = d.repeat_interleave(10).unsqueeze(1)
    db = d.repeat(10).unsqueeze(1)
    return da, db


def make_eval(places_list, batch, device, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    out = []
    for n in places_list:
        if n == 1:
            da, db = all_pairs_batch(device)
        else:
            da, db = data.sample(batch, n, device, gen, held_out=True)
        out.append(data.tokens_and_targets(da, db))
    return out


def loss_and_stats(params, tokens, targets, scale, metric="l1", norm=True, tie_w=0.0):
    """Training loss.

    The shipped read-out ranks digits by squared distance; ranking by absolute
    distance is the identical decision rule, but it conditions the loss far
    better at initialisation, where a random code puts prototypes tens of units
    apart and the squared form makes "shrink the whole code" the steepest
    descent direction.
    """
    stream, code = ens.stream_ens(params, tokens)
    # During training the read-out prototypes may be a separate table, held to
    # the embedding by an annealed penalty.  The constraint that forces the code
    # to be linear -- code[a]+code[b] must depend on a+b alone -- is untouched
    # by this; what it removes is the self-referential coupling that makes a
    # collapsing code a strong attractor.
    table = params["proto"] if "proto" in params else code
    dist = (stream[:, :, 1:, None] - table[:, None, None, :]).abs()
    if norm:
        # Measure distances in units of the code's own spread.  Without this,
        # shrinking the tied code flattens every logit toward ln(10) and is a
        # strong attractor that a randomly-ordered code falls into.
        spread = table.std(dim=-1).detach().clamp(min=1e-3)
        dist = dist / spread[:, None, None, None]
    logits = -scale * (dist if metric == "l1" else dist ** 2)
    logp = torch.log_softmax(logits, dim=-1)
    tgt = targets[None, :, :, None].expand(logp.shape[0], -1, -1, 1)
    nll = -logp.gather(-1, tgt).squeeze(-1).mean(dim=(1, 2))
    if "proto" in params:
        nll = nll + tie_w * (params["proto"] - code).pow(2).mean(dim=-1)
    return nll


@torch.no_grad()
def evaluate(params, eval_sets, chunk=4096):
    """Per-member exact-match rate (all digits right) and worst read-out margin."""
    e = params["code_free"].shape[0]
    exact = torch.zeros(e, device=params["code_free"].device)
    margin = torch.full((e,), 1e9, device=params["code_free"].device)
    for tokens, targets in eval_sets:
        ok = torch.zeros(e, device=exact.device)
        mg = torch.full((e,), 1e9, device=exact.device)
        for s in range(0, e, chunk):
            sub = {k: v[s:s + chunk] for k, v in params.items()}
            stream, code = ens.stream_ens(sub, tokens)
            dist = (stream[:, :, 1:, None] - code[:, None, None, :]).abs()
            good = dist.gather(-1, targets[None, :, :, None].expand(dist.shape[0], -1, -1, 1))
            other = dist.scatter(-1, targets[None, :, :, None].expand(dist.shape[0], -1, -1, 1),
                                 float("inf")).amin(-1, keepdim=True)
            gap = (other - good).squeeze(-1)
            ok[s:s + chunk] = (gap > 0).all(-1).float().mean(-1)
            mg[s:s + chunk] = gap.amin(dim=(1, 2))
        exact += ok / len(eval_sets)
        margin = torch.minimum(margin, mg)
    return exact, margin


@torch.no_grad()
def code_residual(params):
    """Per-member deviation of the learned code from an arithmetic progression,
    in units of its own best-fit step.  This is the quantity training has to
    discover: code[a] + code[b] == code[a+b] forces the code to be linear.
    """
    dev = params["code_free"].device
    e = params["code_free"].shape[0]
    head = torch.tensor([ens.CONST["code0"], ens.CONST["code1"]], device=dev).expand(e, 2)
    code = torch.cat([head, params["code_free"]], dim=1)
    d = torch.arange(10.0, device=dev)
    step = (code * d).sum(1) / (d * d).sum()
    return ((code - step[:, None] * d).abs().amax(1) / step.abs().clamp(min=1e-6))


@torch.no_grad()
def saturation(params):
    """Worst-case gate saturation slack over all 100 digit pairs (>0 == saturated)."""
    dev = params["code_free"].device
    e = params["code_free"].shape[0]
    head = torch.tensor([ens.CONST["code0"], ens.CONST["code1"]], device=dev).expand(e, 2)
    code = torch.cat([head, params["code_free"]], dim=1)
    x = code[:, :, None] + code[:, None, :]                       # (E,10,10)
    t = ens.CONST["bank_w"] * (x[..., None] - params["knee"][:, None, None, :])
    return torch.maximum(-t, t - 1.0).amin(dim=(1, 2, 3))


def per_member_clip(params, limit):
    sq = torch.zeros(params["code_free"].shape[0], device=params["code_free"].device)
    for v in params.values():
        if v.grad is not None:
            sq = sq + v.grad.reshape(v.shape[0], -1).pow(2).sum(1)
    scale = (limit / (sq.sqrt() + 1e-12)).clamp(max=1.0)
    for v in params.values():
        if v.grad is not None:
            v.grad.mul_(scale.reshape(-1, *([1] * (v.grad.dim() - 1))))


def run(args):
    os.makedirs(RUNS, exist_ok=True)
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(args.seed)

    if args.init_from:
        blob = torch.load(args.init_from, map_location=dev)
        parents = blob["params"]
        n_par = parents["code_free"].shape[0]
        rep = max(1, args.ensemble // n_par)
        params = {k: v.repeat_interleave(rep, dim=0).clone() for k, v in parents.items()}
        e = params["code_free"].shape[0]
        # One-place problems leave the transparency knee unidentified -- there
        # is nothing for a carry to travel through -- and a saturated gate has
        # no gradient, so it cannot simply be trained afterwards.  Re-draw it on
        # a stratified grid across the span of each parent's own code, so the
        # replicas of a parent cover every interval it could sit in.
        head = torch.tensor([ens.CONST["code0"], ens.CONST["code1"]], device=dev).expand(e, 2)
        code = torch.cat([head, params["code_free"]], dim=1)
        lo = 2 * code.amin(1) - 0.5
        hi = 2 * code.amax(1) + 0.5
        rung = torch.arange(e, device=dev) % rep
        u = (rung + torch.rand(e, device=dev, generator=gen)) / rep
        fresh = lo + u * (hi - lo)
        keep = torch.arange(e, device=dev) % rep == 0
        params["knee"][:, 0] = torch.where(keep, params["knee"][:, 0], fresh)
        for k, v in params.items():
            noise = torch.randn(v.shape, device=dev, generator=gen) * args.jitter
            v.add_(torch.where(keep.reshape(-1, *([1] * (v.dim() - 1))), 0.0, noise))
        print(f"warm start: {n_par} parents x {rep} replicas = {e} members")
    else:
        e = args.ensemble
        params = ens.init_params(e, dev, gen, code_sigma=args.code_sigma)
    if args.untie:
        head = torch.tensor([ens.CONST["code0"], ens.CONST["code1"]], device=dev)
        code0 = torch.cat([head.expand(params["code_free"].shape[0], 2),
                           params["code_free"]], dim=1)
        params["proto"] = code0.clone() + torch.randn(
            code0.shape, device=dev, generator=gen) * args.code_sigma
    else:
        params.pop("proto", None)

    for v in params.values():
        v.requires_grad_(True)
    opt = torch.optim.Adam(list(params.values()), lr=args.lr, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=args.pct_start)

    places = [int(p) for p in args.places.split(",")]
    eval_sets = make_eval([int(p) for p in args.eval_places.split(",")],
                          args.eval_batch, dev, args.seed + 991)
    best_score = torch.full((e,), -1e9, device=dev)
    best = {k: v.detach().clone() for k, v in params.items()}

    t0 = time.time()
    for step in range(args.steps):
        n = places[step % len(places)]
        if n == 1 and args.exhaustive1:
            da, db = all_pairs_batch(dev)
        else:
            md = 9
            if args.digit_ramp > 0:
                span = args.digit_ramp * args.steps
                md = min(9, 2 + int(8 * step / span)) if step < span else 9
            da, db = data.sample(args.batch, n, dev, gen, max_digit=md)
        tokens, targets = data.tokens_and_targets(da, db)
        frac = step / max(1, args.steps - 1)
        ls = args.ls * (args.ls_end / args.ls) ** frac
        tie_w = args.tie0 * (args.tie1 / args.tie0) ** min(1.0, frac / 0.6)
        loss = loss_and_stats(params, tokens, targets, ls, args.metric,
                              bool(args.norm), tie_w)
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        per_member_clip(params, args.clip)
        opt.step()
        sched.step()

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            det = {k: v.detach() for k, v in params.items()}
            exact, margin = evaluate(det, eval_sets)
            sat = saturation(det)
            score = (exact + 0.02 * sat.clamp(-1, 1)
                     + 0.01 * margin.clamp(-1, 1))
            better = score > best_score
            best_score = torch.where(better, score, best_score)
            for k in best:
                m = better.reshape(-1, *([1] * (best[k].dim() - 1)))
                best[k] = torch.where(m, det[k], best[k])
            print(f"step {step+1:6d} loss {loss.mean().item():7.4f} "
                  f"best_exact {exact.max().item():.5f} "
                  f"n>=0.999 {int((exact >= 0.999).sum())} "
                  f"n==1.0 {int((exact >= 1.0).sum())} "
                  f"sat>0 {int((sat > 0).sum())} "
                  f"lin {int((code_residual(det) < 0.25).sum())} "
                  f"{time.time()-t0:6.1f}s", flush=True)

    exact, margin = evaluate(best, eval_sets)
    sat = saturation(best)
    out = os.path.join(RUNS, args.out)
    torch.save({"params": {k: v.detach().cpu() for k, v in best.items()},
                "exact": exact.cpu(), "margin": margin.cpu(), "sat": sat.cpu(),
                "args": vars(args)}, out)
    print(json.dumps({"out": out, "members": e,
                      "linear_code": int((code_residual(best) < 0.25).sum()),
                      "exact_1.0": int((exact >= 1.0).sum()),
                      "exact_1.0_and_saturated": int(((exact >= 1.0) & (sat > 0)).sum()),
                      "best_exact": exact.max().item()}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ensemble", type=int, default=16384)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.012)
    p.add_argument("--pct_start", type=float, default=0.15)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--ls", type=float, default=4.0)
    p.add_argument("--ls_end", type=float, default=0.0)
    p.add_argument("--digit_ramp", type=float, default=0.0)
    p.add_argument("--norm", type=int, default=1)
    p.add_argument("--untie", type=int, default=0)
    p.add_argument("--tie0", type=float, default=0.01)
    p.add_argument("--tie1", type=float, default=30.0)
    p.add_argument("--metric", type=str, default="l1")
    p.add_argument("--places", type=str, default="1")
    p.add_argument("--eval_places", type=str, default="1")
    p.add_argument("--eval_batch", type=int, default=2048)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--code_sigma", type=float, default=3.0)
    p.add_argument("--jitter", type=float, default=0.0)
    p.add_argument("--exhaustive1", type=int, default=1)
    p.add_argument("--init_from", type=str, default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="phase1.pt")
    a = p.parse_args()
    if a.ls_end <= 0:
        a.ls_end = a.ls
    run(a)
