"""Train an ensemble of tiny adder transformers and save the best members.

Seed variance dominates at these sizes, so E independently-initialised models
are trained at once on a leading member axis.  With --rounds > 1 the run is
split into that many LR cycles; after each cycle the worst members are thrown
away and replaced by fresh random initialisations, which multiplies the number
of lottery tickets a fixed compute budget buys.  A hall of fame keeps the best
member state ever seen.

Usage:
    python train.py --d 3 --f1 6 --f2 6 --E 256 --steps 30000 --tag d3_6_6
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

import data
from ens import EnsembleAdder, single_param_count
from transfer import warm_start

CKPT_DIR = "/workspace/ckpt"


def evaluate(model, device, gen, n=200_000, batch=8192, mix=(0.35, 0.40, 0.25),
             p_trans=0.40):
    """Exact-match accuracy per member on freshly drawn held-out pairs."""
    correct = torch.zeros(model.E, device=device)
    total = 0
    with torch.no_grad():
        while total < n:
            bs = min(batch, n - total)
            a, b = data.sample_eval(bs, device, gen, mix=mix, p_trans=p_trans)
            t = data.targets(a, b)
            logits, _ = model(a, b)
            pred = logits[:, :, 1:, :].argmax(-1)
            correct += (pred == t.unsqueeze(0)).all(-1).float().sum(1)
            total += bs
    return correct / total


def reinit_members(model, opt, idx, seed, donor=None, sigma=0.6):
    """Replace the listed members with fresh weights and clear their optimizer
    moments.  With a donor the replacements are warm starts rather than random
    inits, so a warm-started run stays warm across rounds."""
    if len(idx) == 0:
        return
    fresh = EnsembleAdder(len(idx), model.d_model, model.d_ff_in, model.d_ff_out,
                          n_pos=model.n_pos, vocab=model.vocab, act=model.act,
                          d_v=model.d_v, res_bias=model.res_bias,
                          share_qk=model.share_qk, norm=model.norm,
                          q_bias=model.q_bias, self_bias=model.use_self_bias,
                          out_scale=model.out_scale,
                          res_bias_in=model.rb_in,
                          res_bias_out=model.rb_out,
                          emb_rank=model.emb_rank,
                          alibi=model.use_alibi,
                          emb0_zero=model.emb0_zero,
                          emb_fixed_up=model.emb_fixed_up,
                          plane_io=model.plane_io, ffn_in_axis=model.ffn_in_axis, seed=seed)
    fresh = fresh.to(next(model.parameters()).device)
    if donor is not None:
        warm_start(fresh, donor, sigma_max=sigma, seed=seed, verbose=False)
    with torch.no_grad():
        for name in model.spec:
            getattr(model, name)[idx] = getattr(fresh, name).detach()
    for p in model.parameters():
        st = opt.state.get(p)
        if st:
            st["exp_avg"][idx] = 0
            st["exp_avg_sq"][idx] = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--f1", type=int, default=5)
    ap.add_argument("--f2", type=int, default=5)
    ap.add_argument("--E", type=int, default=256)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--reinit_frac", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.012)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--act", type=str, default="relu")
    ap.add_argument("--dv", type=int, default=0)
    ap.add_argument("--no_res_bias", action="store_true")
    ap.add_argument("--share_qk", action="store_true")
    ap.add_argument("--no_norm", action="store_true")
    ap.add_argument("--no_qb", action="store_true")
    ap.add_argument("--no_sb", action="store_true")
    ap.add_argument("--no_ls", action="store_true")
    ap.add_argument("--no_bin2", action="store_true")
    ap.add_argument("--no_bout2", action="store_true")
    ap.add_argument("--emb_rank", type=int, default=0)
    ap.add_argument("--no_alibi", action="store_true")
    ap.add_argument("--emb0_zero", action="store_true")
    ap.add_argument("--emb_fixed_up", action="store_true")
    ap.add_argument("--plane_io", action="store_true")
    ap.add_argument("--ffn_in_axis", action="store_true")
    ap.add_argument("--alibi_mean", type=float, default=-1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--keep", type=int, default=24)
    ap.add_argument("--final_eval", type=int, default=1_000_000)
    ap.add_argument("--warm", type=float, default=0.15)
    ap.add_argument("--init_from", type=str, default=None)
    ap.add_argument("--init_member", type=int, default=None)
    ap.add_argument("--init_sigma", type=float, default=0.6)
    args = ap.parse_args()

    tag = args.tag or f"d{args.d}_{args.f1}_{args.f2}_s{args.seed}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(CKPT_DIR, exist_ok=True)

    rb = not args.no_res_bias
    rb_in = rb and not args.no_bin2
    rb_out = rb and not args.no_bout2
    donor = None
    n_params = single_param_count(args.d, args.f1, args.f2, d_v=args.dv,
                                  res_bias=rb, share_qk=args.share_qk,
                                  q_bias=not args.no_qb,
                                  self_bias=not args.no_sb,
                                  out_scale=not args.no_ls,
                                  res_bias_in=rb_in, res_bias_out=rb_out,
                                  emb_rank=args.emb_rank,
                                  alibi=not args.no_alibi,
                                  emb0_zero=args.emb0_zero,
                                  emb_fixed_up=args.emb_fixed_up,
                                  plane_io=args.plane_io,
                                  ffn_in_axis=args.ffn_in_axis)
    model = EnsembleAdder(args.E, args.d, args.f1, args.f2, act=args.act,
                          d_v=args.dv, res_bias=rb, share_qk=args.share_qk,
                          norm=not args.no_norm, q_bias=not args.no_qb,
                          self_bias=not args.no_sb, out_scale=not args.no_ls,
                          res_bias_in=rb_in, res_bias_out=rb_out,
                          emb_rank=args.emb_rank,
                          alibi=not args.no_alibi,
                          emb0_zero=args.emb0_zero,
                          emb_fixed_up=args.emb_fixed_up,
                          plane_io=args.plane_io,
                          ffn_in_axis=args.ffn_in_axis,
                          alibi_mean=args.alibi_mean, seed=args.seed).to(device)
    assert model.member_params() == n_params, (model.member_params(), n_params)
    if args.init_from:
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        mem = args.init_member
        if mem is None:
            mem = max(ck["acc"], key=lambda k: ck["acc"][k])
        print(f"[{tag}] warm start from {args.init_from} member {mem} "
              f"(donor acc {ck['acc'][mem]:.6f})", flush=True)
        donor = ck["members"][mem]
        warm_start(model, donor, sigma_max=args.init_sigma, seed=args.seed)
    print(f"[{tag}] params/member = {n_params}  E={args.E}  steps={args.steps} "
          f"rounds={args.rounds}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99),
                            weight_decay=args.wd)
    per_round = args.steps // args.rounds
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=per_round, pct_start=args.warm,
        div_factor=10.0, final_div_factor=200.0)

    gen = torch.Generator(device=device).manual_seed(1234 + args.seed)
    egen = torch.Generator(device=device).manual_seed(999)
    t0 = time.time()
    log = []
    hof_acc, hof_state = [], []          # hall of fame

    def push_hof(acc, top=None):
        """Snapshot the current best members.  Scores here are noisy (small
        eval sample), so the hall of fame is deliberately over-filled and every
        entry is rescored on a large fresh sample at the end of the run."""
        nonlocal hof_acc, hof_state
        n = args.keep if top is None else top
        order = torch.argsort(acc, descending=True)[:n].tolist()
        for i in order:
            hof_acc.append(float(acc[i]))
            hof_state.append(model.extract(int(i)))
        keep = sorted(range(len(hof_acc)), key=lambda i: -hof_acc[i])[:args.keep]
        hof_acc = [hof_acc[i] for i in keep]
        hof_state = [hof_state[i] for i in keep]

    step = 0
    for rnd in range(args.rounds):
        if rnd > 0:
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=args.lr, total_steps=per_round, pct_start=args.warm,
                div_factor=10.0, final_div_factor=200.0)
        for _ in range(per_round):
            a, b, w = data.sample_train(args.batch, device, gen)
            t = data.targets(a, b)
            logits, _ = model(a, b)
            lp = logits[:, :, 1:, :]
            E, B, P, V = lp.shape
            tt = t.reshape(1, B, P).expand(E, B, P).reshape(-1)
            ce = F.cross_entropy(lp.reshape(-1, V), tt,
                                 reduction="none").view(E, B, P)
            loss_e = (ce.mean(-1) * w).sum(-1) / w.sum().clamp(min=1.0)
            loss = loss_e.sum()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip > 0:
                sq = torch.zeros(args.E, device=device)
                for p in model.parameters():
                    if p.grad is not None:
                        sq += p.grad.reshape(args.E, -1).pow(2).sum(1)
                scale = (args.clip / (sq.sqrt() + 1e-12)).clamp(max=1.0)
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(scale.view(-1, *([1] * (p.dim() - 1))))
            opt.step()
            sched.step()
            step += 1

            if step % 2000 == 0 or step == args.steps:
                acc = evaluate(model, device, egen, n=32768)
                # These tiny models are not monotone in training time: a member
                # can peak mid-cycle and drift away again, so snapshot at every
                # evaluation rather than only at round boundaries.
                push_hof(acc, top=4)
                el = time.time() - t0
                print(f"[{tag}] r{rnd} step {step:6d} loss "
                      f"{loss_e.min().item():.4f} best {acc.max().item():.4f} "
                      f"n>=99% {int((acc >= 0.99).sum()):3d} "
                      f"({el:.0f}s, {step / max(el, 1e-9):.1f} it/s)", flush=True)
                log.append(dict(step=step, best=float(acc.max()),
                                n99=int((acc >= 0.99).sum())))

        acc = evaluate(model, device, egen, n=131072)
        push_hof(acc)
        if rnd < args.rounds - 1:
            k = int(math.floor(args.E * args.reinit_frac))
            worst = torch.argsort(acc)[:k].tolist()
            reinit_members(model, opt, worst, seed=1000 * (rnd + 1) + args.seed,
                           donor=donor, sigma=args.init_sigma)
            print(f"[{tag}] round {rnd} done, best {acc.max().item():.5f}; "
                  f"reinitialised {k} members", flush=True)

    # ---- rescore the hall of fame on a big fresh held-out sample ---------
    if hof_state:
        scorer = EnsembleAdder(len(hof_state), args.d, args.f1, args.f2,
                               act=args.act, d_v=args.dv, res_bias=rb,
                               share_qk=args.share_qk, norm=not args.no_norm,
                               q_bias=not args.no_qb, self_bias=not args.no_sb,
                               out_scale=not args.no_ls, res_bias_in=rb_in,
                               res_bias_out=rb_out,
                               emb_rank=args.emb_rank,
                               alibi=not args.no_alibi,
                               emb0_zero=args.emb0_zero,
                               emb_fixed_up=args.emb_fixed_up,
                               plane_io=args.plane_io,
                               ffn_in_axis=args.ffn_in_axis,
                               seed=0).to(device)
        with torch.no_grad():
            for name in scorer.spec:
                getattr(scorer, name).copy_(
                    torch.stack([s[name] for s in hof_state]))
        final = evaluate(scorer, device,
                         torch.Generator(device=device).manual_seed(4242),
                         n=args.final_eval)
        order = torch.argsort(final, descending=True).tolist()
    else:
        final, order = torch.zeros(1), [0]

    res = dict(tag=tag, params=n_params, d=args.d, f1=args.f1, f2=args.f2,
               act=args.act, d_v=args.dv, res_bias=rb,
               share_qk=args.share_qk, norm=not args.no_norm,
              q_bias=not args.no_qb, self_bias=not args.no_sb,
              out_scale=not args.no_ls, res_bias_in=rb_in,
              res_bias_out=rb_out, emb_rank=args.emb_rank,
              alibi=not args.no_alibi, emb0_zero=args.emb0_zero,
              emb_fixed_up=args.emb_fixed_up,
              plane_io=args.plane_io, ffn_in_axis=args.ffn_in_axis,
              E=args.E,
               steps=args.steps, rounds=args.rounds, seed=args.seed, lr=args.lr,
               final_eval=args.final_eval,
               top=[dict(member=int(i), acc=float(final[i])) for i in order],
               n_ge_9990=int((final >= 0.999).sum()),
               n_ge_9999=int((final >= 0.9999).sum()), log=log)
    print(f"[{tag}] FINAL best={final.max().item():.6f} "
          f"n>=99.9%={res['n_ge_9990']} n>=99.99%={res['n_ge_9999']}", flush=True)

    torch.save(dict(cfg=dict(d_model=args.d, d_ff_in=args.f1, d_ff_out=args.f2,
                             act=args.act, d_v=args.dv, res_bias=rb,
                             share_qk=args.share_qk, q_bias=not args.no_qb,
                             self_bias=not args.no_sb, out_scale=not args.no_ls,
                             res_bias_in=rb_in, res_bias_out=rb_out,
                             emb_rank=args.emb_rank,
                             alibi=not args.no_alibi,
                             emb0_zero=args.emb0_zero,
                             emb_fixed_up=args.emb_fixed_up,
                             plane_io=args.plane_io,
                             ffn_in_axis=args.ffn_in_axis,
                             norm=not args.no_norm),
                    members={int(i): hof_state[i] for i in order},
                    acc={int(i): float(final[i]) for i in order},
                    meta=res),
               os.path.join(CKPT_DIR, f"{tag}.pt"))
    with open(os.path.join(CKPT_DIR, f"{tag}.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
