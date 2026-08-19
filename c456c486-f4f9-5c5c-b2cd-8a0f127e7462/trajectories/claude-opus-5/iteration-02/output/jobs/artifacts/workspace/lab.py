"""Instrumented training runs used to find a recipe that learns carry chains."""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from data import MAXPOW, make_batch
from model_def import TinyAdder

HERE = os.path.dirname(os.path.abspath(__file__))


def count_params(m):
    return sum(p.numel() for p in m.parameters())


class NoisyAdder(TinyAdder):
    """Training-only subclass that perturbs the value read out of attention.

    Left alone, training finds an analogue carry: the attention weights settle
    into a geometric series with ratio 1/10 (slope -> ln 10) so the readout is
    the fractional value of the prefix and the carry is a threshold on it.
    That needs ~10^-k relative precision for a propagate chain of length k, so
    it fails on long chains.  Multiplying the readout by (1 + eps) with eps
    scaled to the batch's own magnitude makes such precision unusable while
    leaving a discrete carry lookahead, whose margins are order one, intact.
    The noise is off in eval mode and absent from the exported model.
    """

    noise = 0.0

    def forward(self, da, db):
        B, S = da.shape
        code = self.digit_code[da] + self.digit_code[db]
        pair = self.pair_feature(da, db, code)
        rest = code.new_zeros(B, S, self.d_model - 2)
        x = torch.cat([code[..., None], pair[..., None], rest], dim=-1)

        q = x @ self.wq + self.bq
        k = x @ self.wk + self.bk
        v = x @ self.wv + self.bv
        scores = q @ k.transpose(1, 2) + self.slope * self.rel[:S, :S]
        scores = scores.masked_fill(self.attn_mask[:S, :S], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        self.attn = attn
        r = attn @ v
        if self.training and self.noise > 0:
            r = r + torch.randn_like(r) * (self.noise * r.detach().abs().mean())
        x = x + r @ self.wo

        h = F.relu(x @ self.w1 + self.b1)
        upd = h @ self.w2 + self.b2
        if self.d_out == self.d_model:
            x = x + upd
        else:
            x = torch.cat([x[..., :1] + upd, x[..., 1:]], dim=-1)

        if self.head == "tied":
            return -self.tau * (x[..., 0:1] - self.digit_code) ** 2
        if self.head == "dist":
            return -self.tau * (x[..., 0:1] - self.centre) ** 2
        return x @ self.wout + self.bout


@torch.no_grad()
def diagnose(model, device, n=40000, maxlen=MAXPOW):
    """Exact-match accuracy overall and split by longest propagate run."""
    model.eval()
    da, db, y = make_batch(n, device, maxlen=maxlen)
    pred = model(da, db).argmax(-1)
    ok = (pred[:, 1:] == y[:, 1:])
    exact = ok.all(-1).float().mean().item()
    perpos = ok.float().mean(0)

    # longest run of digit pairs summing to 9 (a propagate chain) per sample
    prop = ((da + db) == 9).int()
    run = torch.zeros_like(prop)
    best = torch.zeros(n, dtype=torch.int32, device=device)
    cur = torch.zeros(n, dtype=torch.int32, device=device)
    for i in range(prop.shape[1]):
        cur = (cur + 1) * prop[:, i]
        best = torch.maximum(best, cur)
    buckets = {}
    for lo, hi in [(0, 0), (1, 1), (2, 3), (4, 6), (7, 14)]:
        m = (best >= lo) & (best <= hi)
        if m.any():
            buckets[f"{lo}-{hi}"] = round(ok[m].all(-1).float().mean().item(), 4)
    model.train()
    return exact, perpos, buckets


@torch.no_grad()
def attn_stats(model, device, n=4096):
    model.eval()
    da, db, _ = make_batch(n, device)
    code = model.digit_code[da] + model.digit_code[db]
    pair = model.pair_feature(da, db, code)
    rest = code.new_zeros(n, da.shape[1], model.d_model - 2)
    x = torch.cat([code[..., None], pair[..., None], rest], -1)
    q = x @ model.wq + model.bq
    k = x @ model.wk + model.bk
    s = q @ k.transpose(1, 2) + model.slope * model.rel[:da.shape[1], :da.shape[1]]
    s = s.masked_fill(model.attn_mask[:da.shape[1], :da.shape[1]], float("-inf"))
    a = torch.softmax(s, -1)[:, 1:]
    model.train()
    return a.max(-1).values.mean().item(), -(a.clamp_min(1e-9).log() * a).sum(-1).mean().item()


def run(args):
    torch.set_num_threads(args.threads)
    dev = args.device
    torch.manual_seed(args.seed)
    cls = NoisyAdder if args.noise > 0 else TinyAdder
    m = cls(d_model=args.d_model, d_ff=args.d_ff, head=args.head,
            mlp_full_out=args.mlp_out == "full", pair_mode=args.pair_mode,
            d_feat=args.d_feat).to(dev)
    m.noise = args.noise
    if args.init:
        src = torch.load(args.init, map_location="cpu", weights_only=False)
        m.load_state_dict(src["state"], strict=False)
        print(f"[{args.tag}] warm start {args.init}", flush=True)
    if not args.init:
        with torch.no_grad():
            m.slope.fill_(args.slope0)
    print(f"[{args.tag}] params={count_params(m)} {vars(args)}", flush=True)

    attn_names = {"wq", "bq", "wk", "bk", "slope"}
    groups = [
        {"params": [p for n, p in m.named_parameters() if n not in attn_names], "lr": args.lr},
        {"params": [p for n, p in m.named_parameters() if n in attn_names],
         "lr": args.lr * args.attn_mult},
    ]
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.98), weight_decay=0.0,
                            fused=dev == "cuda")
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr, args.lr * args.attn_mult], total_steps=args.steps,
        pct_start=args.pct_start)
    t0 = time.time()
    best, best_state = -1.0, None
    for step in range(args.steps):
        if args.curric > 0:
            f = min(1.0, step / args.curric)
            maxlen = int(round(args.len0 + (MAXPOW - args.len0) * f))
        else:
            maxlen = args.fixlen if args.fixlen else MAXPOW
        if args.noise > 0:
            m.noise = args.noise if step < args.steps * args.noise_off else 0.0
        da, db, y = make_batch(args.bs, dev, maxlen=maxlen)
        loss = F.cross_entropy(m(da, db)[:, 1:].reshape(-1, 10), y[:, 1:].reshape(-1))
        if args.ent_reg > 0 and step < args.ent_anneal:
            beta = args.ent_reg * (1 - step / args.ent_anneal)
            a = m.attn[:, 1:]
            ent = -(a.clamp_min(1e-9).log() * a).sum(-1).mean()
            loss = loss - beta * ent
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            ex, perpos, buckets = diagnose(m, dev, maxlen=maxlen)
            mx, ent = attn_stats(m, dev)
            print(f"[{args.tag}] {step+1} len{maxlen} loss {loss.item():.4f} exact {ex:.4f} "
                  f"chain {buckets} attn(max {mx:.2f} ent {ent:.2f}) slope {m.slope.item():.1f} "
                  f"{time.time()-t0:.0f}s", flush=True)
            if step + 1 > args.steps * 0.5 and ex >= best:
                best, best_state = ex, {k: v.clone() for k, v in m.state_dict().items()}
    if best_state:
        m.load_state_dict(best_state)
    from train import evaluate, evaluate_edges
    acc_u = evaluate(m, dev, n=args.final_n, bs=100_000, uniform=True)
    acc_s = evaluate(m, dev, n=args.final_n, bs=100_000, uniform=False)
    ne, nt, bad = evaluate_edges(m, dev)
    print(f"[{args.tag}] FINAL params={count_params(m)} uniform={acc_u:.6f} "
          f"stress={acc_s:.6f} edges={ne}/{nt} bad={bad}", flush=True)
    cfg = vars(args)
    with open(os.path.join(HERE, "runs.jsonl"), "a") as f:
        f.write(json.dumps({"tag": args.tag, "params": count_params(m), "uniform": acc_u,
                            "stress": acc_s, "edges": f"{ne}/{nt}", "config": cfg}) + "\n")
    os.makedirs(os.path.join(HERE, "ckpt"), exist_ok=True)
    torch.save({"state": m.state_dict(), "config": cfg, "uniform": acc_u, "stress": acc_s,
                "params": count_params(m)}, os.path.join(HERE, "ckpt", f"{args.tag}.pt"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="lab")
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--init", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d_model", type=int, default=3)
    p.add_argument("--d_ff", type=int, default=4)
    p.add_argument("--head", default="linear", choices=["tied", "linear", "dist"])
    p.add_argument("--mlp_out", default="full", choices=["full", "slim"])
    p.add_argument("--pair_mode", default="table", choices=["table", "feat"])
    p.add_argument("--d_feat", type=int, default=3)
    p.add_argument("--slope0", type=float, default=2.0)
    p.add_argument("--attn_mult", type=float, default=1.0)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--noise_off", type=float, default=0.85)
    p.add_argument("--ent_reg", type=float, default=0.0)
    p.add_argument("--ent_anneal", type=int, default=12000)
    p.add_argument("--lr", type=float, default=6e-3)
    p.add_argument("--pct_start", type=float, default=0.05)
    p.add_argument("--bs", type=int, default=4096)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--curric", type=int, default=0)
    p.add_argument("--len0", type=int, default=2)
    p.add_argument("--fixlen", type=int, default=0)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--final_n", type=int, default=1_000_000)
    run(p.parse_args())
