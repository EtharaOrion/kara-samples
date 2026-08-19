"""Trainer for the tiny addition transformer.

Trains a TinyAdder on freshly sampled operand pairs and writes the trained
weights into /workspace/submission.py.
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from data import MAXPOW, make_batch, _labels, _pad
from model_def import SEQ, TinyAdder

HERE = os.path.dirname(os.path.abspath(__file__))


def count_params(m):
    return sum(p.numel() for p in m.parameters())


@torch.no_grad()
def evaluate(model, device, n=1_000_000, bs=100_000, uniform=True):
    model.eval()
    ok = 0
    for _ in range(n // bs):
        da, db, y = make_batch(bs, device, uniform=uniform)
        pred = model(da, db).argmax(-1)
        ok += (pred[:, 1:] == y[:, 1:]).all(-1).sum().item()
    model.train()
    return ok / (n // bs * bs)


@torch.no_grad()
def evaluate_edges(model, device):
    """Deterministic edge cases plus a few structured stress families."""
    cases = [(0, 0), (0, 99999999999999), (99999999999999, 99999999999999),
             (1, 99999999999999), (99999999999999, 1), (12345678901234, 87654321098766),
             (55555555555555, 44444444444445), (99999999999999, 0), (9, 1), (1, 9)]
    for k in range(14):
        cases.append((10 ** k - 1, 1))
        cases.append((10 ** k, 10 ** k))
        cases.append((10 ** k - 1, 10 ** k - 1))
    das, dbs, want = [], [], []
    for a, b in cases:
        das.append([(a // 10 ** i) % 10 for i in range(MAXPOW)])
        dbs.append([(b // 10 ** i) % 10 for i in range(MAXPOW)])
        want.append(a + b)
    da = _pad(torch.tensor(das, device=device))
    db = _pad(torch.tensor(dbs, device=device))
    pred = model(da, db).argmax(-1)[:, 1:]
    pw = torch.tensor([10 ** i for i in range(SEQ - 1)], device=device)
    got = (pred * pw).sum(-1).tolist()
    bad = [(c, g, w) for c, g, w in zip(cases, got, want) if g != w]
    return len(cases) - len(bad), len(cases), bad[:5]


def train(args):
    torch.set_num_threads(args.threads)
    device = args.device
    torch.manual_seed(args.seed)
    model = TinyAdder(d_model=args.d_model, d_ff=args.d_ff,
                      head=args.head,
                      mlp_full_out=args.mlp_out == "full",
                      pair_mode=args.pair_mode, d_feat=args.d_feat).to(device)
    if args.init:
        src = torch.load(args.init, map_location="cpu", weights_only=False)
        print(f"[{args.tag}] warm start from {args.init} "
              f"(uniform={src.get('uniform')})", flush=True)
        model.load_state_dict(src["state"], strict=False)
    nparam = count_params(model)
    print(f"[{args.tag}] params={nparam} config={vars(args)}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98),
                            weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)

    t0 = time.time()
    best = -1.0
    best_state = None
    for step in range(args.steps):
        da, db, y = make_batch(args.bs, device)
        logits = model(da, db)
        loss = F.cross_entropy(logits[:, 1:].reshape(-1, 10), y[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            acc = evaluate(model, device, n=args.eval_n, bs=min(args.eval_n, 100_000))
            el = time.time() - t0
            print(f"[{args.tag}] step {step+1} loss {loss.item():.5f} "
                  f"acc {acc:.5f} lr {sched.get_last_lr()[0]:.2e} {el:.0f}s", flush=True)
            if acc >= best:
                best = acc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    acc_u = evaluate(model, device, n=args.final_n, bs=100_000, uniform=True)
    acc_s = evaluate(model, device, n=args.final_n, bs=100_000, uniform=False)
    ne, nt, bad = evaluate_edges(model, device)
    print(f"[{args.tag}] FINAL params={nparam} uniform={acc_u:.6f} stress={acc_s:.6f} "
          f"edges={ne}/{nt} bad={bad}", flush=True)

    out = {"tag": args.tag, "params": nparam, "uniform": acc_u, "stress": acc_s,
           "edges": f"{ne}/{nt}", "config": vars(args)}
    with open(os.path.join(HERE, "runs.jsonl"), "a") as f:
        f.write(json.dumps(out) + "\n")
    ckpt = os.path.join(HERE, "ckpt", f"{args.tag}.pt")
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    torch.save({"state": model.state_dict(), "config": vars(args),
                "uniform": acc_u, "stress": acc_s, "params": nparam}, ckpt)
    return model, acc_u, acc_s


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="run")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--init", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d_model", type=int, default=3)
    p.add_argument("--pair_mode", default="table", choices=["table", "feat"])
    p.add_argument("--d_feat", type=int, default=3)
    p.add_argument("--d_ff", type=int, default=4)
    p.add_argument("--head", default="tied", choices=["tied", "linear"])
    p.add_argument("--mlp_out", default="full", choices=["full", "slim"])
    p.add_argument("--lr", type=float, default=4e-3)
    p.add_argument("--bs", type=int, default=2048)
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--eval_n", type=int, default=100_000)
    p.add_argument("--final_n", type=int, default=2_000_000)
    train(p.parse_args())
