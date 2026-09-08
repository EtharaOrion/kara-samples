"""Trainer for the tiny addition transformer.

Usage:
    python train.py --tag run1 --d_model 5 --blocks "1,1,4;1,1,10" --steps 120000
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

import data
from model_src import AdderTransformer

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def parse_blocks(s):
    out = []
    for part in s.split(";"):
        h, k, m = part.split(",")
        out.append((int(h), int(k), int(m)))
    return out


def n_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def evaluate(model, sets):
    model.eval()
    res = {}
    for name, (tok, tgt) in sets.items():
        correct = 0
        total = 0
        for i in range(0, tok.shape[0], 65536):
            t = tok[i:i + 65536]
            y = tgt[i:i + 65536]
            pred = model(t)[:, 1:, :].argmax(-1)
            correct += (pred == y).all(-1).sum().item()
            total += t.shape[0]
        res[name] = correct / total
    model.train()
    return res


@torch.no_grad()
def attention_stats(model, tok):
    """How much does the attention pattern actually move with the input?"""
    model.eval()
    _, attns = model(tok[:4096], return_attn=True)
    stats = []
    for a in attns:
        mean = a.mean(0, keepdim=True)
        # mean absolute deviation of the attention map from its input-average
        dev = (a - mean).abs().mean().item()
        # fraction of query rows whose argmax is not the batch-modal argmax
        am = a.argmax(-1)  # B,H,T
        mode = am.mode(dim=0).values.unsqueeze(0)
        varies = (am != mode).float().mean().item()
        stats.append({"mad": dev, "argmax_varies": varies})
    model.train()
    return stats


def run(cfg, device="cuda", verbose=True, log_path=None):
    torch.manual_seed(cfg["seed"])
    model = AdderTransformer(cfg["d_model"], cfg["blocks"], norm=cfg["norm"]).to(device)
    npar = n_params(model)

    gen = torch.Generator(device=device)
    gen.manual_seed(cfg["seed"] + 9973)

    eval_sets = {
        "uniform": data.heldout_set(cfg["eval_n"], device, 20240001, "uniform"),
        "chain": data.heldout_set(cfg["eval_n"] // 4, device, 20240002, "chain"),
        "maxchain": data.heldout_set(20000, device, 20240003, "maxchain"),
    }

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.99),
                            weight_decay=cfg["wd"])
    steps = cfg["steps"]
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg["lr"], total_steps=steps, pct_start=cfg["pct_start"],
        div_factor=cfg.get("div_factor", 10.0), final_div_factor=cfg.get("final_div", 200.0))

    best = {"score": -1.0}
    log = []
    t0 = time.time()
    ent_w = cfg.get("ent_w", 0.0)
    ent_start = int(steps * cfg.get("ent_start", 0.25))

    for step in range(steps):
        tok, tgt = data.make_batch(cfg["batch"], device, gen,
                                   frac_uniform=cfg["frac_uniform"],
                                   frac_chain=cfg["frac_chain"])
        if ent_w > 0 and step >= ent_start:
            logits, attns = model(tok, return_attn=True)
        else:
            logits, attns = model(tok), None
        loss = F.cross_entropy(logits[:, 1:, :].reshape(-1, 10), tgt.reshape(-1))
        if attns is not None:
            ent = 0.0
            for a in attns:
                ent = ent + (-(a.clamp_min(1e-9).log() * a).sum(-1)).mean()
            loss = loss + ent_w * ent
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if (step + 1) % cfg["eval_every"] == 0 or step == steps - 1:
            acc = evaluate(model, eval_sets)
            score = acc["uniform"] + 0.05 * acc["chain"] + 0.02 * acc["maxchain"]
            rec = {"step": step + 1, "loss": float(loss.item()), **acc,
                   "lr": sched.get_last_lr()[0], "t": time.time() - t0}
            log.append(rec)
            if verbose:
                print(f"[{cfg.get('tag','')}] step {step+1}/{steps} loss {loss.item():.5f} "
                      f"uni {acc['uniform']:.5f} chain {acc['chain']:.5f} "
                      f"max {acc['maxchain']:.5f} ({rec['t']:.0f}s)", flush=True)
            if score > best["score"]:
                best = {"score": score, "step": step + 1, "acc": acc,
                        "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}

    model.load_state_dict(best["state"])
    stats = attention_stats(model, eval_sets["chain"][0])
    out = {
        "cfg": cfg, "params": npar, "best_step": best["step"], "acc": best["acc"],
        "attn": stats, "log": log, "seconds": time.time() - t0,
    }
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(out, f, indent=1)
    return model, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="run")
    p.add_argument("--d_model", type=int, default=5)
    p.add_argument("--blocks", default="1,1,4;1,1,10")
    p.add_argument("--norm", default="rms")
    p.add_argument("--steps", type=int, default=80000)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pct_start", type=float, default=0.15)
    p.add_argument("--ent_w", type=float, default=0.0)
    p.add_argument("--ent_start", type=float, default=0.25)
    p.add_argument("--frac_uniform", type=float, default=0.35)
    p.add_argument("--frac_chain", type=float, default=0.25)
    p.add_argument("--eval_n", type=int, default=100000)
    p.add_argument("--eval_every", type=int, default=5000)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    cfg = vars(a).copy()
    cfg.pop("device")
    cfg["blocks"] = parse_blocks(a.blocks)
    os.makedirs(RUNS, exist_ok=True)
    model, out = run(cfg, device=a.device, log_path=os.path.join(RUNS, a.tag + ".json"))
    torch.save({"cfg": cfg, "state": model.state_dict(), "result": {k: v for k, v in out.items() if k != "log"}},
               os.path.join(RUNS, a.tag + ".pt"))
    print(json.dumps({"tag": a.tag, "params": out["params"], "acc": out["acc"],
                      "attn": out["attn"]}, indent=1))


if __name__ == "__main__":
    main()
