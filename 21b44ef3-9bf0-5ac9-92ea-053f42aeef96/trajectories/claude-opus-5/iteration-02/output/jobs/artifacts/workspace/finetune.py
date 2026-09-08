"""Continue training a saved member on its own, with a fresh eval set.

Selection inside a 192-member ensemble is optimistic, so this re-scores the
candidate on evaluation sets built from different seeds and keeps the best
state seen under that independent measurement.
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

import data
from model_src import AdderTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")


def load_model(path, device="cuda"):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    m = AdderTransformer(cfg["d_model"], [tuple(b) for b in cfg["blocks"]],
                         norm=cfg["norm"], rel_mode=cfg.get("rel_mode", "full")).to(device)
    m.load_state_dict(ck["state"])
    return m, cfg, ck


@torch.no_grad()
def big_eval(model, sets, chunk=131072):
    model.eval()
    out = {}
    for name, (tok, tgt) in sets.items():
        ok = 0
        for i in range(0, tok.shape[0], chunk):
            p = model(tok[i:i + chunk])[:, 1:, :].argmax(-1)
            ok += (p == tgt[i:i + chunk]).all(-1).sum().item()
        out[name] = ok / tok.shape[0]
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--final_div", type=float, default=100.0)
    ap.add_argument("--pct_start", type=float, default=0.05)
    ap.add_argument("--frac_uniform", type=float, default=0.30)
    ap.add_argument("--frac_chain", type=float, default=0.35)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--eval_n", type=int, default=400000)
    ap.add_argument("--seed", type=int, default=31337)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    dev = a.device
    model, cfg, ck = load_model(a.ckpt, dev)
    tag = a.tag or (os.path.basename(a.ckpt)[:-3] + "_ft")
    npar = sum(p.numel() for p in model.parameters())

    sets = {
        "uniform": data.heldout_set(a.eval_n, dev, a.seed + 1, "uniform"),
        "chain": data.heldout_set(a.eval_n // 4, dev, a.seed + 2, "chain"),
        "maxchain": data.heldout_set(50000, dev, a.seed + 3, "maxchain"),
    }
    start = big_eval(model, sets)
    print("[%s] params %d  start %s" % (tag, npar, {k: round(v, 6) for k, v in start.items()}),
          flush=True)

    gen = torch.Generator(device=dev)
    gen.manual_seed(a.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.steps, pct_start=a.pct_start,
        div_factor=5.0, final_div_factor=a.final_div)

    best = {"score": start["uniform"] + 0.05 * start["chain"] + 0.02 * start["maxchain"],
            "acc": start, "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
            "step": 0}
    t0 = time.time()
    for step in range(a.steps):
        tok, tgt = data.make_batch(a.batch, dev, gen, frac_uniform=a.frac_uniform,
                                   frac_chain=a.frac_chain)
        loss = F.cross_entropy(model(tok)[:, 1:, :].reshape(-1, 10), tgt.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % a.eval_every == 0 or step == a.steps - 1:
            acc = big_eval(model, sets)
            score = acc["uniform"] + 0.05 * acc["chain"] + 0.02 * acc["maxchain"]
            if score > best["score"]:
                best = {"score": score, "acc": acc, "step": step + 1,
                        "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            print("[%s] %6d/%d loss %.6f uni %.6f chain %.6f max %.6f | best uni %.6f (%.0fs)"
                  % (tag, step + 1, a.steps, loss.item(), acc["uniform"], acc["chain"],
                     acc["maxchain"], best["acc"]["uniform"], time.time() - t0), flush=True)

    model.load_state_dict(best["state"])
    out_cfg = dict(cfg)
    out_cfg["finetuned_from"] = a.ckpt
    path = os.path.join(RUNS, tag + ".pt")
    torch.save({"cfg": out_cfg, "state": {k: v.cpu() for k, v in best["state"].items()},
                "result": {"params": npar, **best["acc"], "step": best["step"]}}, path)
    print(json.dumps({"tag": tag, "path": path, "params": npar, "acc": best["acc"],
                      "start": start}, indent=1))


if __name__ == "__main__":
    main()
