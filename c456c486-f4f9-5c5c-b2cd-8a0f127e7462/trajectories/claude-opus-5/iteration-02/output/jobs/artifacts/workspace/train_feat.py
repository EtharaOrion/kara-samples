"""Replace the 55-entry pair table with a 9-parameter learned ReLU featurizer.

The table is used only as a training scaffold: a mixing weight lam is annealed
from 0 (table) to 1 (featurizer) while the whole model keeps training, then the
model is fine-tuned with the featurizer alone and exported without the table.
"""

import argparse
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import make_batch
from model_def import TinyAdder
from train import count_params, evaluate, evaluate_edges

HERE = os.path.dirname(os.path.abspath(__file__))


class DualAdder(TinyAdder):
    """TinyAdder whose pair channel mixes the scaffold table and the featurizer."""

    def __init__(self, **kw):
        super().__init__(pair_mode="feat", **kw)
        self.pair_code = nn.Parameter(torch.randn(55) * 0.5)
        pair_index = torch.zeros(10, 10, dtype=torch.long)
        r = 0
        for i in range(10):
            for j in range(i, 10):
                pair_index[i, j] = r
                pair_index[j, i] = r
                r += 1
        self.register_buffer("pair_index", pair_index, persistent=False)
        self.lam = 0.0

    def pair_feature(self, da, db, code):
        tab = self.pair_code[self.pair_index[da, db]]
        if self.lam <= 0.0:
            return tab
        feat = F.relu(code[..., None] * self.fw + self.fb) @ self.fo
        if self.lam >= 1.0:
            return feat
        return (1 - self.lam) * tab + self.lam * feat


def main(args):
    torch.set_num_threads(args.threads)
    device = args.device
    torch.manual_seed(args.seed)

    src = torch.load(args.init, map_location="cpu", weights_only=False)
    scfg = src["config"]
    kw = dict(d_model=scfg["d_model"], d_ff=scfg["d_ff"],
              head=scfg["head"],
              mlp_full_out=scfg["mlp_out"] == "full", d_feat=args.d_feat)
    model = DualAdder(**kw).to(device)
    missing = model.load_state_dict(src["state"], strict=False)
    print(f"[{args.tag}] loaded {args.init} (unmatched: {missing.missing_keys})", flush=True)

    export_kw = dict(kw)
    export_kw["pair_mode"] = "feat"
    nparam = count_params(TinyAdder(**export_kw))
    print(f"[{args.tag}] export params={nparam}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98),
                            weight_decay=0.0)
    total = args.anneal + args.finetune
    for step in range(total):
        # lam: 0 for a short warm-up, then linear to 1, then pure featurizer
        w = args.anneal * 0.1
        model.lam = 0.0 if step < w else min(1.0, (step - w) / max(1, args.anneal - w))
        lr = args.lr if step < args.anneal else args.lr2 * (
            0.5 * (1 + torch.cos(torch.tensor(
                (step - args.anneal) / max(1, args.finetune) * 3.14159)).item()))
        for g in opt.param_groups:
            g["lr"] = lr
        da, db, y = make_batch(args.bs, device)
        logits = model(da, db)
        loss = F.cross_entropy(logits[:, 1:].reshape(-1, 10), y[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % args.eval_every == 0 or step + 1 == total:
            acc = evaluate(model, device, n=100_000, bs=100_000)
            print(f"[{args.tag}] step {step+1} lam {model.lam:.3f} loss {loss.item():.5f} "
                  f"acc {acc:.5f} lr {lr:.2e}", flush=True)

    model.lam = 1.0
    out = TinyAdder(**export_kw).to(device)
    sd = {k: v for k, v in model.state_dict().items() if k != "pair_code"}
    out.load_state_dict(sd, strict=False)
    acc_u = evaluate(out, device, n=args.final_n, bs=100_000, uniform=True)
    acc_s = evaluate(out, device, n=args.final_n, bs=100_000, uniform=False)
    ne, nt, bad = evaluate_edges(out, device)
    print(f"[{args.tag}] FINAL params={nparam} uniform={acc_u:.6f} stress={acc_s:.6f} "
          f"edges={ne}/{nt} bad={bad}", flush=True)

    cfg = dict(scfg)
    cfg.update(tag=args.tag, pair_mode="feat", d_feat=args.d_feat)
    with open(os.path.join(HERE, "runs.jsonl"), "a") as f:
        f.write(json.dumps({"tag": args.tag, "params": nparam, "uniform": acc_u,
                            "stress": acc_s, "edges": f"{ne}/{nt}", "config": cfg}) + "\n")
    os.makedirs(os.path.join(HERE, "ckpt"), exist_ok=True)
    torch.save({"state": out.state_dict(), "config": cfg, "uniform": acc_u,
                "stress": acc_s, "params": nparam},
               os.path.join(HERE, "ckpt", f"{args.tag}.pt"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="feat")
    p.add_argument("--init", required=True, help="table checkpoint to start from")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d_feat", type=int, default=3)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--lr2", type=float, default=1.5e-3)
    p.add_argument("--bs", type=int, default=2048)
    p.add_argument("--anneal", type=int, default=8000)
    p.add_argument("--finetune", type=int, default=12000)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--final_n", type=int, default=1_000_000)
    main(p.parse_args())
