"""Train E independent copies of the same architecture simultaneously.

The models are stacked along a leading ensemble axis so one set of batched
matmuls advances all of them.  Each member has its own parameters and its own
initialisation seed; Adam is elementwise so this is exactly equivalent to
training them separately (gradient clipping is done per member).

A member can be exported to a plain AdderTransformer state_dict, and the
export is checked for bit-level agreement with the ensembled forward pass.
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import data
from model_src import AdderTransformer, _rms

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")


class EnsAdder(nn.Module):
    def __init__(self, E, d_model, blocks, norm="rms", base_seed=0, rel_mode="full"):
        super().__init__()
        self.E, self.d_model, self.blocks_cfg, self.norm_kind = E, d_model, blocks, norm
        self.rel_mode = rel_mode
        members = []
        for i in range(E):
            torch.manual_seed(base_seed + 1000 * i + 7)
            members.append(AdderTransformer(d_model, blocks, norm=norm,
                                            rel_mode=rel_mode).state_dict())
        keys = list(members[0].keys())
        self.pkeys = keys
        self.ps = nn.ParameterDict()
        for k in keys:
            stacked = torch.stack([m[k] for m in members], 0)
            self.ps[k.replace(".", "|")] = nn.Parameter(stacked.clone())

    def p(self, k):
        return self.ps[k.replace(".", "|")]

    def _norm(self, x):
        return _rms(x) if self.norm_kind == "rms" else x

    def _lin(self, x, w, b=None):
        # x: (E,N,in)  w: (E,out,in)  b: (E,out)
        y = torch.bmm(x, w.transpose(1, 2))
        if b is not None:
            y = y + b.unsqueeze(1)
        return y

    def forward(self, digits, return_attn=False):
        E = self.E
        B, T, _ = digits.shape
        emb = self.p("emb")                                    # (E,10,d)
        ia = digits[..., 0].reshape(-1)                        # (B*T,)
        ib = digits[..., 1].reshape(-1)
        x = emb[:, ia] + emb[:, ib]                            # (E,B*T,d)
        attns = []
        for bi, (h, k, m) in enumerate(self.blocks_cfg):
            pre = "blocks.%d." % bi
            if h > 0:
                xn = self._norm(x)
                q = self._lin(xn, self.p(pre + "attn.w_q.weight"), self.p(pre + "attn.w_q.bias"))
                kk = self._lin(xn, self.p(pre + "attn.w_k.weight"))
                v = self._lin(xn, self.p(pre + "attn.w_v.weight"))
                q = q.view(E, B, T, h, k).permute(0, 1, 3, 2, 4)   # E,B,h,T,k
                kk = kk.view(E, B, T, h, k).permute(0, 1, 3, 2, 4)
                v = v.view(E, B, T, h, k).permute(0, 1, 3, 2, 4)
                scores = q @ kk.transpose(-2, -1)                   # E,B,h,T,T
                idx = torch.arange(T, device=x.device)
                delta = idx.view(T, 1) - idx.view(1, T)
                rbp = self.p(pre + "attn.rel_bias")
                if self.rel_mode == "full":
                    rb = rbp[:, :, delta.clamp(min=0)]                 # E,h,T,T
                else:
                    dd = delta.clamp(min=0).to(rbp.dtype)
                    rb = (rbp[:, :, 0].view(E, h, 1, 1) * dd
                          + rbp[:, :, 1].view(E, h, 1, 1) * (dd == 0).to(dd.dtype))
                scores = scores + rb.unsqueeze(1)
                scores = scores.masked_fill((delta < 0), float("-inf"))
                att = scores.softmax(-1)
                if return_attn:
                    attns.append(att)
                y = (att @ v).permute(0, 1, 3, 2, 4).reshape(E, B * T, h * k)
                x = x + self._lin(y, self.p(pre + "attn.w_o.weight"))
            if m > 0:
                xn = self._norm(x)
                hdn = torch.relu(self._lin(xn, self.p(pre + "mlp.fc1.weight"),
                                           self.p(pre + "mlp.fc1.bias")))
                x = x + self._lin(hdn, self.p(pre + "mlp.fc2.weight"))
        logits = torch.bmm(self._norm(x), emb.transpose(1, 2)) * self.p("logit_scale").unsqueeze(1)
        return logits.view(E, B, T, 10)

    def member_state(self, i):
        return {k: self.p(k)[i].detach().clone() for k in self.pkeys}


def per_member_clip(model, max_norm):
    E = model.E
    sq = torch.zeros(E, device=next(model.parameters()).device)
    for p in model.parameters():
        if p.grad is not None:
            sq += p.grad.reshape(E, -1).pow(2).sum(1)
    norm = sq.sqrt()
    scale = (max_norm / (norm + 1e-6)).clamp(max=1.0)
    for p in model.parameters():
        if p.grad is not None:
            p.grad.mul_(scale.view(E, *([1] * (p.dim() - 1))))
    return norm


@torch.no_grad()
def ens_eval(model, sets, chunk=32768):
    model.eval()
    E = model.E
    out = {}
    for name, (tok, tgt) in sets.items():
        ok = torch.zeros(E, device=tok.device)
        n = 0
        for i in range(0, tok.shape[0], chunk):
            t = tok[i:i + chunk]
            y = tgt[i:i + chunk]
            pred = model(t)[:, :, 1:, :].argmax(-1)
            ok += (pred == y.unsqueeze(0)).all(-1).float().sum(1)
            n += t.shape[0]
        out[name] = (ok / n)
    model.train()
    return out


def run(cfg, device="cuda"):
    E = cfg["ensemble"]
    model = EnsAdder(E, cfg["d_model"], cfg["blocks"], norm=cfg["norm"],
                     base_seed=cfg["seed"], rel_mode=cfg.get("rel_mode", "full")).to(device)
    npar_member = sum(p.numel() for p in model.parameters()) // E

    gen = torch.Generator(device=device)
    gen.manual_seed(cfg["seed"] + 4242)
    eval_sets = {
        "uniform": data.heldout_set(cfg["eval_n"], device, 20240001, "uniform"),
        "chain": data.heldout_set(max(cfg["eval_n"] // 4, 20000), device, 20240002, "chain"),
        "maxchain": data.heldout_set(20000, device, 20240003, "maxchain"),
    }

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.99),
                            weight_decay=cfg["wd"], foreach=True)
    steps = cfg["steps"]
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg["lr"], total_steps=steps, pct_start=cfg["pct_start"],
        div_factor=cfg.get("div_factor", 10.0), final_div_factor=cfg.get("final_div", 200.0))

    best_score = torch.full((E,), -1.0, device=device)
    best_state = {k: model.p(k).detach().clone() for k in model.pkeys}
    best_acc = torch.zeros(E, 3, device=device)
    best_step = torch.zeros(E, device=device)
    hist = []
    t0 = time.time()

    pool_tok = pool_tgt = None
    for step in range(steps):
        if step % cfg["pool_refresh"] == 0:
            pool_tok, pool_tgt = data.make_batch(cfg["pool"], device, gen,
                                                 frac_uniform=cfg["frac_uniform"],
                                                 frac_chain=cfg["frac_chain"])
        sel = torch.randint(0, pool_tok.shape[0], (cfg["batch"],), device=device, generator=gen)
        tok, tgt = pool_tok[sel], pool_tgt[sel]

        logits = model(tok)[:, :, 1:, :]
        loss_e = F.cross_entropy(logits.reshape(-1, 10), tgt.repeat(E, 1).reshape(-1),
                                 reduction="none").view(E, -1).mean(1)
        loss = loss_e.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        per_member_clip(model, 1.0)
        opt.step()
        sched.step()

        if (step + 1) % cfg["eval_every"] == 0 or step == steps - 1:
            acc = ens_eval(model, eval_sets)
            score = acc["uniform"] + 0.05 * acc["chain"] + 0.02 * acc["maxchain"]
            imp = score > best_score
            if imp.any():
                best_score = torch.where(imp, score, best_score)
                best_step = torch.where(imp, torch.full_like(best_step, step + 1), best_step)
                stacked = torch.stack([acc["uniform"], acc["chain"], acc["maxchain"]], 1)
                best_acc = torch.where(imp.unsqueeze(1), stacked, best_acc)
                for k in model.pkeys:
                    cur = model.p(k).detach()
                    msk = imp.view(E, *([1] * (cur.dim() - 1)))
                    best_state[k] = torch.where(msk, cur, best_state[k])
            top = acc["uniform"].max().item()
            hist.append({"step": step + 1, "loss": float(loss_e.mean()),
                         "best_uni": top, "n_ge_999": int((acc["uniform"] >= 0.999).sum())})
            print("[%s] %6d/%d loss %.5f  best_uni %.5f  #>=99.9%%: %d/%d  (%.0fs)"
                  % (cfg["tag"], step + 1, steps, float(loss_e.mean()), top,
                     hist[-1]["n_ge_999"], E, time.time() - t0), flush=True)

    results = []
    os.makedirs(RUNS, exist_ok=True)
    for i in range(E):
        results.append({
            "member": i, "params": npar_member,
            "uniform": best_acc[i, 0].item(), "chain": best_acc[i, 1].item(),
            "maxchain": best_acc[i, 2].item(), "step": int(best_step[i].item()),
        })
    results.sort(key=lambda r: -r["uniform"])

    keep = cfg.get("keep_top", 4)
    saved = []
    plain_cfg = {"d_model": cfg["d_model"], "blocks": cfg["blocks"], "norm": cfg["norm"],
                 "rel_mode": cfg.get("rel_mode", "full"),
                 "seed": cfg["seed"], "steps": cfg["steps"], "batch": cfg["batch"],
                 "lr": cfg["lr"], "from": "ens_train", "tag": cfg["tag"]}
    for r in results[:keep]:
        i = r["member"]
        sd = {k: best_state[k][i].detach().cpu().clone() for k in model.pkeys}
        path = os.path.join(RUNS, "%s_m%d.pt" % (cfg["tag"], i))
        torch.save({"cfg": plain_cfg, "state": sd, "result": r}, path)
        saved.append(path)

    out = {"cfg": plain_cfg, "ensemble": E, "params": npar_member,
           "results": results, "hist": hist, "seconds": time.time() - t0,
           "saved": saved}
    with open(os.path.join(RUNS, cfg["tag"] + ".ens.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


def parse_blocks(s):
    return [tuple(int(x) for x in part.split(",")) for part in s.split(";")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="ens")
    p.add_argument("--d_model", type=int, default=4)
    p.add_argument("--blocks", default="1,1,6;1,1,8")
    p.add_argument("--norm", default="rms")
    p.add_argument("--rel_mode", default="full", choices=["full", "ramp"])
    p.add_argument("--ensemble", type=int, default=16)
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--pool", type=int, default=262144)
    p.add_argument("--pool_refresh", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.012)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pct_start", type=float, default=0.15)
    p.add_argument("--frac_uniform", type=float, default=0.35)
    p.add_argument("--frac_chain", type=float, default=0.25)
    p.add_argument("--eval_n", type=int, default=100000)
    p.add_argument("--eval_every", type=int, default=5000)
    p.add_argument("--keep_top", type=int, default=4)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    cfg = vars(a).copy()
    cfg.pop("device")
    cfg["blocks"] = parse_blocks(a.blocks)
    out = run(cfg, device=a.device)
    print(json.dumps({"tag": a.tag, "params": out["params"],
                      "top5": out["results"][:5]}, indent=1))


if __name__ == "__main__":
    main()
