import copy
import math
import os
import random
import sys
import time

import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, "/workspace")
from submission import Model

DEVICE = "cuda"
BATCH = 8192
MAX_N = 100_000_000
MIN_N = 10_000_000


def digits8(x):
    p = torch.tensor([1, 10, 100, 1000, 10000, 100000, 1000000, 10000000], device=x.device)
    return (x[:, None] // p % 10).long()


def make_batch(n, structured=0.35):
    a = torch.randint(MIN_N, MAX_N, (n,), device=DEVICE)
    b = torch.randint(MIN_N, MAX_N, (n,), device=DEVICE)
    k = int(n * structured)
    if k:
        kind = torch.randint(0, 8, (k,), device=DEVICE)
        aa = torch.randint(MIN_N, MAX_N, (k,), device=DEVICE)
        bb = torch.randint(MIN_N, MAX_N, (k,), device=DEVICE)
        # Exact and near complements around powers of ten / 100m.
        m = kind == 0
        aa[m] = torch.randint(MIN_N, 90_000_001, (int(m.sum()),), device=DEVICE)
        bb[m] = MAX_N - aa[m]
        m = kind == 1
        aa[m] = torch.randint(MIN_N, 90_000_001, (int(m.sum()),), device=DEVICE)
        delta = torch.randint(-10, 11, (int(m.sum()),), device=DEVICE)
        bb[m] = (MAX_N - aa[m] + delta).clamp(MIN_N, MAX_N - 1)
        # Asymmetric long 9/0 suffixes with arbitrary prefixes and partners.
        m = kind == 2
        c = int(m.sum())
        lengths = torch.randint(1, 8, (c,), device=DEVICE)
        powers = torch.tensor([10,100,1000,10000,100000,1000000,10000000], device=DEVICE)[lengths-1]
        prefix = torch.randint(1, 10_000_000, (c,), device=DEVICE)
        aa[m] = (prefix // powers * powers + powers - 1).clamp(MIN_N, MAX_N - 1)
        bb[m] = torch.randint(MIN_N, MAX_N, (c,), device=DEVICE)
        # Round values at arbitrary decimal boundaries, plus nearby offset.
        m = kind == 3
        c = int(m.sum())
        lengths = torch.randint(1, 8, (c,), device=DEVICE)
        powers = torch.tensor([10,100,1000,10000,100000,1000000,10000000], device=DEVICE)[lengths-1]
        aa[m] = (aa[m] // powers * powers).clamp(MIN_N, MAX_N - 1)
        bb[m] = (bb[m] // powers * powers + torch.randint(-2, 3, (c,), device=DEVICE)).clamp(MIN_N, MAX_N - 1)
        # Repeated-digit values.
        m = kind == 4
        c = int(m.sum())
        da = torch.randint(1, 10, (c,), device=DEVICE)
        db = torch.randint(1, 10, (c,), device=DEVICE)
        aa[m] = da * 11_111_111
        bb[m] = db * 11_111_111
        # Sparse interior digits, preserving full-width leading digit.
        m = kind == 5
        c = int(m.sum())
        lead_a = torch.randint(1, 10, (c,), device=DEVICE) * 10_000_000
        lead_b = torch.randint(1, 10, (c,), device=DEVICE) * 10_000_000
        place_a = torch.tensor([1,10,100,1000,10000,100000,1000000], device=DEVICE)[torch.randint(0,7,(c,),device=DEVICE)]
        place_b = torch.tensor([1,10,100,1000,10000,100000,1000000], device=DEVICE)[torch.randint(0,7,(c,),device=DEVICE)]
        aa[m] = lead_a + torch.randint(0,10,(c,),device=DEVICE) * place_a
        bb[m] = lead_b + torch.randint(0,10,(c,),device=DEVICE) * place_b
        # Near extrema.
        m = kind == 6
        c = int(m.sum())
        aa[m] = torch.where(torch.rand(c,device=DEVICE)<.5, MIN_N+torch.randint(0,1001,(c,),device=DEVICE), MAX_N-1-torch.randint(0,1001,(c,),device=DEVICE))
        bb[m] = torch.where(torch.rand(c,device=DEVICE)<.5, MIN_N+torch.randint(0,1001,(c,),device=DEVICE), MAX_N-1-torch.randint(0,1001,(c,),device=DEVICE))
        # Deliberate carry start at arbitrary column followed by zeros/nines.
        m = kind == 7
        c = int(m.sum())
        lengths = torch.randint(1, 8, (c,), device=DEVICE)
        powers = torch.tensor([10,100,1000,10000,100000,1000000,10000000], device=DEVICE)[lengths-1]
        aa[m] = (aa[m] // powers * powers + powers - 1).clamp(MIN_N, MAX_N-1)
        bb[m] = (bb[m] // powers * powers + 1).clamp(MIN_N, MAX_N-1)
        a[:k], b[:k] = aa, bb
    ad, bd = digits8(a), digits8(b)
    tokens = torch.full((n, 25), 10, dtype=torch.long, device=DEVICE)
    tokens[:, 0:16:2] = ad
    tokens[:, 1:16:2] = bd
    s = a + b
    target = torch.stack([(s // (10 ** i)) % 10 for i in range(9)], 1).long()
    return tokens, target


@torch.no_grad()
def validate(model, batches=10, structured=0.0, batch=10000):
    model.eval()
    exact = total = 0
    worst_margin = 100.0
    for _ in range(batches):
        x, y = make_batch(batch, structured)
        logits = model(x)
        pred = logits.argmax(-1)
        exact += (pred == y).all(1).sum().item()
        total += batch
        true = logits.gather(-1, y[...,None]).squeeze(-1)
        other = logits.masked_fill(F.one_hot(y,10).bool(), -1e9).amax(-1)
        worst_margin = min(worst_margin, (true-other).amin().item())
    model.train()
    return exact / total, worst_margin


def train(model, steps, lr, structured, label, weight_decay=0.01):
    model.to(DEVICE).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9,0.98), weight_decay=weight_decay)
    warmup = min(1000, steps // 10)
    start = time.time()
    for step in range(1, steps+1):
        x, y = make_batch(BATCH, structured)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.flatten(0,1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if step <= warmup:
            rate = lr * step / warmup
        else:
            q = (step-warmup) / max(1, steps-warmup)
            rate = lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi*q)))
        for g in opt.param_groups: g["lr"] = rate
        opt.step()
        if step % 500 == 0 or step == steps:
            acc, margin = validate(model, 2, 0.0, 10000)
            edge, em = validate(model, 2, 0.75, 10000)
            print(f"{label} {step}/{steps} loss={loss.item():.5f} random={acc:.5f} edge={edge:.5f} margin={margin:.3f}/{em:.3f} sec={time.time()-start:.0f}", flush=True)
            torch.save(model.state_dict(), "/workspace/latest.pt")
    return model


if __name__ == "__main__":
    torch.manual_seed(int(os.environ.get("SEED", "1701")))
    model = Model().to(DEVICE)
    print("parameters", sum(p.numel() for p in model.parameters()), flush=True)
    train(model, int(os.environ.get("STEPS", "28000")), 2e-3, 0.35, "base")
    train(model, 5000, 8e-5, 0.55, "fine")
    train(model, 3000, 2e-5, 0.70, "edge", weight_decay=0.0)
    torch.save(model.state_dict(), "/workspace/final.pt")
    print("FINAL", validate(model, 20, 0.0), validate(model, 20, 0.75), flush=True)
