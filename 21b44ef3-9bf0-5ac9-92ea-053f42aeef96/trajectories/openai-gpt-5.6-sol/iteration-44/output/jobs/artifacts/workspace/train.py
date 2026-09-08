import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from submission import AdditionTransformer

DEVICE = "cuda"
BATCH = 8192
LOW = 10_000_000
HIGH = 99_999_999
BASE = 100_000_000


def digits8(x):
    places = torch.tensor([1, 10, 100, 1000, 10000, 100000, 1000000, 10000000], device=x.device)
    return (x[:, None] // places) % 10


def make_batch(n, structured=0.18):
    a = torch.randint(LOW, HIGH + 1, (n,), device=DEVICE)
    b = torch.randint(LOW, HIGH + 1, (n,), device=DEVICE)
    m = int(n * structured)
    if m:
        kind = torch.randint(0, 8, (m,), device=DEVICE)
        aa = torch.randint(LOW, HIGH + 1, (m,), device=DEVICE)
        bb = torch.randint(LOW, HIGH + 1, (m,), device=DEVICE)
        # Complements and near-complements around 10^8 and other decimal boundaries.
        sel = kind == 0
        delta = torch.randint(-9, 10, (m,), device=DEVICE)
        bb[sel] = (BASE - aa + delta).clamp(LOW, HIGH)[sel]
        sel = kind == 1
        powers = torch.tensor([20_000_000, 30_000_000, 40_000_000, 50_000_000, 60_000_000,
                               70_000_000, 80_000_000, 90_000_000, 100_000_000,
                               110_000_000, 120_000_000, 150_000_000, 190_000_000], device=DEVICE)
        boundary = powers[torch.randint(0, len(powers), (m,), device=DEVICE)]
        delta = torch.randint(-100, 101, (m,), device=DEVICE)
        bb[sel] = (boundary - aa + delta).clamp(LOW, HIGH)[sel]
        # Long asymmetric zero/nine suffixes.
        lengths = torch.randint(1, 8, (m,), device=DEVICE)
        p10 = (10 ** lengths)
        sel = kind == 2
        aa9 = (aa // p10) * p10 + p10 - 1
        bbsmall = (bb // p10) * p10 + torch.randint(1, 10, (m,), device=DEVICE)
        aa[sel], bb[sel] = aa9[sel].clamp(LOW, HIGH), bbsmall[sel].clamp(LOW, HIGH)
        sel = kind == 3
        bb9 = (bb // p10) * p10 + p10 - 1
        aasmall = (aa // p10) * p10 + torch.randint(1, 10, (m,), device=DEVICE)
        aa[sel], bb[sel] = aasmall[sel].clamp(LOW, HIGH), bb9[sel].clamp(LOW, HIGH)
        # Rounded, repeated, sparse, and extremes.
        sel = kind == 4
        roundp = 10 ** torch.randint(1, 8, (m,), device=DEVICE)
        aa[sel] = ((aa // roundp) * roundp).clamp(LOW, HIGH)[sel]
        sel = kind == 5
        rep = torch.tensor([11_111_111,22_222_222,33_333_333,44_444_444,55_555_555,
                            66_666_666,77_777_777,88_888_888,99_999_999], device=DEVICE)
        aa[sel] = rep[torch.randint(0, len(rep), (m,), device=DEVICE)][sel]
        sel = kind == 6
        sparse = torch.tensor([10_000_000,10_000_001,10_000_009,10_000_099,10_000_999,
                               10_009_999,10_099_999,10_999_999,20_000_000,40_000_009,
                               49_999_999,50_000_000,89_999_999,90_000_000,99_000_001,
                               99_899_999,99_999_900,99_999_990,99_999_999], device=DEVICE)
        aa[sel] = sparse[torch.randint(0, len(sparse), (m,), device=DEVICE)][sel]
        sel = kind == 7
        bb[sel] = sparse[torch.randint(0, len(sparse), (m,), device=DEVICE)][sel]
        a[:m], b[:m] = aa, bb
    ad, bd = digits8(a), digits8(b)
    operands = torch.stack((ad, bd), dim=2).reshape(n, 16)
    total = a + b
    places9 = torch.tensor([1,10,100,1000,10000,100000,1000000,10000000,100000000], device=DEVICE)
    target = (total[:, None] // places9) % 10
    inp = torch.cat((operands, torch.full((n, 1), 10, device=DEVICE), target[:, :8]), dim=1).long()
    return inp, target.long(), a, b


@torch.no_grad()
def evaluate(model, n=100000, structured=0.0, batch=10000):
    model.eval()
    good = count = 0
    min_margin = 1e9
    while count < n:
        k = min(batch, n-count)
        inp, target, _, _ = make_batch(k, structured)
        seq = inp[:, :17]
        pred = []
        margins = []
        for _ in range(9):
            logits = model(seq)[:, -1]
            top = logits.topk(2, dim=-1)
            pred.append(top.indices[:, 0])
            margins.append(top.values[:, 0] - top.values[:, 1])
            seq = torch.cat((seq, top.indices[:, :1]), 1)
        pred = torch.stack(pred, 1)
        good += (pred == target).all(1).sum().item()
        min_margin = min(min_margin, torch.stack(margins, 1).min().item())
        count += k
    model.train()
    return good, n, min_margin


def train_phase(model, optimizer, steps, lr, structured, label, checkpoint):
    for group in optimizer.param_groups:
        group["lr"] = lr
    model.train()
    start = time.time()
    for step in range(1, steps + 1):
        inp, target, _, _ = make_batch(BATCH, structured)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inp)[:, 16:25]
        loss = F.cross_entropy(logits.reshape(-1, 10), target.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 1000 == 0 or step == steps:
            elapsed = time.time() - start
            print(f"{label} {step}/{steps} loss={loss.item():.6f} sec={elapsed:.1f}", flush=True)
        if step % 6000 == 0 or step == steps:
            score = evaluate(model, 20000, structured=0.0)
            edge = evaluate(model, 20000, structured=0.8)
            print(f"  eval random={score} structured={edge}", flush=True)
            torch.save({"state": model.state_dict(),
                        "phase": label, "step": step, "random": score, "structured": edge}, checkpoint)


def prune(model, width):
    old = model.ff_in.out_features
    assert width == old - 1
    importance = model.ff_in.weight.norm(dim=1) * model.ff_out.weight.norm(dim=0)
    keep = [i for i in range(old) if i != int(importance.argmin())]
    new = AdditionTransformer(width).to(DEVICE)
    state = model.state_dict()
    new_state = new.state_dict()
    for name in new_state:
        if name == "ff_in.weight":
            new_state[name].copy_(state[name][keep])
        elif name == "ff_out.weight":
            new_state[name].copy_(state[name][:, keep])
        else:
            new_state[name].copy_(state[name])
    return new


def main():
    torch.manual_seed(2025)
    random.seed(2025)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = AdditionTransformer(4).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01, fused=True)
    train_phase(model, optimizer, 36000, 2e-3, 0.18, "teacher", "/workspace/teacher.pt")
    train_phase(model, optimizer, 6000, 5e-5, 0.25, "stabilize", "/workspace/teacher_stable.pt")
    print("teacher large", evaluate(model, 200000, 0.0), evaluate(model, 200000, 0.8), flush=True)
    model = prune(model, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.0, fused=True)
    train_phase(model, optimizer, 18000, 2e-5, 0.30, "width3", "/workspace/width3.pt")
    print("width3 large", evaluate(model, 200000, 0.0), evaluate(model, 200000, 0.8), flush=True)
    model = prune(model, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-5, weight_decay=0.0, fused=True)
    train_phase(model, optimizer, 30000, 1.2e-5, 0.35, "width2", "/workspace/width2.pt")
    train_phase(model, optimizer, 12000, 4e-6, 0.40, "polish", "/workspace/final.pt")
    print("final large", evaluate(model, 500000, 0.0), evaluate(model, 500000, 0.8), flush=True)
    torch.save({"width": 2, "state": model.state_dict()}, "/workspace/final.pt")


if __name__ == "__main__":
    main()
