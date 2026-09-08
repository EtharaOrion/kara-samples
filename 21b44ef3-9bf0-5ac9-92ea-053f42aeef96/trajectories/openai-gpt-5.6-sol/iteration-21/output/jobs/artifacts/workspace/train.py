import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace")
from submission import AdditionTransformer

DEVICE = "cuda"
BATCH = 8192
STEPS = 42000
LOW = 10_000_000
HIGH = 99_999_999
POW10 = torch.tensor([10 ** i for i in range(9)], device=DEVICE, dtype=torch.long)


def digits(x, count=9):
    return (x[:, None] // POW10[None, :count]) % 10


def uniform(n):
    return (torch.randint(LOW, HIGH + 1, (n,), device=DEVICE),
            torch.randint(LOW, HIGH + 1, (n,), device=DEVICE))


def structured(n):
    a, b = uniform(n)
    kind = torch.randint(0, 8, (n,), device=DEVICE)

    # Exact and near complements to 100,000,000, covering every leading digit.
    m = kind == 0
    c = int(m.sum())
    if c:
        aa = torch.randint(LOW, 90_000_001, (c,), device=DEVICE)
        delta = torch.randint(-20, 21, (c,), device=DEVICE)
        bb = 100_000_000 + delta - aa
        valid = (bb >= LOW) & (bb <= HIGH)
        bb = torch.where(valid, bb, 100_000_000 - aa)
        a[m], b[m] = aa, bb

    # Force complementary suffixes, producing carries of every possible length.
    m = kind == 1
    c = int(m.sum())
    if c:
        k = torch.randint(1, 8, (c,), device=DEVICE)
        p = 10 ** k
        x = torch.floor(torch.rand(c, device=DEVICE) * (p - 1)).long() + 1
        ap = torch.randint(1, 10, (c,), device=DEVICE) * 10_000_000
        bp = torch.randint(1, 10, (c,), device=DEVICE) * 10_000_000
        aa = (ap + torch.randint(0, 10_000_000, (c,), device=DEVICE)) // p * p + x
        bb = (bp + torch.randint(0, 10_000_000, (c,), device=DEVICE)) // p * p + (p - x)
        a[m], b[m] = aa.clamp(LOW, HIGH), bb.clamp(LOW, HIGH)

    # Long asymmetric runs of terminal nines and zeros.
    m = kind == 2
    c = int(m.sum())
    if c:
        k = torch.randint(1, 8, (c,), device=DEVICE)
        p = 10 ** k
        base_a = torch.randint(LOW, HIGH + 1, (c,), device=DEVICE)
        base_b = torch.randint(LOW, HIGH + 1, (c,), device=DEVICE)
        aa = base_a // p * p + (p - 1)
        suffix = torch.randint(0, 3, (c,), device=DEVICE)
        bb = base_b // p * p + suffix
        swap = torch.rand(c, device=DEVICE) < .5
        a[m], b[m] = torch.where(swap, bb, aa), torch.where(swap, aa, bb)

    # Decimal round numbers with small signed perturbations.
    m = kind == 3
    c = int(m.sum())
    if c:
        k = torch.randint(1, 8, (c,), device=DEVICE)
        p = 10 ** k
        aa = torch.randint(LOW, HIGH + 1, (c,), device=DEVICE) // p * p
        bb = torch.randint(LOW, HIGH + 1, (c,), device=DEVICE) // p * p
        da = torch.randint(-2, 3, (c,), device=DEVICE)
        db = torch.randint(-2, 3, (c,), device=DEVICE)
        a[m], b[m] = (aa + da).clamp(LOW, HIGH), (bb + db).clamp(LOW, HIGH)

    # Repeated-digit operands.
    m = kind == 4
    c = int(m.sum())
    if c:
        d1 = torch.randint(1, 10, (c,), device=DEVICE)
        d2 = torch.randint(1, 10, (c,), device=DEVICE)
        a[m], b[m] = d1 * 11_111_111, d2 * 11_111_111

    # Sparse operands, independently selecting every digit after a nonzero lead.
    m = kind == 5
    c = int(m.sum())
    if c:
        lead1 = torch.randint(1, 10, (c,), device=DEVICE) * 10_000_000
        lead2 = torch.randint(1, 10, (c,), device=DEVICE) * 10_000_000
        pos1 = torch.randint(0, 7, (c,), device=DEVICE)
        pos2 = torch.randint(0, 7, (c,), device=DEVICE)
        val1 = torch.randint(0, 10, (c,), device=DEVICE)
        val2 = torch.randint(0, 10, (c,), device=DEVICE)
        a[m] = lead1 + val1 * (10 ** pos1)
        b[m] = lead2 + val2 * (10 ** pos2)

    # Near extrema and combinations of all-zero/all-nine suffixes.
    m = kind == 6
    c = int(m.sum())
    if c:
        ends = torch.tensor([LOW, LOW + 1, LOW + 9, LOW + 99, HIGH,
                             HIGH - 1, HIGH - 9, HIGH - 99], device=DEVICE)
        a[m] = ends[torch.randint(0, len(ends), (c,), device=DEVICE)]
        b[m] = torch.randint(LOW, HIGH + 1, (c,), device=DEVICE)

    return a, b


def batch(step):
    fraction = .28 if step < 30000 else .52
    ns = int(BATCH * fraction)
    au, bu = uniform(BATCH - ns)
    ast, bst = structured(ns)
    a = torch.cat((au, ast))
    b = torch.cat((bu, bst))
    order = torch.randperm(BATCH, device=DEVICE)
    return a[order], b[order]


def encode(a, b):
    ad, bd = digits(a, 8), digits(b, 8)
    result = digits(a + b, 9)
    x = torch.empty((len(a), 25), dtype=torch.long, device=DEVICE)
    x[:, 0:16:2] = ad
    x[:, 1:16:2] = bd
    x[:, 16] = 10
    x[:, 17:] = result[:, :-1]
    return x, result


@torch.no_grad()
def greedy_accuracy(model, a, b, chunk=16384):
    good = 0
    min_margin = 100.0
    for start in range(0, len(a), chunk):
        aa, bb = a[start:start + chunk], b[start:start + chunk]
        ad, bd = digits(aa, 8), digits(bb, 8)
        seq = torch.empty((len(aa), 17), dtype=torch.long, device=DEVICE)
        seq[:, 0:16:2], seq[:, 1:16:2], seq[:, 16] = ad, bd, 10
        predictions = []
        for _ in range(9):
            logits = model(seq)[:, -1]
            top = logits.topk(2, dim=-1)
            pred = top.indices[:, 0]
            min_margin = min(min_margin, float((top.values[:, 0] - top.values[:, 1]).min()))
            predictions.append(pred)
            seq = torch.cat((seq, pred[:, None]), 1)
        pred_digits = torch.stack(predictions, 1)
        good += int((pred_digits == digits(aa + bb, 9)).all(1).sum())
    return good, len(a), min_margin


def validation_sets(n=100000):
    random_a, random_b = uniform(n)
    edge_a, edge_b = structured(n)
    # Deterministic cross-product emphasizes verifier-style corners.
    vals = set([LOW, LOW + 1, LOW + 9, LOW + 10, LOW + 99, 20_000_000,
                40_000_009, 49_999_999, 50_000_000, 50_000_001, 89_999_999,
                90_000_000, 99_000_000, 99_900_000, 99_990_000, 99_999_000,
                99_999_900, 99_999_990, HIGH])
    for lead in range(1, 10):
        for k in range(1, 8):
            p = 10 ** k
            vals.add(lead * 10_000_000)
            vals.add(min(HIGH, lead * 10_000_000 + p - 1))
    vals = sorted(x for x in vals if LOW <= x <= HIGH)
    ca = torch.tensor([x for x in vals for _ in vals], device=DEVICE)
    cb = torch.tensor(vals * len(vals), device=DEVICE)
    return (random_a, random_b), (edge_a, edge_b), (ca, cb)


def save(model, path):
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)


def main():
    torch.manual_seed(19)
    random.seed(19)
    torch.set_float32_matmul_precision("high")
    model = AdditionTransformer().to(DEVICE)
    print("parameters", sum(p.numel() for p in model.parameters()), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(.9, .98), weight_decay=.01)
    scaler = torch.amp.GradScaler("cuda")
    best = -1
    started = time.time()
    for step in range(1, STEPS + 1):
        a, b = batch(step)
        x, target = encode(a, b)
        if step <= 1000:
            lr = 2e-3 * step / 1000
        elif step <= 32000:
            progress = (step - 1000) / 31000
            lr = 2e-3 * (.08 + .92 * .5 * (1 + math.cos(math.pi * progress)))
        elif step <= 38000:
            lr = 5e-5
        else:
            lr = 2e-5
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)[:, 16:25]
            loss = F.cross_entropy(logits.reshape(-1, 10), target.reshape(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step % 500 == 0:
            elapsed = time.time() - started
            print(f"step {step} loss {loss.item():.6f} lr {lr:.2g} time {elapsed:.1f}", flush=True)
        if step % 4000 == 0 and step >= 24000:
            va, ve, vc = validation_sets(30000)
            scores = []
            for name, pair in zip(("random", "structured", "corners"), (va, ve, vc)):
                score = greedy_accuracy(model, *pair)
                scores.append(score[0] / score[1])
                print(name, score, flush=True)
            criterion = min(scores)
            save(model, f"/workspace/checkpoint_{step}.pt")
            if criterion > best:
                best = criterion
                save(model, "/workspace/best.pt")
                print("new best", best, flush=True)
    save(model, "/workspace/final.pt")


if __name__ == "__main__":
    main()
