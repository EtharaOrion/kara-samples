import math
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn

from submission import AdditionTransformer

DEVICE = "cuda"
BATCH = 8192
LOW = 10_000_000
HIGH = 99_999_999


def digits8(x):
    places = torch.tensor([1, 10, 100, 1000, 10000, 100000, 1000000, 10000000], device=x.device)
    return (x[:, None] // places % 10).long()


def make_batch(n, structured=0.30):
    a = torch.randint(LOW, HIGH + 1, (n,), device=DEVICE)
    b = torch.randint(LOW, HIGH + 1, (n,), device=DEVICE)
    count = int(n * structured)
    if count:
        kind = torch.randint(0, 8, (count,), device=DEVICE)
        sa = torch.randint(LOW, HIGH + 1, (count,), device=DEVICE)
        sb = torch.randint(LOW, HIGH + 1, (count,), device=DEVICE)

        # Exact and near complements exercise full carry chains.
        idx = kind <= 1
        target = torch.where(kind[idx] == 0, 100_000_000, 110_000_000)
        sb[idx] = (target - sa[idx]).clamp(LOW, HIGH)

        # Asymmetric suffixes of zeroes/nines at every possible length.
        idx = kind == 2
        m = int(idx.sum())
        if m:
            p = 10 ** torch.randint(1, 8, (m,), device=DEVICE)
            sa[idx] = (sa[idx] // p) * p + (p - 1)
            sb[idx] = (sb[idx] // p) * p + torch.randint(0, 2, (m,), device=DEVICE)

        idx = kind == 3
        m = int(idx.sum())
        if m:
            p = 10 ** torch.randint(1, 8, (m,), device=DEVICE)
            sa[idx] = (sa[idx] // p) * p
            sb[idx] = (sb[idx] // p) * p + (p - 1)

        # Numbers close to decimal boundaries and extrema.
        idx = kind == 4
        m = int(idx.sum())
        if m:
            p = 10 ** torch.randint(1, 9, (m,), device=DEVICE)
            off = torch.randint(-12, 13, (m,), device=DEVICE)
            sa[idx] = ((sa[idx] // p) * p + off).clamp(LOW, HIGH)

        idx = kind == 5
        m = int(idx.sum())
        if m:
            edge = torch.where(torch.rand(m, device=DEVICE) < .5, LOW, HIGH)
            sa[idx] = (edge + torch.randint(-1000, 1001, (m,), device=DEVICE)).clamp(LOW, HIGH)

        # Repeated and sparse digits.
        idx = kind == 6
        m = int(idx.sum())
        if m:
            d = torch.randint(1, 10, (m,), device=DEVICE)
            sa[idx] = d * 11_111_111

        idx = kind == 7
        m = int(idx.sum())
        if m:
            lead = torch.randint(1, 10, (m,), device=DEVICE)
            pos = 10 ** torch.randint(0, 7, (m,), device=DEVICE)
            val = torch.randint(0, 10, (m,), device=DEVICE)
            sa[idx] = lead * 10_000_000 + val * pos

        a[:count], b[:count] = sa, sb

    result = a + b
    da, db = digits8(a), digits8(b)
    dr = torch.stack([(result // (10 ** i) % 10).long() for i in range(9)], dim=1)
    tokens = torch.empty(n, 25, dtype=torch.long, device=DEVICE)
    tokens[:, :16:2] = da
    tokens[:, 1:16:2] = db
    tokens[:, 16] = 10
    tokens[:, 17:] = dr[:, :8]
    return tokens, dr


@torch.no_grad()
def evaluate(model, batches=20, structured=0.0, autoregressive=False):
    model.eval()
    correct = total = 0
    for _ in range(batches):
        tokens, targets = make_batch(4096, structured)
        if autoregressive:
            seq = tokens[:, :17]
            predictions = []
            for _ in range(9):
                digit = model(seq)[:, -1].argmax(-1)
                predictions.append(digit)
                seq = torch.cat((seq, digit[:, None]), 1)
            pred = torch.stack(predictions, 1)
        else:
            pred = model(tokens)[:, 16:25].argmax(-1)
        correct += (pred == targets).all(1).sum().item()
        total += targets.shape[0]
    model.train()
    return correct, total


def train():
    torch.manual_seed(1901)
    random.seed(1901)
    torch.set_float32_matmul_precision("high")
    model = AdditionTransformer().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, betas=(0.9, 0.98), weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    start = time.time()
    steps = 36000
    for step in range(1, steps + 1):
        # Keep broad uniform coverage dominant; raise edge coverage late.
        structured = 0.30 if step < 30000 else 0.45
        tokens, targets = make_batch(BATCH, structured)
        if step < 1000:
            lr = 0.002 * step / 1000
        elif step < 30000:
            progress = (step - 1000) / 29000
            lr = 5e-5 + 0.5 * (0.002 - 5e-5) * (1 + math.cos(math.pi * progress))
        else:
            lr = 2e-5
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(tokens)[:, 16:25]
            loss = nn.functional.cross_entropy(logits.reshape(-1, 10), targets.reshape(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step % 1000 == 0 or step == 1:
            elapsed = time.time() - start
            print(f"step={step} loss={loss.item():.6f} lr={lr:.2g} time={elapsed:.1f}", flush=True)
        if step % 4000 == 0 and step >= 16000:
            u = evaluate(model, 10, 0.0)
            s = evaluate(model, 10, 1.0)
            print(f"  teacher uniform={u[0]/u[1]:.6f} structured={s[0]/s[1]:.6f}", flush=True)
            torch.save(model.state_dict(), "/workspace/checkpoint.pt")
    torch.save(model.state_dict(), "/workspace/checkpoint.pt")
    for mode in (0.0, 1.0):
        score = evaluate(model, 25, mode, autoregressive=True)
        print(f"autoregressive structured={mode}: {score[0]}/{score[1]}={score[0]/score[1]:.8f}")


if __name__ == "__main__":
    train()
