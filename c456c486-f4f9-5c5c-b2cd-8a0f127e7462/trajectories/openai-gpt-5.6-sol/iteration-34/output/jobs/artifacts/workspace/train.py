import math
import random
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

DEVICE = "cuda"
WIDTH = 9
FF = 2
DIGITS = 14
BATCH = 4096
POWERS = (10 ** torch.arange(DIGITS, device=DEVICE, dtype=torch.long))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(WIDTH)
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH)
        self.proj = nn.Linear(WIDTH, WIDTH)
        self.norm2 = nn.LayerNorm(WIDTH)
        self.fc1 = nn.Linear(WIDTH, FF)
        self.fc2 = nn.Linear(FF, WIDTH)

    def forward(self, x):
        y = self.norm1(x)
        q, k, v = self.qkv(y).chunk(3, dim=-1)
        shape = (y.shape[0], y.shape[1], 3, 3)
        q, k, v = (z.view(shape).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(x.shape[0], x.shape[1], WIDTH)
        x = x + self.proj(y)
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


class Adder(nn.Module):
    def __init__(self):
        super().__init__()
        self.a_embed = nn.Embedding(11, WIDTH)
        self.b_embed = nn.Embedding(11, WIDTH)
        self.out_embed = nn.Embedding(11, WIDTH)
        self.position = nn.Parameter(torch.empty(30, WIDTH))
        self.blocks = nn.ModuleList((Block(), Block()))
        self.norm = nn.LayerNorm(WIDTH)
        self.head = nn.Linear(WIDTH, 10)
        nn.init.normal_(self.position, std=0.02)

    def sequence(self, a, b, previous):
        x = torch.cat((self.a_embed(a) + self.b_embed(b), self.out_embed(previous)), 1)
        x = x + self.position[:x.shape[1]]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

    def forward(self, a, b, previous):
        return self.sequence(a, b, previous)[:, -1]


def digits(values, count):
    return (values[:, None] // (10 ** torch.arange(count, device=DEVICE))) % 10


def make_batch(n=BATCH, structured=True):
    a = torch.randint(0, 10, (n, DIGITS), device=DEVICE)
    b = torch.randint(0, 10, (n, DIGITS), device=DEVICE)
    if structured:
        idx = torch.arange(DIGITS, device=DEVICE)[None, :]
        # Half the batch remains uniform; the rest spans seven regenerated families.
        kind = torch.randint(0, 14, (n,), device=DEVICE)

        # Random carry chains, including chains reaching the top column.
        chosen = kind == 1
        start = torch.randint(0, DIGITS, (n,), device=DEVICE)
        length = torch.randint(1, DIGITS + 1, (n,), device=DEVICE)
        end = torch.minimum(start + length - 1, torch.full_like(start, DIGITS - 1))
        chain = chosen[:, None] & (idx >= start[:, None]) & (idx <= end[:, None])
        first = chosen[:, None] & (idx == start[:, None])
        av = torch.randint(1, 10, (n, DIGITS), device=DEVICE)
        a = torch.where(chain, av, a)
        b = torch.where(chain, 9 - av, b)
        b = torch.where(first, 10 - av, b)

        # Matched non-carry runs: long sum-nine runs with no incoming carry.
        chosen = kind == 2
        start = torch.randint(0, DIGITS, (n,), device=DEVICE)
        length = torch.randint(1, DIGITS + 1, (n,), device=DEVICE)
        end = torch.minimum(start + length - 1, torch.full_like(start, DIGITS - 1))
        run = chosen[:, None] & (idx >= start[:, None]) & (idx <= end[:, None])
        av = torch.randint(0, 10, (n, DIGITS), device=DEVICE)
        a = torch.where(run, av, a)
        b = torch.where(run, 9 - av, b)
        before = chosen[:, None] & (idx == (start - 1)[:, None]) & (start[:, None] > 0)
        a = torch.where(before, torch.zeros_like(a), a)
        b = torch.where(before, torch.zeros_like(b), b)

        # Sparse isolated boundaries, heavily covering 5+5 and 9+9 at every position.
        chosen = (kind == 3) | (kind == 4)
        a = torch.where(chosen[:, None], torch.zeros_like(a), a)
        b = torch.where(chosen[:, None], torch.zeros_like(b), b)
        pos = torch.randint(0, DIGITS, (n,), device=DEVICE)
        spot = chosen[:, None] & (idx == pos[:, None])
        val = torch.where(kind == 3, torch.full_like(kind, 5), torch.full_like(kind, 9))
        a = torch.where(spot, val[:, None], a)
        b = torch.where(spot, val[:, None], b)

        # Repeated operands and complementary columns.
        chosen = kind == 5
        ra = torch.randint(0, 10, (n,), device=DEVICE)
        rb = torch.randint(0, 10, (n,), device=DEVICE)
        a = torch.where(chosen[:, None], ra[:, None], a)
        b = torch.where(chosen[:, None], rb[:, None], b)
        chosen = kind == 6
        av = torch.randint(0, 10, (n, DIGITS), device=DEVICE)
        a = torch.where(chosen[:, None], av, a)
        b = torch.where(chosen[:, None], 9 - av, b)

        # Internal carry boundaries surrounded by random context.
        chosen = kind == 7
        pos = torch.randint(1, DIGITS, (n,), device=DEVICE)
        prev = chosen[:, None] & (idx == (pos - 1)[:, None])
        spot = chosen[:, None] & (idx == pos[:, None])
        av = torch.randint(1, 10, (n, DIGITS), device=DEVICE)
        a = torch.where(prev, av, a)
        b = torch.where(prev, 10 - av, b)
        boundary_a = torch.randint(0, 10, (n, 1), device=DEVICE)
        boundary_sum = torch.randint(8, 11, (n, 1), device=DEVICE)
        boundary_b = (boundary_sum - boundary_a).clamp(0, 9)
        a = torch.where(spot, boundary_a, a)
        b = torch.where(spot, boundary_b, b)

    av = (a.long() * POWERS).sum(1)
    bv = (b.long() * POWERS).sum(1)
    target = digits(av + bv, 15)
    source_a = torch.cat((a, torch.full((n, 1), 10, device=DEVICE, dtype=torch.long)), 1)
    source_b = torch.cat((b, torch.full((n, 1), 10, device=DEVICE, dtype=torch.long)), 1)
    previous = torch.cat((torch.full((n, 1), 10, device=DEVICE, dtype=torch.long), target[:, :-1]), 1)
    return source_a, source_b, previous, target, av, bv


def exact_accuracy(model, batches, structured=False, n=8192):
    model.eval()
    good = total = 0
    with torch.no_grad():
        for _ in range(batches):
            a, b, _, target, _, _ = make_batch(n, structured)
            previous = torch.full((n, 1), 10, device=DEVICE, dtype=torch.long)
            outputs = []
            for _ in range(15):
                prediction = model(a, b, previous).argmax(1)
                outputs.append(prediction)
                previous = torch.cat((previous, prediction[:, None]), 1)
            prediction = torch.stack(outputs, 1)
            good += (prediction == target).all(1).sum().item()
            total += n
    model.train()
    return good, total


def save(model, path):
    torch.save(model.state_dict(), path)


def main():
    torch.manual_seed(34)
    random.seed(34)
    torch.set_float32_matmul_precision("high")
    model = Adder().to(DEVICE)
    count = sum(p.numel() for p in model.parameters())
    print("parameters", count, flush=True)
    assert count == 1571
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.005)
    best = -1
    phases = [
        (12000, 3e-3, False),
        (12000, 1e-3, False),
        (12000, 3e-4, True),
        (12000, 1e-4, True),
        (16000, 5e-5, True),
        (16000, 2e-5, True),
        (16000, 1e-5, True),
    ]
    step = 0
    start_time = time.time()
    for steps, lr, structured in phases:
        for group in optimizer.param_groups:
            group["lr"] = lr
        for _ in range(steps):
            step += 1
            a, b, previous, target, _, _ = make_batch(BATCH, structured)
            logits = model.sequence(a, b, previous)[:, 15:]
            loss = F.cross_entropy(logits.reshape(-1, 10), target.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"step {step} lr {lr:g} loss {loss.item():.7f} time {elapsed:.1f}", flush=True)
            if step % 6000 == 0:
                good, total = exact_accuracy(model, 4, False)
                sg, st = exact_accuracy(model, 4, True)
                score = good * 2 + sg
                print(f"validation random {good}/{total} structured {sg}/{st}", flush=True)
                if score >= best:
                    best = score
                    save(model, "/workspace/best.pt")
                    print("saved", flush=True)
    save(model, "/workspace/final.pt")
    good, total = exact_accuracy(model, 16, False)
    sg, st = exact_accuracy(model, 16, True)
    print(f"FINAL random {good}/{total} structured {sg}/{st}", flush=True)


if __name__ == "__main__":
    main()
