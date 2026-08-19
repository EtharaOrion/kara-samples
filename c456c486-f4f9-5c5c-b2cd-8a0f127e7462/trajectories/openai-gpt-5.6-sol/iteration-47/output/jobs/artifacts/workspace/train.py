import math
import random
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

DEVICE = "cuda"
WIDTH = 9
HEADS = 3
FF = 3
LENGTH = 15
ROUNDS = 7
LIMIT = 100_000_000_000_000


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.LayerNorm(WIDTH)
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH)
        self.proj = nn.Linear(WIDTH, WIDTH)
        self.n2 = nn.LayerNorm(WIDTH)
        self.ff1 = nn.Linear(WIDTH, FF)
        self.ff2 = nn.Linear(FF, WIDTH)

    def forward(self, x, mask):
        z = self.n1(x)
        q, k, v = self.qkv(z).chunk(3, -1)
        shape = (x.shape[0], LENGTH, HEADS, WIDTH // HEADS)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        z = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        z = z.transpose(1, 2).reshape_as(x)
        x = x + self.proj(z)
        z = self.n2(x)
        return x + self.ff2(F.gelu(self.ff1(z)))


class AdditionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.a_emb = nn.Embedding(10, WIDTH)
        self.b_emb = nn.Embedding(10, WIDTH)
        self.pos = nn.Parameter(torch.randn(LENGTH, 2) * 0.02)
        self.blocks = nn.ModuleList([Block(), Block()])
        self.final = nn.LayerNorm(WIDTH)
        self.head = nn.Linear(WIDTH, 10)
        self.register_buffer("causal", torch.tril(torch.ones(LENGTH, LENGTH, dtype=torch.bool)), persistent=False)

    def forward(self, a, b):
        p = F.pad(self.pos, (0, WIDTH - 2))
        x = self.a_emb(a) + self.b_emb(b) + p
        for _ in range(ROUNDS):
            x = self.blocks[0](x, self.causal)
            x = self.blocks[1](x, self.causal)
        return self.head(self.final(x))


POW10 = torch.tensor([10**i for i in range(15)], device=DEVICE, dtype=torch.long)


def targets(a, b):
    av = (a * POW10).sum(1)
    bv = (b * POW10).sum(1)
    total = av + bv
    return (total[:, None] // POW10[None, :]) % 10


def uniform_batch(n):
    a = torch.randint(0, 10, (n, LENGTH), device=DEVICE)
    b = torch.randint(0, 10, (n, LENGTH), device=DEVICE)
    a[:, 14] = 0
    b[:, 14] = 0
    return a, b


def structured_batch(n):
    a, b = uniform_batch(n)
    rows = torch.arange(n, device=DEVICE)
    kind = rows % 6
    positions = torch.arange(LENGTH, device=DEVICE)[None, :]

    # Propagating carry: a[start]+b[start]>=10, followed by a+b=9.
    take = kind == 0
    r = rows[take]
    m = r.numel()
    start = torch.randint(0, 14, (m,), device=DEVICE)
    end = start + torch.randint(1, 15, (m,), device=DEVICE)
    end.clamp_(max=14)
    x = torch.randint(1, 10, (m,), device=DEVICE)
    y = 10 - x + torch.randint(0, 10, (m,), device=DEVICE) % x
    a[r, start] = x
    b[r, start] = y
    active = (positions < end[:, None]) & (positions > start[:, None])
    vals = torch.randint(0, 10, (m, LENGTH), device=DEVICE)
    a[r] = torch.where(active, vals, a[r])
    b[r] = torch.where(active, 9 - vals, b[r])

    # Matched long non-carry runs of nines.
    take = kind == 1
    r = rows[take]
    m = r.numel()
    start = torch.randint(0, 14, (m,), device=DEVICE)
    end = start + torch.randint(1, 15, (m,), device=DEVICE)
    end.clamp_(max=14)
    active = (positions < end[:, None]) & (positions >= start[:, None])
    vals = torch.randint(0, 10, (m, LENGTH), device=DEVICE)
    a[r] = torch.where(active, vals, a[r])
    b[r] = torch.where(active, 9 - vals, b[r])
    if m:
        low = positions < start[:, None]
        a[r] = torch.where(low, torch.zeros_like(a[r]), a[r])
        b[r] = torch.where(low, torch.zeros_like(b[r]), b[r])

    # Sparse shifted all-nine chains plus one.
    take = kind == 2
    r = rows[take]
    m = r.numel()
    a[r] = 0
    b[r] = 0
    start = torch.randint(0, 14, (m,), device=DEVICE)
    end = start + torch.randint(1, 15, (m,), device=DEVICE)
    end.clamp_(max=14)
    active = (positions < end[:, None]) & (positions >= start[:, None])
    a[r] = torch.where(active, torch.full_like(a[r], 9), a[r])
    b[r, start] = 1

    # Isolated columns, deliberately emphasizing equality boundaries.
    take = kind == 3
    r = rows[take]
    m = r.numel()
    a[r] = 0
    b[r] = 0
    col = torch.randint(0, 14, (m,), device=DEVICE)
    choices = torch.randint(0, 3, (m,), device=DEVICE)
    x = torch.where(choices == 0, 5, torch.where(choices == 1, 9, torch.randint(0, 10, (m,), device=DEVICE)))
    y = torch.where(choices == 0, 5, torch.where(choices == 1, 9, torch.randint(0, 10, (m,), device=DEVICE)))
    a[r, col] = x
    b[r, col] = y

    # Repeated and blockwise digits.
    take = kind == 4
    r = rows[take]
    m = r.numel()
    da = torch.randint(0, 10, (m, 1), device=DEVICE)
    db = torch.randint(0, 10, (m, 1), device=DEVICE)
    a[r, :14] = da
    b[r, :14] = db

    # Complementary random columns with random incoming-carry triggers.
    take = kind == 5
    r = rows[take]
    m = r.numel()
    vals = torch.randint(0, 10, (m, 14), device=DEVICE)
    a[r, :14] = vals
    b[r, :14] = 9 - vals
    trigger = torch.randint(0, 14, (m,), device=DEVICE)
    a[r, trigger] = torch.clamp(a[r, trigger] + 1, max=9)

    a[:, 14] = 0
    b[:, 14] = 0
    return a, b


@torch.no_grad()
def evaluate(model, batches=16, batch=8192, structured=False):
    model.eval()
    errors = 0
    digits = 0
    for _ in range(batches):
        a, b = structured_batch(batch) if structured else uniform_batch(batch)
        y = targets(a, b)
        pred = model(a, b).argmax(-1)
        errors += (pred != y).any(1).sum().item()
        digits += (pred != y).sum().item()
    model.train()
    return errors, batches * batch, digits


@torch.no_grad()
def systematic(model):
    pairs = [(0, 0), (LIMIT - 1, 0), (LIMIT - 1, 1), (LIMIT - 1, LIMIT - 1)]
    for start in range(14):
        p = 10**start
        for length in range(1, 15 - start):
            run = (10**length - 1) * p
            pairs.extend([(run, p), (run, 0), (run - p, p), (5 * p, 5 * p), (9 * p, 9 * p)])
    av = torch.tensor([x for x, _ in pairs], device=DEVICE)
    bv = torch.tensor([y for _, y in pairs], device=DEVICE)
    a = (av[:, None] // POW10) % 10
    b = (bv[:, None] // POW10) % 10
    pred = model(a, b).argmax(-1)
    err = (pred != targets(a, b)).any(1)
    return int(err.sum()), len(pairs), [pairs[i] for i in err.nonzero().flatten().tolist()[:10]]


def train():
    torch.manual_seed(47)
    model = AdditionTransformer().to(DEVICE)
    print("parameters", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.003, fused=True)
    schedule = [(10000, 3e-3, 0.0), (14000, 1e-3, 0.35), (16000, 3e-4, 0.5), (16000, 1e-4, 0.5), (16000, 3e-5, 0.5)]
    step = 0
    started = time.time()
    for count, lr, structured_fraction in schedule:
        for group in opt.param_groups:
            group["lr"] = lr
        for _ in range(count):
            step += 1
            n = 4096
            if structured_fraction == 0:
                a, b = uniform_batch(n)
            else:
                ns = int(n * structured_fraction)
                au, bu = uniform_batch(n - ns)
                ast, bst = structured_batch(ns)
                a = torch.cat((au, ast))
                b = torch.cat((bu, bst))
            y = targets(a, b)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(a, b).reshape(-1, 10), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            if step % 2000 == 0:
                e, total, d = evaluate(model, 4)
                es, totals, ds = evaluate(model, 4, structured=True)
                sy, sn, examples = systematic(model)
                print(f"step={step} lr={lr:g} loss={loss.item():.6g} uniform={e}/{total} structured={es}/{totals} digit={d+ds} systematic={sy}/{sn} time={time.time()-started:.0f}s {examples[:2]}", flush=True)
                torch.save(model.state_dict(), "/workspace/latest.pt")
    torch.save(model.state_dict(), "/workspace/model.pt")
    e, total, d = evaluate(model, 128)
    es, totals, ds = evaluate(model, 128, structured=True)
    print("FINAL", e, total, es, totals, d, ds, systematic(model), flush=True)


if __name__ == "__main__":
    train()
