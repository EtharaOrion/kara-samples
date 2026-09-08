import math
import random
import time

import torch
import torch.nn.functional as F

from submission import AdderTransformer

DEVICE = "cuda"
BATCH = 8192
LOW = 10_000_000
HIGH = 99_999_999
POWERS = torch.tensor([10 ** i for i in range(8)], device=DEVICE, dtype=torch.long)


def structured_pair(n):
    a = torch.randint(LOW, HIGH + 1, (n,), device=DEVICE)
    b = torch.randint(LOW, HIGH + 1, (n,), device=DEVICE)
    mode = torch.randint(0, 7, (n,), device=DEVICE)

    take = mode == 0
    count = int(take.sum())
    if count:
        x = torch.randint(LOW, 90_000_001, (count,), device=DEVICE)
        a[take], b[take] = x, 100_000_000 - x

    take = mode == 1
    count = int(take.sum())
    if count:
        k = torch.randint(1, 8, (count,), device=DEVICE)
        p = 10 ** k
        prefix = torch.randint(1, 10, (count,), device=DEVICE)
        x = prefix * 10_000_000 + (p - 1)
        x = torch.clamp(x, LOW, HIGH)
        y = torch.randint(LOW, HIGH + 1, (count,), device=DEVICE)
        a[take], b[take] = x, y

    take = mode == 2
    count = int(take.sum())
    if count:
        k = torch.randint(1, 8, (count,), device=DEVICE)
        p = 10 ** k
        x = torch.randint(LOW, HIGH + 1, (count,), device=DEVICE)
        x = torch.clamp((x // p) * p, LOW, HIGH)
        y = torch.randint(LOW, HIGH + 1, (count,), device=DEVICE)
        a[take], b[take] = x, y

    take = mode == 3
    count = int(take.sum())
    if count:
        d1 = torch.randint(1, 10, (count,), device=DEVICE)
        d2 = torch.randint(1, 10, (count,), device=DEVICE)
        a[take], b[take] = d1 * 11_111_111, d2 * 11_111_111

    take = mode == 4
    count = int(take.sum())
    if count:
        k = torch.randint(1, 8, (count,), device=DEVICE)
        p = 10 ** k
        lead = torch.randint(1, 10, (count,), device=DEVICE)
        x = lead * 10_000_000 + torch.randint(0, 10, (count,), device=DEVICE) * (p - 1) // 9
        y = torch.randint(LOW, HIGH + 1, (count,), device=DEVICE)
        a[take], b[take] = torch.clamp(x, LOW, HIGH), y

    take = mode == 5
    count = int(take.sum())
    if count:
        x = torch.randint(LOW, HIGH + 1, (count,), device=DEVICE)
        choices = torch.tensor([LOW, LOW + 1, 90_000_000, 99_000_000, HIGH - 1, HIGH], device=DEVICE)
        y = choices[torch.randint(0, len(choices), (count,), device=DEVICE)]
        a[take], b[take] = x, y

    take = mode == 6
    count = int(take.sum())
    if count:
        target = torch.randint(20_000_000, 190_000_001, (count,), device=DEVICE)
        lo = torch.maximum(torch.full_like(target, LOW), target - HIGH)
        hi = torch.minimum(torch.full_like(target, HIGH), target - LOW)
        valid = hi >= lo
        span = (hi - lo + 1).clamp_min(1)
        x = lo + (torch.rand(count, device=DEVICE) * span.float()).long()
        y = target - x
        fallback = torch.randint(LOW, HIGH + 1, (count,), device=DEVICE)
        x = torch.where(valid, x, fallback)
        y = torch.where(valid, y, fallback)
        a[take], b[take] = x, y
    return a, b


def batch_data(n, structured_fraction):
    a = torch.randint(LOW, HIGH + 1, (n,), device=DEVICE)
    b = torch.randint(LOW, HIGH + 1, (n,), device=DEVICE)
    count = int(n * structured_fraction)
    if count:
        sa, sb = structured_pair(count)
        a[:count], b[:count] = sa, sb
    ad = (a[:, None] // POWERS) % 10
    bd = (b[:, None] // POWERS) % 10
    operands = torch.stack((ad, bd), dim=2).reshape(n, 16)
    sums = a + b
    targets = (sums[:, None] // torch.tensor([10 ** i for i in range(9)], device=DEVICE)) % 10
    sentinel = torch.full((n, 1), 10, device=DEVICE, dtype=torch.long)
    inputs = torch.cat((operands, sentinel, targets[:, :8]), dim=1)
    return inputs, targets


@torch.no_grad()
def teacher_accuracy(model, n=100_000, structured=0.0):
    model.eval()
    correct = 0
    digits = 0
    for _ in range((n + BATCH - 1) // BATCH):
        size = min(BATCH, n - digits)
        x, y = batch_data(size, structured)
        pred = model(x)[:, 16:25].argmax(-1)
        correct += int((pred == y).all(1).sum())
        digits += size
    model.train()
    return correct / n


@torch.no_grad()
def autoregressive_accuracy(model, n=20_000, structured=0.0):
    model.eval()
    correct = 0
    seen = 0
    chunk = 2048
    while seen < n:
        size = min(chunk, n - seen)
        x, y = batch_data(size, structured)
        seq = x[:, :17]
        outputs = []
        for _ in range(9):
            digit = model(seq)[:, -1].argmax(-1)
            outputs.append(digit)
            seq = torch.cat((seq, digit[:, None]), dim=1)
        pred = torch.stack(outputs, 1)
        correct += int((pred == y).all(1).sum())
        seen += size
    model.train()
    return correct / n


def main():
    torch.manual_seed(1801)
    random.seed(1801)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    model = AdderTransformer().to(DEVICE)
    print("parameters", sum(p.numel() for p in model.parameters()), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(0.9, 0.98), weight_decay=0.01)
    start = time.time()
    total = 32000
    for step in range(1, total + 1):
        if step <= 25000:
            warm = min(1.0, step / 500)
            cosine = 0.15 + 0.85 * (1 + math.cos(math.pi * step / 25000)) / 2
            lr = 2e-3 * warm * cosine
            fraction = 0.25
        else:
            lr = 5e-5 if step <= 29000 else 2e-5
            fraction = 0.55
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = batch_data(BATCH, fraction)
        logits = model(x)[:, 16:25]
        loss = F.cross_entropy(logits.reshape(-1, 10), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 1000 == 0 or step == 1:
            uniform = teacher_accuracy(model, 20_000, 0.0)
            edge = teacher_accuracy(model, 20_000, 1.0)
            print(step, f"loss={loss.item():.6f}", f"lr={lr:.2g}", f"teacher={uniform:.5f}/{edge:.5f}", f"seconds={time.time()-start:.1f}", flush=True)
            torch.save(model.state_dict(), "/workspace/checkpoint.pt")
    uniform = autoregressive_accuracy(model, 100_000, 0.0)
    edge = autoregressive_accuracy(model, 100_000, 1.0)
    print(f"FINAL autoregressive={uniform:.6f}/{edge:.6f}", flush=True)
    torch.save(model.state_dict(), "/workspace/checkpoint.pt")


if __name__ == "__main__":
    main()
