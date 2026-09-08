import math
import random
import time
import torch
import torch.nn.functional as F
from submission import AdditionTransformer

DEVICE = "cuda"
BATCH = 8192
LOW = 10_000_000
HIGH = 99_999_999
BASE = 100_000_000


def digits(n, count):
    places = (10 ** torch.arange(count, device=n.device, dtype=torch.long)).view(1, -1)
    return (n.view(-1, 1) // places) % 10


def make_batch(batch, structured=0.30):
    a = torch.randint(LOW, HIGH + 1, (batch,), device=DEVICE)
    b = torch.randint(LOW, HIGH + 1, (batch,), device=DEVICE)
    n = int(batch * structured)
    if n:
        kind = torch.randint(0, 7, (n,), device=DEVICE)
        sa = torch.randint(LOW, HIGH + 1, (n,), device=DEVICE)
        sb = torch.randint(LOW, HIGH + 1, (n,), device=DEVICE)
        # Exact and near complements exercise the final carry and all carry lengths.
        m = kind == 0
        offsets = torch.randint(-20, 21, (n,), device=DEVICE)
        sb[m] = (BASE - sa[m] + offsets[m]).clamp(LOW, HIGH)
        # Asymmetric suffixes of nines against small nonzero suffixes.
        for code, power in ((1, 10), (2, 100), (3, 1000), (4, 10000)):
            m = kind == code
            if m.any():
                p = power
                sa[m] = (sa[m] // p) * p + (p - 1)
                sb[m] = (sb[m] // p) * p + torch.randint(1, min(p, 100), (int(m.sum()),), device=DEVICE)
        # Decimal boundaries and sparse/repeated patterns.
        m = kind == 5
        if m.any():
            powers = torch.tensor([10,100,1000,10000,100000,1000000,10000000], device=DEVICE)
            p = powers[torch.randint(0, len(powers), (int(m.sum()),), device=DEVICE)]
            sa[m] = ((sa[m] // p) * p).clamp(LOW, HIGH)
            sb[m] = (((sb[m] // p) * p) + p - 1).clamp(LOW, HIGH)
        m = kind == 6
        if m.any():
            patterns = torch.tensor([11111111,22222222,33333333,44444444,55555555,66666666,77777777,88888888,99999999,10000000,90000000,99000000,99900000,99990000,99999000,99999900,99999990], device=DEVICE)
            sa[m] = patterns[torch.randint(0, len(patterns), (int(m.sum()),), device=DEVICE)]
        a[:n], b[:n] = sa, sb
    da, db = digits(a, 8), digits(b, 8)
    y = digits(a + b, 9)
    x = torch.empty(batch, 25, dtype=torch.long, device=DEVICE)
    x[:, :16:2], x[:, 1:16:2] = da, db
    x[:, 16] = 10
    x[:, 17:] = y[:, :8]
    return x, y


@torch.no_grad()
def evaluate(model, batches=20, structured=0.0, batch=8192):
    model.eval()
    correct = total = 0
    min_margin = 100.0
    for _ in range(batches):
        x, y = make_batch(batch, structured)
        logits = model(x)[:, 16:25]
        pred = logits.argmax(-1)
        correct += (pred == y).all(1).sum().item()
        total += batch
        true = logits.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        other = logits.masked_fill(F.one_hot(y, 10).bool(), -torch.inf).amax(-1)
        min_margin = min(min_margin, (true - other).min().item())
    return correct / total, min_margin


def save(model, step, name="checkpoint.pt"):
    torch.save({"model": model.state_dict(), "step": step}, "/workspace/" + name)


def main():
    torch.manual_seed(2301)
    random.seed(2301)
    model = AdditionTransformer().to(DEVICE)
    print("parameters", sum(p.numel() for p in model.parameters()), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(0.9, 0.98), weight_decay=0.01)
    start = time.time()
    best = 0.0
    steps = 48000
    for step in range(1, steps + 1):
        model.train()
        # Favor uniform data early, then broaden carry/boundary coverage.
        frac = 0.20 if step < 20000 else 0.35
        x, y = make_batch(BATCH, frac)
        logits = model(x)[:, 16:25]
        loss = F.cross_entropy(logits.reshape(-1, 10), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        warmup = min(1.0, step / 1000)
        decay = 0.5 * (1 + math.cos(math.pi * max(0, step - 1000) / (steps - 1000)))
        lr = 2e-3 * warmup * (0.08 + 0.92 * decay)
        for group in optimizer.param_groups: group["lr"] = lr
        if step % 1000 == 0:
            acc, margin = evaluate(model, 4, 0.0)
            edge, emargin = evaluate(model, 4, 0.8)
            print(step, f"loss={loss.item():.5f} random={acc:.6f} edge={edge:.6f} margins={margin:.3f}/{emargin:.3f} lr={lr:.2g} sec={time.time()-start:.0f}", flush=True)
            score = min(acc, edge)
            if score > best:
                best = score
                save(model, step, "best.pt")
        if step % 4000 == 0: save(model, step)
    save(model, steps, "final.pt")


if __name__ == "__main__":
    main()
