import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import torch.nn.functional as F
from submission import Adder

DEVICE = "cuda"
BATCH = 4096
POW10 = torch.tensor([10**i for i in range(15)], device=DEVICE, dtype=torch.long)
POS = torch.arange(14, device=DEVICE)[None, :]


def digits_of(x, places=15):
    return (x[:, None] // POW10[None, :places]) % 10


def make_batch(n, structured=0.25):
    a = torch.randint(0, 100_000_000_000_000, (n,), device=DEVICE)
    b = torch.randint(0, 100_000_000_000_000, (n,), device=DEVICE)
    count = int(n * structured)
    if count:
        da = digits_of(a[:count], 14)
        db = digits_of(b[:count], 14)
        family = torch.randint(0, 6, (count,), device=DEVICE)

        # Randomized carry chains: initiating column sums to 10, subsequent columns to 9.
        rows = family == 0
        m = int(rows.sum())
        if m:
            x, y = da[rows], db[rows]
            start = torch.randint(0, 14, (m, 1), device=DEVICE)
            length = torch.randint(1, 15, (m, 1), device=DEVICE)
            chain = (POS >= start) & (POS < torch.minimum(start + length, start.new_tensor(14)))
            first = POS == start
            x = torch.where(chain, torch.randint(0, 10, x.shape, device=DEVICE), x)
            y = torch.where(chain, 9 - x, y)
            init_x = torch.randint(1, 10, (m, 14), device=DEVICE)
            x = torch.where(first, init_x, x)
            y = torch.where(first, 10 - x, y)
            da[rows], db[rows] = x, y

        # Long carries and top overflow.
        rows = family == 1
        m = int(rows.sum())
        if m:
            x = torch.randint(0, 10, (m, 14), device=DEVICE)
            y = 9 - x
            x[:, 0] = torch.randint(1, 10, (m,), device=DEVICE)
            y[:, 0] = 10 - x[:, 0]
            da[rows], db[rows] = x, y

        # Repeated-digit operands.
        rows = family == 2
        m = int(rows.sum())
        if m:
            da[rows] = torch.randint(0, 10, (m, 1), device=DEVICE)
            db[rows] = torch.randint(0, 10, (m, 1), device=DEVICE)

        # Sparse increments against runs of nines.
        rows = family == 3
        m = int(rows.sum())
        if m:
            x = torch.randint(0, 10, (m, 14), device=DEVICE)
            y = torch.zeros_like(x)
            start = torch.randint(0, 14, (m, 1), device=DEVICE)
            length = torch.randint(1, 15, (m, 1), device=DEVICE)
            run = (POS >= start) & (POS < torch.minimum(start + length, start.new_tensor(14)))
            x = torch.where(run, 9, x)
            y = torch.where(POS == start, 1, y)
            da[rows], db[rows] = x, y

        # Complementary runs at arbitrary locations.
        rows = family == 4
        m = int(rows.sum())
        if m:
            x, y = da[rows], db[rows]
            start = torch.randint(0, 14, (m, 1), device=DEVICE)
            length = torch.randint(1, 15, (m, 1), device=DEVICE)
            run = (POS >= start) & (POS < torch.minimum(start + length, start.new_tensor(14)))
            x = torch.where(run, torch.randint(0, 10, x.shape, device=DEVICE), x)
            y = torch.where(run, 9 - x, y)
            da[rows], db[rows] = x, y

        # Blockwise and sparse operands.
        rows = family == 5
        m = int(rows.sum())
        if m:
            block = torch.randint(1, 8, (m, 1), device=DEVICE)
            left = torch.randint(0, 10, (m, 1), device=DEVICE)
            right = torch.randint(0, 10, (m, 1), device=DEVICE)
            da[rows] = torch.where(POS < block, left, right)
            db[rows] = torch.where(torch.rand((m, 14), device=DEVICE) < 0.2,
                                   torch.randint(0, 10, (m, 14), device=DEVICE), 0)

        a[:count] = (da * POW10[:14]).sum(1)
        b[:count] = (db * POW10[:14]).sum(1)
        swap = torch.rand(count, device=DEVICE) < 0.5
        aa = a[:count].clone()
        a[:count] = torch.where(swap, b[:count], a[:count])
        b[:count] = torch.where(swap, aa, b[:count])

    da = digits_of(a, 14)
    db = digits_of(b, 14)
    out = digits_of(a + b, 15)
    seq = torch.empty((n, 45), dtype=torch.long, device=DEVICE)
    seq[:, 0] = 10
    seq[:, 1:29:2] = da
    seq[:, 2:29:2] = db
    seq[:, 29] = 11
    seq[:, 30:] = out
    return seq[:, :-1], out


@torch.inference_mode()
def evaluate(model, batches=8, structured=0.0, batch=8192):
    model.eval()
    errors = 0
    total = 0
    for _ in range(batches):
        x, target = make_batch(batch, structured)
        prefix = x[:, :30]
        generated = []
        for _ in range(15):
            digit = model(prefix)[:, -1].argmax(-1)
            generated.append(digit)
            prefix = torch.cat((prefix, digit[:, None]), 1)
        pred = torch.stack(generated, 1)
        errors += int((pred != target).any(1).sum())
        total += batch
    model.train()
    return errors, total


def main():
    torch.manual_seed(180017)
    torch.set_float32_matmul_precision("high")
    model = Adder().to(DEVICE)
    print("parameters", sum(p.numel() for p in model.parameters()), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.005, fused=True)
    best = None
    phases = [(8000, 3e-3, 0.18), (8000, 1e-3, 0.25),
              (7000, 3e-4, 0.30), (5000, 1e-4, 0.35)]
    step = 0
    for steps, lr, mix in phases:
        for group in optimizer.param_groups:
            group["lr"] = lr
        for _ in range(steps):
            step += 1
            x, target = make_batch(BATCH, mix)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits[:, 29:].reshape(-1, 10), target.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % 1000 == 0:
                print(f"step {step} lr {lr:g} loss {loss.item():.7f}", flush=True)
            if step >= 12000 and step % 2000 == 0:
                er, nr = evaluate(model, batches=4, structured=0.0)
                es, ns = evaluate(model, batches=4, structured=0.65)
                score = er + es
                print(f"validation random {er}/{nr}, structured {es}/{ns}", flush=True)
                if best is None or score <= best:
                    best = score
                    torch.save(model.state_dict(), "/workspace/model.pt")
                    print("saved checkpoint", score, flush=True)
    torch.save(model.state_dict(), "/workspace/model_final.pt")
    print("best", best, flush=True)


if __name__ == "__main__":
    main()
