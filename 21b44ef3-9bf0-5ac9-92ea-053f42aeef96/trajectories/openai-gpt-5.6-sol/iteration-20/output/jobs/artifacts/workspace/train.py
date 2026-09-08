import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from submission import AdditionTransformer

DEVICE = "cuda"
BATCH = 8192
STEPS = 40000
LOW = 10_000_000
HIGH = 99_999_999
POW10 = torch.tensor([10 ** i for i in range(9)], device=DEVICE, dtype=torch.long)


def digits(x, count=8):
    return (x[:, None] // POW10[None, :count]) % 10


def uniform(n):
    return (
        torch.randint(LOW, HIGH + 1, (n,), device=DEVICE),
        torch.randint(LOW, HIGH + 1, (n,), device=DEVICE),
    )


def structured(n):
    a, b = uniform(n)
    kind = torch.randint(0, 8, (n,), device=DEVICE)

    # Complements around 100,000,000 and 110,000,000 exercise full carry chains.
    m = kind == 0
    if m.any():
        aa = torch.randint(LOW, 90_000_001, (int(m.sum()),), device=DEVICE)
        delta = torch.randint(-20, 21, aa.shape, device=DEVICE)
        bb = (100_000_000 - aa + delta).clamp(LOW, HIGH)
        a[m], b[m] = aa, bb
    m = kind == 1
    if m.any():
        aa = torch.randint(10_000_001, HIGH + 1, (int(m.sum()),), device=DEVICE)
        delta = torch.randint(-9, 10, aa.shape, device=DEVICE)
        bb = (110_000_000 - aa + delta).clamp(LOW, HIGH)
        a[m], b[m] = aa, bb

    # Asymmetric zero/nine suffixes at every possible length.
    m = kind == 2
    if m.any():
        count = int(m.sum())
        k = torch.randint(1, 8, (count,), device=DEVICE)
        scale = POW10[k]
        prefix_a = torch.randint(1, 10, (count,), device=DEVICE) * 10_000_000
        prefix_a += torch.randint(0, 10_000_000, (count,), device=DEVICE)
        aa = (prefix_a // scale) * scale + scale - 1
        bb = torch.randint(LOW, HIGH + 1, (count,), device=DEVICE)
        btail = torch.randint(1, 10, (count,), device=DEVICE)
        bb = (bb // scale) * scale + btail
        a[m], b[m] = aa.clamp(LOW, HIGH), bb.clamp(LOW, HIGH)
    m = kind == 3
    if m.any():
        count = int(m.sum())
        k = torch.randint(1, 8, (count,), device=DEVICE)
        scale = POW10[k]
        aa = torch.randint(LOW, HIGH + 1, (count,), device=DEVICE)
        aa = (aa // scale) * scale
        bb = torch.randint(LOW, HIGH + 1, (count,), device=DEVICE)
        bb = (bb // scale) * scale + scale - torch.randint(1, 11, (count,), device=DEVICE)
        a[m], b[m] = aa.clamp(LOW, HIGH), bb.clamp(LOW, HIGH)

    # Decimal boundaries and nearby values.
    m = kind == 4
    if m.any():
        count = int(m.sum())
        k = torch.randint(1, 8, (count,), device=DEVICE)
        scale = POW10[k]
        aa = torch.randint(LOW, HIGH + 1, (count,), device=DEVICE)
        bb = torch.randint(LOW, HIGH + 1, (count,), device=DEVICE)
        aa = (aa // scale) * scale + torch.randint(-12, 13, (count,), device=DEVICE)
        bb = (bb // scale) * scale + torch.randint(-12, 13, (count,), device=DEVICE)
        a[m], b[m] = aa.clamp(LOW, HIGH), bb.clamp(LOW, HIGH)

    # Repeated digit operands.
    m = kind == 5
    if m.any():
        count = int(m.sum())
        da = torch.randint(1, 10, (count,), device=DEVICE)
        db = torch.randint(1, 10, (count,), device=DEVICE)
        rep = torch.tensor(11_111_111, device=DEVICE)
        a[m], b[m] = da * rep, db * rep

    # Sparse internal digits while retaining nonzero leading digits.
    m = kind == 6
    if m.any():
        count = int(m.sum())
        aa = torch.randint(1, 10, (count,), device=DEVICE) * 10_000_000
        bb = torch.randint(1, 10, (count,), device=DEVICE) * 10_000_000
        for _ in range(2):
            p = torch.randint(0, 7, (count,), device=DEVICE)
            aa += torch.randint(0, 10, (count,), device=DEVICE) * POW10[p]
            p = torch.randint(0, 7, (count,), device=DEVICE)
            bb += torch.randint(0, 10, (count,), device=DEVICE) * POW10[p]
        a[m], b[m] = aa, bb

    # Near extrema, independently and asymmetrically.
    m = kind == 7
    if m.any():
        count = int(m.sum())
        offsets_a = torch.randint(0, 100_000, (count,), device=DEVICE)
        offsets_b = torch.randint(0, 100_000, (count,), device=DEVICE)
        side_a = torch.randint(0, 2, (count,), device=DEVICE).bool()
        side_b = torch.randint(0, 2, (count,), device=DEVICE).bool()
        aa = torch.where(side_a, LOW + offsets_a, HIGH - offsets_a)
        bb = torch.where(side_b, LOW + offsets_b, HIGH - offsets_b)
        a[m], b[m] = aa, bb
    return a, b


def make_batch(n, structured_fraction):
    a, b = uniform(n)
    take = int(n * structured_fraction)
    if take:
        a[:take], b[:take] = structured(take)
    ad, bd = digits(a), digits(b)
    result = digits(a + b, 9)
    operands = torch.stack((ad, bd), dim=2).reshape(n, 16)
    sentinel = torch.full((n, 1), 10, device=DEVICE, dtype=torch.long)
    tokens = torch.cat((operands, sentinel, result[:, :-1]), dim=1)
    return tokens, result, a, b


@torch.no_grad()
def evaluate(model, batches, structured_fraction):
    model.eval()
    correct = 0
    total = 0
    digit_correct = 0
    margin_min = 100.0
    for _ in range(batches):
        tokens, target, _, _ = make_batch(8192, structured_fraction)
        generated = tokens[:, :17]
        margins = []
        for column in range(9):
            logits = model(generated)[:, -1]
            top = logits.topk(2, dim=-1)
            margins.append(top.values[:, 0] - top.values[:, 1])
            prediction = top.indices[:, 0]
            generated = torch.cat((generated, prediction[:, None]), dim=1)
        prediction = generated[:, 17:]
        correct += (prediction == target).all(dim=1).sum().item()
        digit_correct += (prediction == target).sum().item()
        total += target.shape[0]
        margin_min = min(margin_min, torch.stack(margins, 1).min().item())
    model.train()
    return correct / total, digit_correct / (total * 9), margin_min


def save_checkpoint(model, step, score):
    torch.save({"model": model.state_dict(), "step": step, "score": score}, "/workspace/best.pt")


def export(model):
    path = Path("/workspace/submission.py")
    text = path.read_text()
    values = []
    for parameter in model.parameters():
        values.extend(parameter.detach().cpu().float().reshape(-1).tolist())
    literal = ",".join(format(value, ".9g") for value in values)
    start = text.index("_WEIGHTS = ")
    end = text.index("\n\n\ndef build_model", start)
    text = text[:start] + "_WEIGHTS = (" + literal + ",)" + text[end:]
    path.write_text(text)
    print(f"exported {len(values)} parameters, submission bytes={path.stat().st_size}")


def main():
    torch.manual_seed(7193)
    random.seed(7193)
    torch.set_float32_matmul_precision("high")
    model = AdditionTransformer().to(DEVICE)
    print("parameters", sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.2e-3, betas=(0.9, 0.98), weight_decay=0.01, fused=True)
    best = -1.0
    for step in range(1, STEPS + 1):
        if step <= 1000:
            lr = 2.2e-3 * step / 1000
        elif step <= 30000:
            lr = 2.2e-3 * (0.08 + 0.92 * 0.5 * (1 + math.cos(math.pi * (step - 1000) / 29000)))
        elif step <= 36000:
            lr = 8e-5
        else:
            lr = 2e-5
        for group in optimizer.param_groups:
            group["lr"] = lr
        frac = 0.20 if step <= 30000 else 0.55
        tokens, target, _, _ = make_batch(BATCH, frac)
        logits = model(tokens)[:, 16:25]
        loss = F.cross_entropy(logits.reshape(-1, 10), target.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 1000 == 0:
            print(f"step={step} loss={loss.item():.6f} lr={lr:.2g}", flush=True)
        if step % 4000 == 0 or step >= 38000 and step % 1000 == 0:
            random_acc, digit_acc, margin = evaluate(model, 4, 0.0)
            edge_acc, _, edge_margin = evaluate(model, 4, 1.0)
            score = min(random_acc, edge_acc)
            print(f"VALID step={step} random={random_acc:.6f} edge={edge_acc:.6f} digit={digit_acc:.8f} margins={margin:.4f}/{edge_margin:.4f}", flush=True)
            if score > best:
                best = score
                save_checkpoint(model, step, score)
    checkpoint = torch.load("/workspace/best.pt", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    print("selected", checkpoint["step"], checkpoint["score"])
    random_acc, _, margin = evaluate(model, 32, 0.0)
    edge_acc, _, edge_margin = evaluate(model, 32, 1.0)
    print(f"FINAL random={random_acc:.8f} edge={edge_acc:.8f} margins={margin:.4f}/{edge_margin:.4f}")
    export(model)


if __name__ == "__main__":
    main()
