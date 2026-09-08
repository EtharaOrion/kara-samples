import math
from pathlib import Path
import torch
import torch.nn.functional as F
from submission import AdditionTransformer

DEVICE = "cuda"
BATCH = 4096
LO = 10_000_000
HI = 99_999_999
POW10 = torch.tensor([10 ** i for i in range(9)], device=DEVICE, dtype=torch.long)


def digits(x, width):
    return (x[:, None] // POW10[None, :width]) % 10


def structured(n):
    a = torch.randint(LO, HI + 1, (n,), device=DEVICE)
    b = torch.randint(LO, HI + 1, (n,), device=DEVICE)
    kind = torch.randint(0, 6, (n,), device=DEVICE)

    # Exact and near complements around 1e8, with every leading-digit combination represented.
    m = kind == 0
    count = int(m.sum())
    if count:
        x = torch.randint(LO, 90_000_001, (count,), device=DEVICE)
        delta = torch.randint(-20, 21, (count,), device=DEVICE)
        y = (100_000_000 - x + delta).clamp(LO, HI)
        a[m], b[m] = x, y

    # Long asymmetric 0/9 suffixes with randomized prefixes and carry initiators.
    m = kind == 1
    count = int(m.sum())
    if count:
        k = torch.randint(1, 8, (count,), device=DEVICE)
        p = 10 ** k
        x = torch.randint(1, 10 ** 7, (count,), device=DEVICE) * p + (p - 1)
        y = torch.randint(1, 10 ** 7, (count,), device=DEVICE) * p + torch.randint(1, 10, (count,), device=DEVICE)
        a[m] = x.clamp(LO, HI)
        b[m] = y.clamp(LO, HI)

    # Decimal boundaries at all scales, from either side.
    m = kind == 2
    count = int(m.sum())
    if count:
        k = torch.randint(1, 8, (count,), device=DEVICE)
        p = 10 ** k
        q = torch.randint(LO, HI + 1, (count,), device=DEVICE) // p * p
        x = (q + torch.randint(-12, 13, (count,), device=DEVICE)).clamp(LO, HI)
        y = torch.randint(LO, HI + 1, (count,), device=DEVICE)
        a[m], b[m] = x, y

    # Repeated digit operands, independently chosen.
    m = kind == 3
    count = int(m.sum())
    if count:
        da = torch.randint(1, 10, (count,), device=DEVICE)
        db = torch.randint(1, 10, (count,), device=DEVICE)
        a[m], b[m] = da * 11_111_111, db * 11_111_111

    # Sparse interior digits plus near-max partners.
    m = kind == 4
    count = int(m.sum())
    if count:
        lead = torch.randint(1, 10, (count,), device=DEVICE)
        pos = torch.randint(0, 7, (count,), device=DEVICE)
        val = torch.randint(0, 10, (count,), device=DEVICE)
        a[m] = lead * 10_000_000 + val * (10 ** pos)
        b[m] = HI - torch.randint(0, 1000, (count,), device=DEVICE)

    # Rounded operands with arbitrary leading digits and rounding scales.
    m = kind == 5
    count = int(m.sum())
    if count:
        k = torch.randint(1, 8, (count,), device=DEVICE)
        p = 10 ** k
        a[m] = (torch.randint(LO, HI + 1, (count,), device=DEVICE) // p * p).clamp(LO, HI)
        b[m] = (torch.randint(LO, HI + 1, (count,), device=DEVICE) // p * p).clamp(LO, HI)
    return a, b


def batch(n=BATCH, structured_fraction=0.3):
    a = torch.randint(LO, HI + 1, (n,), device=DEVICE)
    b = torch.randint(LO, HI + 1, (n,), device=DEVICE)
    ns = int(n * structured_fraction)
    if ns:
        a[:ns], b[:ns] = structured(ns)
    ad, bd, target = digits(a, 8), digits(b, 8), digits(a + b, 9)
    source = torch.empty((n, 17), dtype=torch.long, device=DEVICE)
    source[:, 0:16:2], source[:, 1:16:2], source[:, 16] = ad, bd, 10
    tokens = torch.cat((source, target[:, :8]), dim=1)
    return tokens, target, a, b


@torch.no_grad()
def evaluate(model, n=100_000, sf=0.0, chunk=4096):
    good = total = 0
    min_margin = 100.0
    while total < n:
        size = min(chunk, n - total)
        tokens, target, _, _ = batch(size, sf)
        seq = tokens[:, :17]
        pred = []
        margins = []
        for _ in range(9):
            logits = model(seq)[:, -1]
            top = logits.topk(2, dim=-1)
            pred.append(top.indices[:, 0])
            margins.append(top.values[:, 0] - top.values[:, 1])
            seq = torch.cat((seq, top.indices[:, :1]), dim=1)
        pred = torch.stack(pred, 1)
        correct = pred.eq(target).all(1)
        good += int(correct.sum())
        if bool(correct.any()):
            ms = torch.stack(margins, 1)[correct].min().item()
            min_margin = min(min_margin, ms)
        total += size
    return good / total, min_margin


def export(model):
    source = Path("/workspace/submission.py").read_text()
    marker = "_TRAINED_STATE = None"
    state = model.state_dict()
    lines = ["_TRAINED_STATE = {"]
    for name, tensor in state.items():
        values = repr(tensor.detach().cpu().tolist())
        lines.append(f"    {name!r}: torch.tensor({values}),")
    lines.append("}")
    source = source.replace(marker, "\n".join(lines))
    Path("/workspace/submission.py").write_text(source)


def main():
    torch.manual_seed(1501)
    model = AdditionTransformer().to(DEVICE)
    train_model = torch.compile(model, mode="max-autotune")
    print("parameters", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01, fused=True)
    best = 0.0
    for step in range(1, 30001):
        if step <= 22000:
            lr = 2e-3 * min(1.0, step / 500) * (0.15 + 0.85 * 0.5 * (1 + math.cos(math.pi * step / 22000)))
            sf = 0.30
        elif step <= 27000:
            lr, sf = 8e-5, 0.55
        else:
            lr, sf = 2e-5, 0.70
        for group in opt.param_groups:
            group["lr"] = lr
        tokens, target, _, _ = batch(structured_fraction=sf)
        logits = train_model(tokens)[:, 16:25]
        loss = F.cross_entropy(logits.reshape(-1, 10), target.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0:
            print(step, f"loss={loss.item():.6f}", f"lr={lr:.2g}", flush=True)
        if step >= 18000 and step % 2000 == 0:
            random_acc, margin = evaluate(model, 20000, 0.0)
            edge_acc, _ = evaluate(model, 20000, 1.0)
            score = min(random_acc, edge_acc)
            print("eval", random_acc, edge_acc, "margin", margin, flush=True)
            if score >= best:
                best = score
                torch.save(model.state_dict(), "/workspace/best.pt")
    model.load_state_dict(torch.load("/workspace/best.pt", weights_only=True))
    print("final", evaluate(model, 200000, 0.0), evaluate(model, 200000, 1.0), flush=True)
    export(model.cpu())


if __name__ == "__main__":
    main()
