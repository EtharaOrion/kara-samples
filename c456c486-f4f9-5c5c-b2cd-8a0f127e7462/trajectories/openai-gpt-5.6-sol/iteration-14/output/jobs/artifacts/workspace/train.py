import argparse
import copy
import math
import random
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from submission import AdditionTransformer


LENGTH = 15
DATA_DIGITS = 14


def sums_from_digits(a, b):
    target = torch.empty_like(a)
    carry = torch.zeros(a.shape[0], dtype=torch.long, device=a.device)
    for i in range(LENGTH):
        total = a[:, i] + b[:, i] + carry
        target[:, i] = total.remainder(10)
        carry = total.div(10, rounding_mode="floor")
    return target


def random_batch(batch, device):
    a = torch.randint(10, (batch, LENGTH), device=device)
    b = torch.randint(10, (batch, LENGTH), device=device)
    a[:, -1] = 0
    b[:, -1] = 0
    return a, b, sums_from_digits(a, b)


def mixed_batch(batch, device, structured_fraction=0.35, long_fraction=0.10):
    a, b, _ = random_batch(batch, device)
    n = int(batch * structured_fraction)
    if not n:
        return a, b, sums_from_digits(a, b)
    rows = torch.arange(n, device=device)
    family = torch.randint(8, (n,), device=device)

    # Repeated and blockwise patterns.
    mask = family == 0
    count = int(mask.sum())
    if count:
        rr = rows[mask]
        a[rr, :DATA_DIGITS] = torch.randint(10, (count, 1), device=device)
        b[rr, :DATA_DIGITS] = torch.randint(10, (count, 1), device=device)
    mask = family == 1
    count = int(mask.sum())
    if count:
        rr = rows[mask]
        split = torch.randint(1, DATA_DIGITS, (count, 1), device=device)
        pos = torch.arange(DATA_DIGITS, device=device).view(1, -1)
        av = torch.randint(10, (count, 2), device=device)
        bv = torch.randint(10, (count, 2), device=device)
        a[rr, :DATA_DIGITS] = torch.where(pos < split, av[:, :1], av[:, 1:])
        b[rr, :DATA_DIGITS] = torch.where(pos < split, bv[:, :1], bv[:, 1:])

    # Complementary runs, optionally initiated by a carry.
    mask = family == 2
    count = int(mask.sum())
    if count:
        rr = rows[mask]
        start = torch.randint(0, DATA_DIGITS, (count, 1), device=device)
        max_len = DATA_DIGITS - start
        length = 1 + (torch.rand((count, 1), device=device) * max_len).long()
        pos = torch.arange(DATA_DIGITS, device=device).view(1, -1)
        run = (pos >= start) & (pos < start + length)
        da = torch.randint(10, (count, DATA_DIGITS), device=device)
        db = 9 - da
        a[rr, :DATA_DIGITS] = torch.where(run, da, a[rr, :DATA_DIGITS])
        b[rr, :DATA_DIGITS] = torch.where(run, db, b[rr, :DATA_DIGITS])
        s = start.squeeze(1)
        av = torch.randint(1, 10, (count,), device=device)
        a[rr, s] = av
        b[rr, s] = 10 - av

    # Sparse 1 added to a run of 9s at arbitrary position and length.
    mask = (family == 3) | (family == 4)
    count = int(mask.sum())
    if count:
        rr = rows[mask]
        a[rr] = 0
        b[rr] = 0
        start = torch.randint(0, DATA_DIGITS, (count, 1), device=device)
        max_len = DATA_DIGITS - start
        if long_fraction > 0:
            choose_long = torch.rand((count, 1), device=device) < long_fraction
            ordinary = 1 + (torch.rand((count, 1), device=device) * max_len).long()
            long_len = torch.clamp(max_len - torch.randint(0, 3, (count, 1), device=device), min=1)
            length = torch.where(choose_long, long_len, ordinary)
        else:
            length = 1 + (torch.rand((count, 1), device=device) * max_len).long()
        pos = torch.arange(DATA_DIGITS, device=device).view(1, -1)
        run = (pos >= start) & (pos < start + length)
        a[rr, :DATA_DIGITS] = run.long() * 9
        b[rr, start.squeeze(1)] = 1
        swap = torch.rand(count, device=device) < 0.5
        temp = a[rr[swap]].clone()
        a[rr[swap]] = b[rr[swap]]
        b[rr[swap]] = temp

    # Dense exact carry chains: initiating sum 10+, continuation sum 9.
    mask = family == 5
    count = int(mask.sum())
    if count:
        rr = rows[mask]
        start = torch.randint(0, DATA_DIGITS, (count, 1), device=device)
        max_len = DATA_DIGITS - start
        length = 1 + (torch.rand((count, 1), device=device) * max_len).long()
        pos = torch.arange(DATA_DIGITS, device=device).view(1, -1)
        continuation = (pos > start) & (pos < start + length)
        da = torch.randint(10, (count, DATA_DIGITS), device=device)
        a[rr, :DATA_DIGITS] = torch.where(continuation, da, a[rr, :DATA_DIGITS])
        b[rr, :DATA_DIGITS] = torch.where(continuation, 9 - da, b[rr, :DATA_DIGITS])
        s = start.squeeze(1)
        av = torch.randint(1, 10, (count,), device=device)
        extra = torch.randint(0, 9, (count,), device=device)
        extra = torch.minimum(extra, av - 1)
        a[rr, s] = av
        b[rr, s] = 10 - av + extra

    # Mostly sparse operands and near-maximum values.
    mask = family == 6
    count = int(mask.sum())
    if count:
        rr = rows[mask]
        keep_a = torch.rand((count, DATA_DIGITS), device=device) < 0.18
        keep_b = torch.rand((count, DATA_DIGITS), device=device) < 0.18
        a[rr, :DATA_DIGITS] *= keep_a
        b[rr, :DATA_DIGITS] *= keep_b
    mask = family == 7
    count = int(mask.sum())
    if count:
        rr = rows[mask]
        a[rr, :DATA_DIGITS] = 9 - torch.randint(3, (count, DATA_DIGITS), device=device)
        b[rr, :DATA_DIGITS] = torch.randint(3, (count, DATA_DIGITS), device=device)

    a[:, -1] = 0
    b[:, -1] = 0
    return a, b, sums_from_digits(a, b)


def exact_errors(model, batches, batch_size, kind="random"):
    errors = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for _ in range(batches):
            if kind == "random":
                a, b, y = random_batch(batch_size, next(model.parameters()).device)
            else:
                a, b, y = mixed_batch(batch_size, next(model.parameters()).device, 1.0, 0.8)
            pred = model(a, b).argmax(-1)
            errors += int((pred != y).any(dim=1).sum())
            total += batch_size
    model.train()
    return errors, total


def edge_cases(device):
    cases = {(0, 0), (10**14 - 1, 1), (1, 10**14 - 1), (10**14 - 1, 10**14 - 1)}
    for start in range(DATA_DIGITS):
        for length in range(1, DATA_DIGITS - start + 1):
            run = ((10**length) - 1) * (10**start)
            inc = 10**start
            cases.add((run, inc)); cases.add((inc, run))
            if start + length < DATA_DIGITS:
                prefix = 7 * 10**(start + length)
                cases.add((prefix + run, inc)); cases.add((inc, prefix + run))
    for d in range(10):
        v = int(str(d) * DATA_DIGITS)
        for e in range(10):
            cases.add((v, int(str(e) * DATA_DIGITS)))
    pairs = sorted(cases)
    a = torch.tensor([[int(c) for c in f"{x:015d}"[::-1]] for x, _ in pairs], device=device)
    b = torch.tensor([[int(c) for c in f"{y:015d}"[::-1]] for _, y in pairs], device=device)
    return a, b, sums_from_digits(a, b), pairs


def export_submission(model, path):
    template = Path(__file__).with_name("submission.py").read_text()
    marker = "        self.reset_parameters()"
    lines = ["        self.reset_parameters()", "        self.load_state_dict({"]
    for name, tensor in model.state_dict().items():
        values = tensor.detach().float().cpu().reshape(-1).tolist()
        literal = ",".join(f"{v:.9g}" for v in values)
        lines.append(f"            {name!r}: torch.tensor([{literal}]).reshape{tuple(tensor.shape)},")
    lines.append("        })")
    output = template.replace(marker, "\n".join(lines), 1)
    path.write_text(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=33000)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=14)
    args = parser.parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdditionTransformer().to(device)
    print("device", device, "parameters", sum(p.numel() for p in model.parameters()), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.005, fused=device.type == "cuda")
    best = None
    best_score = 10**9
    for step in range(1, args.steps + 1):
        if step <= 16000:
            lr, sf, lf = (3e-3 if step <= 8000 else 1e-3), 0.35, 0.15
        elif step <= 25000:
            lr, sf, lf = 3e-4, 0.45, 0.35
        elif step <= 30000:
            lr, sf, lf = 1e-4, 0.55, 0.75
        else:
            lr, sf, lf = 3e-5, 0.45, 0.55
        for group in optimizer.param_groups:
            group["lr"] = lr
        a, b, y = mixed_batch(args.batch, device, sf, lf)
        optimizer.zero_grad(set_to_none=True)
        logits = model(a, b)
        loss = F.cross_entropy(logits.reshape(-1, 10), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 1000 == 0 or step == args.steps:
            re, rn = exact_errors(model, 8, 4096, "random")
            se, sn = exact_errors(model, 4, 4096, "structured")
            ea, eb, ey, _ = edge_cases(device)
            with torch.no_grad():
                ee = int((model(ea, eb).argmax(-1) != ey).any(dim=1).sum())
            score = re * 10 + se * 3 + ee * 100
            print(f"step {step} loss {loss.item():.5g} random {re}/{rn} structured {se}/{sn} edge {ee}/{len(ea)}", flush=True)
            # Late checkpoints win ties, avoiding the prior early-tie selection bug.
            if score <= best_score:
                best_score = score
                best = copy.deepcopy(model.state_dict())
                torch.save(best, "/workspace/best.pt")
    if best is not None:
        model.load_state_dict(best)
    torch.save(model.state_dict(), "/workspace/final.pt")
    export_submission(model, Path("/workspace/submission.py"))
    print("exported /workspace/submission.py", flush=True)


if __name__ == "__main__":
    main()
