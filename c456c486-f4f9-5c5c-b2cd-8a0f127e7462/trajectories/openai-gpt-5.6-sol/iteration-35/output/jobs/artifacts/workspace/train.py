import argparse
import importlib.util
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace")
from submission import AdditionTransformer

LIMIT = 100_000_000_000_000
POW10 = torch.tensor([10 ** i for i in range(14)], device="cuda", dtype=torch.long)


def integer_digits(values):
    return (values[:, None] // POW10[None, :]) % 10


def digits_integer(digits):
    return (digits * POW10).sum(1)


def labels_for(a_digits, b_digits):
    sums = digits_integer(a_digits) + digits_integer(b_digits)
    powers = torch.cat((POW10, torch.tensor([10 ** 14], device="cuda", dtype=torch.long)))
    return (sums[:, None] // powers[None, :]) % 10


def uniform_batch(count):
    a = torch.randint(0, LIMIT, (count,), device="cuda")
    b = torch.randint(0, LIMIT, (count,), device="cuda")
    return integer_digits(a), integer_digits(b)


def structured_batch(count):
    # Every family is regenerated each batch; all targets remain complete 15-digit additions.
    a = torch.randint(0, 10, (count, 14), device="cuda")
    b = torch.randint(0, 10, (count, 14), device="cuda")
    family = torch.arange(count, device="cuda") % 7
    cols = torch.arange(14, device="cuda")[None, :]

    # Carry chains with randomized start and length, followed by an explicit stopping column.
    rows = (family == 0).nonzero().flatten()
    if rows.numel():
        start = torch.randint(0, 14, (rows.numel(),), device="cuda")
        length = torch.randint(1, 15, (rows.numel(),), device="cuda")
        end = torch.minimum(start + length, torch.full_like(start, 14))
        in_run = (cols >= start[:, None]) & (cols < end[:, None])
        av = torch.randint(1, 10, (rows.numel(), 14), device="cuda")
        extra = torch.randint(0, 2, (rows.numel(), 14), device="cuda")
        bv = torch.minimum(9 - av + extra, torch.full_like(av, 9))
        # The first column must initiate rather than merely propagate a carry.
        bv[torch.arange(rows.numel(), device="cuda"), start] = 10 - av[torch.arange(rows.numel(), device="cuda"), start]
        a[rows] = torch.where(in_run, av, a[rows])
        b[rows] = torch.where(in_run, bv, b[rows])
        valid_end = end < 14
        if valid_end.any():
            rr, cc = rows[valid_end], end[valid_end]
            av_stop = torch.randint(0, 9, (rr.numel(),), device="cuda")
            a[rr, cc] = av_stop
            b[rr, cc] = 8 - av_stop

    # Long matched non-carry contrasts: column sums 8 or 9, with no incoming carry.
    rows = (family == 1).nonzero().flatten()
    if rows.numel():
        start = torch.randint(0, 12, (rows.numel(),), device="cuda")
        length = torch.randint(2, 15, (rows.numel(),), device="cuda")
        end = torch.minimum(start + length, torch.full_like(start, 14))
        run = (cols >= start[:, None]) & (cols < end[:, None])
        av = torch.randint(0, 10, (rows.numel(), 14), device="cuda")
        total = torch.randint(8, 10, (rows.numel(), 14), device="cuda")
        av = torch.minimum(av, total)
        a[rows] = torch.where(run, av, a[rows])
        b[rows] = torch.where(run, total - av, b[rows])
        if (start > 0).any():
            ok = start > 0
            rr, cc = rows[ok], start[ok] - 1
            a[rr, cc] = 0
            b[rr, cc] = 0

    # Sparse isolated boundaries, deliberately including 5+5 and 9+9 at any height.
    rows = (family == 2).nonzero().flatten()
    if rows.numel():
        a[rows] = 0; b[rows] = 0
        col = torch.randint(0, 14, (rows.numel(),), device="cuda")
        kind = torch.arange(rows.numel(), device="cuda") % 4
        av = torch.where(kind == 0, 5, torch.where(kind == 1, 9, torch.randint(0, 10, kind.shape, device="cuda")))
        bv = torch.where(kind == 0, 5, torch.where(kind == 1, 9, 9 - av))
        a[rows, col] = av; b[rows, col] = bv
        second = torch.randint(0, 14, (rows.numel(),), device="cuda")
        a[rows[::3], second[::3]] = torch.randint(0, 10, (rows[::3].numel(),), device="cuda")

    # Repeated and blockwise digits.
    rows = (family == 3).nonzero().flatten()
    if rows.numel():
        da = torch.randint(0, 10, (rows.numel(), 1), device="cuda")
        db = torch.randint(0, 10, (rows.numel(), 1), device="cuda")
        a[rows] = da; b[rows] = db
        half = rows[::2]
        if half.numel():
            cut = torch.randint(1, 14, (half.numel(),), device="cuda")
            replacement = torch.randint(0, 10, (half.numel(), 1), device="cuda")
            a[half] = torch.where(cols < cut[:, None], a[half], replacement)

    # Complement runs with a carry initiator at a random location.
    rows = (family == 4).nonzero().flatten()
    if rows.numel():
        a[rows] = torch.randint(0, 10, (rows.numel(), 14), device="cuda")
        b[rows] = 9 - a[rows]
        start = torch.randint(0, 14, (rows.numel(),), device="cuda")
        rr = torch.arange(rows.numel(), device="cuda")
        can_raise = a[rows, start] < 9
        a[rows[can_raise], start[can_raise]] += 1

    # Near-boundary sparse increments and all-nine overflows.
    rows = (family == 5).nonzero().flatten()
    if rows.numel():
        a[rows] = 0; b[rows] = 0
        length = torch.randint(1, 15, (rows.numel(),), device="cuda")
        run = cols < length[:, None]
        a[rows] = torch.where(run, torch.full_like(a[rows], 9), a[rows])
        variant = torch.arange(rows.numel(), device="cuda") % 3
        b[rows, 0] = torch.where(variant == 0, 1, torch.where(variant == 1, 0, 9))
        a[rows[variant == 1], 0] = 8

    # Operand-swap and random high-column isolated sums.
    rows = (family == 6).nonzero().flatten()
    if rows.numel():
        col = torch.randint(7, 14, (rows.numel(),), device="cuda")
        lowmask = cols < col[:, None]
        a[rows] = torch.where(lowmask, torch.zeros_like(a[rows]), a[rows])
        b[rows] = torch.where(lowmask, torch.zeros_like(b[rows]), b[rows])
        swap = rows[::2]
        old = a[swap].clone(); a[swap] = b[swap]; b[swap] = old
    return a, b


def make_batch(batch, structured_fraction):
    structured = int(batch * structured_fraction)
    uniform = batch - structured
    au, bu = uniform_batch(uniform)
    if not structured:
        return au, bu, labels_for(au, bu)
    ast, bst = structured_batch(structured)
    a = torch.cat((au, ast)); b = torch.cat((bu, bst))
    order = torch.randperm(batch, device="cuda")
    a, b = a[order], b[order]
    return a, b, labels_for(a, b)


def loss_for(model, a, b, target):
    logits = model(a, b, target[:, :-1])[:, 14:29]
    return F.cross_entropy(logits.reshape(-1, 10), target.reshape(-1))

@torch.no_grad()
def autoregressive_errors(model, count, structured=False, chunk=8192):
    errors = 0
    for offset in range(0, count, chunk):
        n = min(chunk, count - offset)
        if structured:
            a, b = structured_batch(n)
        else:
            a, b = uniform_batch(n)
        target = labels_for(a, b)
        out = torch.empty((n, 0), dtype=torch.long, device="cuda")
        for _ in range(15):
            out = torch.cat((out, model(a, b, out)[:, -1].argmax(1, keepdim=True)), 1)
        errors += (out != target).any(1).sum().item()
    return errors


def write_submission(model):
    path = "/workspace/submission.py"
    text = open(path).read()
    marker = "_STATE = None"
    values = []
    for p in model.parameters():
        values.append(p.detach().float().cpu().reshape(-1).tolist())
    literal = "_STATE = " + repr(values)
    text = text.replace(marker, literal)
    open(path, "w").write(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=96000)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(350035)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = AdditionTransformer().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.002, fused=True)
    start = 0
    checkpoint = "/workspace/checkpoint.pt"
    if args.resume and os.path.exists(checkpoint):
        data = torch.load(checkpoint, weights_only=False)
        model.load_state_dict(data["model"]); optimizer.load_state_dict(data["optimizer"]); start = data["step"]
    print("parameters", sum(p.numel() for p in model.parameters()), flush=True)
    began = time.time()
    for step in range(start, args.steps):
        if step < 14000: lr, frac = 3e-3, 0.0
        elif step < 32000: lr, frac = 1e-3, 0.45
        elif step < 52000: lr, frac = 3e-4, 0.48
        elif step < 72000: lr, frac = 1e-4, 0.48
        elif step < 86000: lr, frac = 3e-5, 0.48
        else: lr, frac = 1e-5, 0.48
        for group in optimizer.param_groups: group["lr"] = lr
        a, b, target = make_batch(args.batch, frac)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for(model, a, b, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % 2000 == 0:
            elapsed = time.time() - began
            # Cheap teacher-forced sequence metric for progress; AR tests happen at phase boundaries.
            with torch.no_grad():
                va, vb, vy = make_batch(16384, 0.5)
                pred = model(va, vb, vy[:, :-1])[:, 14:29].argmax(2)
                terr = (pred != vy).any(1).sum().item()
            print(step + 1, "loss", float(loss), "teacher_err", terr, "seconds", round(elapsed, 1), flush=True)
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step + 1}, checkpoint)
    model.eval()
    random_err = autoregressive_errors(model, 524288, False)
    structured_err = autoregressive_errors(model, 524288, True)
    print("FINAL random", random_err, "structured", structured_err, flush=True)
    torch.save({"model": model.state_dict(), "step": args.steps, "random_err": random_err, "structured_err": structured_err}, "/workspace/final.pt")
    write_submission(model)

if __name__ == "__main__":
    main()
