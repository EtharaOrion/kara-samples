import argparse
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from submission import AdditionTransformer

LIMIT = 100_000_000_000_000
POW10 = torch.tensor([10 ** i for i in range(15)], dtype=torch.long)


def to_digits(x):
    return (x[:, None] // POW10.to(x.device)[None, :]) % 10


def targets(a, b):
    return to_digits(a + b)


def from_digits(d):
    return (d * POW10[:14].to(d.device)).sum(1)


def carry_examples(n, device):
    # Randomized explicit carry chains: initiation sum 10, propagation sum 9,
    # and a non-carry terminator where one exists.
    da = torch.randint(0, 10, (n, 14), device=device)
    db = torch.randint(0, 10, (n, 14), device=device)
    start = torch.randint(0, 14, (n, 1), device=device)
    max_len = 14 - start
    # Half are deliberately long/top-overflow chains; half span any valid length.
    long = torch.rand((n, 1), device=device) < 0.55
    sampled = 1 + (torch.rand((n, 1), device=device) * max_len).long()
    length = torch.where(long, max_len, sampled)
    pos = torch.arange(14, device=device)[None, :]
    initiate = pos == start
    propagate = (pos > start) & (pos < start + length)
    terminate = pos == start + length
    first = torch.randint(1, 10, (n, 1), device=device)
    comp = torch.randint(0, 10, (n, 14), device=device)
    da = torch.where(initiate, first, da)
    db = torch.where(initiate, 10 - first, db)
    da = torch.where(propagate, comp, da)
    db = torch.where(propagate, 9 - comp, db)
    end_a = torch.randint(0, 9, (n, 1), device=device)
    end_b = torch.floor(torch.rand((n, 1), device=device) * (9 - end_a)).long()
    da = torch.where(terminate, end_a, da)
    db = torch.where(terminate, end_b, db)
    return from_digits(da), from_digits(db)


def pattern_examples(n, device):
    da = torch.randint(0, 10, (n, 14), device=device)
    db = torch.randint(0, 10, (n, 14), device=device)
    kind = torch.randint(0, 6, (n, 1), device=device)
    pos = torch.arange(14, device=device)[None, :]

    # Repeated operands.
    ra = torch.randint(0, 10, (n, 1), device=device).expand(-1, 14)
    rb = torch.randint(0, 10, (n, 1), device=device).expand(-1, 14)
    da = torch.where(kind == 0, ra, da)
    db = torch.where(kind == 0, rb, db)

    # Sparse increments at arbitrary boundaries.
    loc = torch.randint(0, 14, (n, 1), device=device)
    sparse = (pos == loc).long() * torch.randint(1, 10, (n, 1), device=device)
    db = torch.where(kind == 1, sparse, db)

    # Complementary runs, including operand swaps through independently random a.
    lo = torch.randint(0, 14, (n, 1), device=device)
    run_len = 1 + (torch.rand((n, 1), device=device) * (14 - lo)).long()
    run = (pos >= lo) & (pos < lo + run_len)
    complement = torch.where(run, 9 - da, db)
    db = torch.where(kind == 2, complement, db)

    # Blockwise/repeating-period patterns.
    period = 1 + torch.randint(0, 5, (n, 1), device=device)
    seeds_a = torch.randint(0, 10, (n, 5), device=device)
    seeds_b = torch.randint(0, 10, (n, 5), device=device)
    idx = (pos % period).clamp_max(4)
    block_a = seeds_a.gather(1, idx)
    block_b = seeds_b.gather(1, idx)
    da = torch.where(kind == 3, block_a, da)
    db = torch.where(kind == 3, block_b, db)

    # Many 9s plus a sparse increment (carry stress).
    nine_run = torch.where(run, torch.full_like(da, 9), da)
    da = torch.where(kind == 4, nine_run, da)
    db = torch.where(kind == 4, sparse, db)

    # Boundary overflow and near-maximum values.
    maxish = torch.full_like(da, 9)
    holes = torch.rand((n, 14), device=device) < 0.12
    maxish = torch.where(holes, torch.randint(0, 9, (n, 14), device=device), maxish)
    da = torch.where(kind == 5, maxish, da)
    db = torch.where(kind == 5, sparse, db)
    return from_digits(da), from_digits(db)


def mixed_batch(batch, device, phase):
    if phase == 0:
        nc, npat = batch // 4, batch // 8
    elif phase == 1:
        nc, npat = batch // 2, batch // 4
    else:
        nc, npat = batch * 3 // 8, batch // 4
    nr = batch - nc - npat
    a = torch.randint(0, LIMIT, (nr,), device=device)
    b = torch.randint(0, LIMIT, (nr,), device=device)
    ca, cb = carry_examples(nc, device)
    pa, pb = pattern_examples(npat, device)
    a = torch.cat((a, ca, pa))
    b = torch.cat((b, cb, pb))
    # Random operand swap preserves the data distribution but prevents role artifacts.
    swap = torch.rand(batch, device=device) < 0.5
    return torch.where(swap, b, a), torch.where(swap, a, b)


def inputs(a, b):
    left, right = to_digits(a), to_digits(b)
    return left, right

@torch.no_grad()
def evaluate(model, batches, batch, family, device):
    model.eval()
    errors = 0
    total = 0
    for _ in range(batches):
        if family == "random":
            a = torch.randint(0, LIMIT, (batch,), device=device)
            b = torch.randint(0, LIMIT, (batch,), device=device)
        elif family == "carry":
            a, b = carry_examples(batch, device)
        else:
            a, b = pattern_examples(batch, device)
        left, right = inputs(a, b)
        pred = model(left, right).argmax(-1)
        errors += (pred != targets(a, b)).any(1).sum().item()
        total += batch
    model.train()
    return errors, total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=33000)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--seed", type=int, default=15024)
    p.add_argument("--resume", type=str)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda")
    model = AdditionTransformer().to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, weights_only=True))
    print("parameters", sum(p.numel() for p in model.parameters()), flush=True)
    # Compile the repeated tiny block into fused kernels.
    train_model = torch.jit.script(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.98), weight_decay=0.003)
    scaler = None
    started = time.time()
    best_score = 10**9
    for step in range(1, args.steps + 1):
        progress = step / args.steps
        phase = 0 if progress < 0.58 else (1 if progress < 0.78 else 2)
        if progress < 0.40:
            lr = 3e-3
        elif progress < 0.68:
            lr = 1e-3
        elif progress < 0.84:
            lr = 3e-4
        elif progress < 0.94:
            lr = 1e-4
        else:
            lr = 3e-5
        for group in optimizer.param_groups:
            group["lr"] = lr
        a, b = mixed_batch(args.batch, device, phase)
        left, right = inputs(a, b)
        logits = train_model(left, right)
        loss = F.cross_entropy(logits.flatten(0, 1), targets(a, b).flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 1000 == 0 or step == args.steps:
            er, nr = evaluate(model, 4, 8192, "random", device)
            ec, nc = evaluate(model, 2, 8192, "carry", device)
            ep, npat = evaluate(model, 2, 8192, "pattern", device)
            score = er + 2 * ec + ep
            elapsed = time.time() - started
            print(f"step {step:5d} loss {loss.item():.6f} lr {lr:g} "
                  f"random {er}/{nr} carry {ec}/{nc} pattern {ep}/{npat} "
                  f"elapsed {elapsed:.1f}s", flush=True)
            # Always save late training; tie-breaking intentionally favors later checkpoints.
            if score <= best_score or step >= int(args.steps * 0.84):
                best_score = min(best_score, score)
                torch.save(model.state_dict(), "/workspace/model.pt")
    torch.save(model.state_dict(), "/workspace/model_final.pt")

if __name__ == "__main__":
    main()
