import argparse
import random
import time

import torch
import torch.nn.functional as F

from submission import AdditionTransformer


DEVICE = "cuda"
POSITIONS = 15


def targets(a, b):
    out = torch.empty_like(a)
    carry = torch.zeros(a.shape[0], device=a.device, dtype=torch.long)
    for p in range(POSITIONS):
        total = a[:, p] + b[:, p] + carry
        out[:, p] = total.remainder(10)
        carry = total.div(10, rounding_mode="floor")
    return out


def random_batch(n, generator):
    a = torch.randint(10, (n, POSITIONS), device=DEVICE, generator=generator)
    b = torch.randint(10, (n, POSITIONS), device=DEVICE, generator=generator)
    a[:, 14] = 0
    b[:, 14] = 0
    return a, b, targets(a, b)


def structured_batch(n, generator, long_only=False):
    a, b, _ = random_batch(n, generator)
    rows = torch.arange(n, device=DEVICE)
    family = torch.randint(7, (n,), device=DEVICE, generator=generator)

    # Randomized runs whose columns sum to nine, seeded by an incoming carry.
    chain_rows = rows[(family == 0) | (family == 1) | long_only]
    if chain_rows.numel():
        count = chain_rows.numel()
        if long_only:
            start = torch.randint(0, 4, (count,), device=DEVICE, generator=generator)
            length = torch.randint(9, 15, (count,), device=DEVICE, generator=generator)
        else:
            start = torch.randint(0, 14, (count,), device=DEVICE, generator=generator)
            length = torch.randint(1, 15, (count,), device=DEVICE, generator=generator)
        end = torch.minimum(start + length, torch.full_like(start, 14))
        for j in range(14):
            active = (start <= j) & (j <= end)
            rr = chain_rows[active]
            if rr.numel():
                first = start[active] == j
                av = torch.randint(10, (rr.numel(),), device=DEVICE, generator=generator)
                complement = 9 - av
                # The first column creates carry; later sum-nine columns propagate it.
                bv = torch.where(first, torch.clamp(complement + torch.randint(1, 10, (rr.numel(),), device=DEVICE, generator=generator), max=9), complement)
                # Guarantee the seed column sums to at least ten.
                av = torch.where(first & (av + bv < 10), 9 - bv + 1, av)
                a[rr, j] = av
                b[rr, j] = bv

    # Sparse addends, including increments at arbitrary boundaries.
    rr = rows[family == 2]
    if rr.numel():
        b[rr] = 0
        p = torch.randint(0, 14, (rr.numel(),), device=DEVICE, generator=generator)
        b[rr, p] = torch.randint(1, 10, (rr.numel(),), device=DEVICE, generator=generator)

    # Repeated digit operands.
    rr = rows[family == 3]
    if rr.numel():
        da = torch.randint(10, (rr.numel(), 1), device=DEVICE, generator=generator)
        db = torch.randint(10, (rr.numel(), 1), device=DEVICE, generator=generator)
        a[rr, :14] = da
        b[rr, :14] = db

    # Complementary full or partial patterns.
    rr = rows[family == 4]
    if rr.numel():
        a[rr, :14] = torch.randint(10, (rr.numel(), 14), device=DEVICE, generator=generator)
        b[rr, :14] = 9 - a[rr, :14]
        flip = torch.randint(0, 14, (rr.numel(),), device=DEVICE, generator=generator)
        b[rr, flip] = torch.clamp(b[rr, flip] + torch.randint(-1, 2, (rr.numel(),), device=DEVICE, generator=generator), 0, 9)

    # Blockwise constant digits.
    rr = rows[family == 5]
    if rr.numel():
        for j in range(0, 14, 2):
            a[rr, j:j + 2] = torch.randint(10, (rr.numel(), 1), device=DEVICE, generator=generator)
            b[rr, j:j + 2] = torch.randint(10, (rr.numel(), 1), device=DEVICE, generator=generator)

    # Extreme and alternating patterns, with operand swaps naturally represented.
    rr = rows[family == 6]
    if rr.numel():
        choice = torch.randint(4, (rr.numel(), 1), device=DEVICE, generator=generator)
        a[rr, :14] = torch.where(choice % 2 == 0, 9, 0)
        b[rr, :14] = torch.where(choice < 2, 1, 9)
        odd = torch.arange(14, device=DEVICE).remainder(2).bool()
        alt = choice[:, 0] == 3
        if alt.any():
            ar = rr[alt]
            a[ar, :14] = odd.long() * 9
            b[ar, :14] = (~odd).long() * 9

    if torch.rand((), device=DEVICE, generator=generator) < .5:
        a, b = b, a
    return a, b, targets(a, b)


@torch.no_grad()
def evaluate(model, batches, batch_size, structured, seed):
    model.eval()
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    wrong = 0
    total = 0
    digit_wrong = 0
    for _ in range(batches):
        if structured:
            a, b, y = structured_batch(batch_size, generator)
        else:
            a, b, y = random_batch(batch_size, generator)
        pred = model(a, b).argmax(-1)
        bad = pred.ne(y)
        wrong += bad.any(1).sum().item()
        digit_wrong += bad.sum().item()
        total += batch_size
    model.train()
    return wrong, total, digit_wrong


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=25000)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1313)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    model = AdditionTransformer().to(DEVICE)
    raw_model = model
    model = torch.compile(model, mode="max-autotune")
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=3e-3, weight_decay=.005, fused=True)
    generator = torch.Generator(device=DEVICE).manual_seed(args.seed + 1)
    started = time.time()
    best = None

    for step in range(1, args.steps + 1):
        if step <= 16000:
            structured_fraction = .35
            lr = 3e-3 if step <= 7000 else (1e-3 if step <= 12000 else 3e-4)
            long_only = False
        elif step <= 21000:
            structured_fraction = .65
            lr = 1e-4
            long_only = True
        else:
            structured_fraction = .5
            lr = 5e-5
            long_only = False
        optimizer.param_groups[0]["lr"] = lr
        structured_n = int(args.batch * structured_fraction)
        random_n = args.batch - structured_n
        ra, rb, ry = random_batch(random_n, generator)
        sa, sb, sy = structured_batch(structured_n, generator, long_only)
        a = torch.cat((ra, sa))
        b = torch.cat((rb, sb))
        y = torch.cat((ry, sy))
        logits = model(a, b)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        optimizer.step()

        if step % 1000 == 0 or step == args.steps:
            random_result = evaluate(model, 16, 4096, False, 100000 + step)
            structured_result = evaluate(model, 16, 4096, True, 200000 + step)
            print(f"step={step} loss={loss.item():.6g} random={random_result} structured={structured_result} elapsed={time.time()-started:.1f}", flush=True)
            score = random_result[0] + 2 * structured_result[0]
            if step >= 12000 and (best is None or score < best):
                best = score
                torch.save(raw_model.state_dict(), "/workspace/model.pt")
                print(f"saved score={score}", flush=True)

    print("final large validation", flush=True)
    raw_model.load_state_dict(torch.load("/workspace/model.pt", weights_only=True))
    print("random", evaluate(raw_model, 256, 4096, False, 991337), flush=True)
    print("structured", evaluate(raw_model, 128, 4096, True, 881337), flush=True)


if __name__ == "__main__":
    main()
