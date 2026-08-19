import argparse
import os
import time

import torch
import torch.nn.functional as F

from submission import AdditionTransformer


MAXIMUM = 100_000_000_000_000
POWERS = torch.tensor([10**i for i in range(15)], device="cuda", dtype=torch.long)


def random_batch(batch, structured=0.0):
    a = torch.randint(MAXIMUM, (batch,), device="cuda")
    b = torch.randint(MAXIMUM, (batch,), device="cuda")
    left = (a[:, None] // POWERS % 10).long()
    right = (b[:, None] // POWERS % 10).long()
    if structured:
        count = int(batch * structured)
        rows = torch.arange(count, device="cuda")
        family = torch.randint(5, (count,), device="cuda")
        start = torch.randint(14, (count,), device="cuda")
        length = torch.minimum(torch.randint(1, 15, (count,), device="cuda"), 14 - start)
        columns = torch.arange(15, device="cuda")[None]
        active = (columns >= start[:, None]) & (columns <= (start + length)[:, None])
        first = columns == start[:, None]

        carry = family <= 1
        carry_rows = rows[carry]
        if len(carry_rows):
            sparse_carry_rows = rows[family == 0]
            left[sparse_carry_rows] = 0
            right[sparse_carry_rows] = 0
            cfirst = first[carry]
            crest = active[carry] & ~cfirst
            x = torch.randint(1, 10, (len(carry_rows), 15), device="cuda")
            left[carry_rows] = torch.where(cfirst, x, left[carry_rows])
            right[carry_rows] = torch.where(cfirst, 10 - x, right[carry_rows])
            y = torch.randint(10, (len(carry_rows), 15), device="cuda")
            left[carry_rows] = torch.where(crest, y, left[carry_rows])
            right[carry_rows] = torch.where(crest, 9 - y, right[carry_rows])

        near = family == 2
        near_rows = rows[near]
        if len(near_rows):
            mask = active[near]
            x = torch.randint(10, (len(near_rows), 15), device="cuda")
            total = torch.randint(8, 10, (len(near_rows), 15), device="cuda")
            x = torch.minimum(x, total)
            left[near_rows] = torch.where(mask, x, left[near_rows])
            right[near_rows] = torch.where(mask, total - x, right[near_rows])

        sparse = family == 3
        sparse_rows = rows[sparse]
        if len(sparse_rows):
            left[sparse_rows] = 0
            right[sparse_rows] = 0
            col = start[sparse]
            values = torch.where(torch.rand(len(sparse_rows), device="cuda") < .5, 5, 9)
            left[sparse_rows, col] = values
            right[sparse_rows, col] = values

        repeated = family == 4
        repeated_rows = rows[repeated]
        if len(repeated_rows):
            x = torch.randint(10, (len(repeated_rows), 1), device="cuda")
            y = torch.randint(10, (len(repeated_rows), 1), device="cuda")
            left[repeated_rows, :14] = x
            right[repeated_rows, :14] = y
            left[repeated_rows, 14] = 0
            right[repeated_rows, 14] = 0

    left[:, 14] = 0
    right[:, 14] = 0
    carry = torch.zeros(batch, device="cuda", dtype=torch.long)
    target = torch.empty_like(left)
    for column in range(15):
        total = left[:, column] + right[:, column] + carry
        target[:, column] = total % 10
        carry = total // 10
    return left, right, target


@torch.no_grad()
def validate(model, batches=16, batch=8192, structured=0.0):
    model.eval()
    errors = 0
    total = 0
    for _ in range(batches):
        a, b, target = random_batch(batch, structured)
        prediction = model(a, b).argmax(-1)
        errors += (prediction != target).any(-1).sum().item()
        total += batch
    model.train()
    return errors, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120000)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(44)
    torch.set_float32_matmul_precision("high")
    model = AdditionTransformer().cuda()
    for parameter in model.parameters():
        if parameter.dim() > 1:
            torch.nn.init.normal_(parameter, std=.02)
    checkpoint = "/workspace/model.pt"
    start = 0
    if args.resume and os.path.exists(checkpoint):
        state = torch.load(checkpoint, weights_only=True)
        model.load_state_dict(state["model"])
        start = state["step"]
    print("parameters", sum(p.numel() for p in model.parameters()), "start", start, flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=.002, fused=True)
    if args.resume and os.path.exists(checkpoint) and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    compiled = model
    began = time.time()
    best = 10**9
    for step in range(start + 1, args.steps + 1):
        if step <= 12000:
            lr, fraction = 3e-3, 0.0
        elif step <= 32000:
            lr, fraction = 1e-3, .25
        elif step <= 60000:
            lr, fraction = 3e-4, .40
        elif step <= 90000:
            lr, fraction = 1e-4, .45
        else:
            lr, fraction = 3e-5, .45
        for group in optimizer.param_groups:
            group["lr"] = lr
        left, right, target = random_batch(args.batch, fraction)
        logits = compiled(left, right)
        loss = F.cross_entropy(logits.reshape(-1, 10), target.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 2000 == 0 or step == args.steps:
            errors, total = validate(model, 8)
            structured_errors, structured_total = validate(model, 8, structured=.75)
            elapsed = time.time() - began
            print(step, f"loss={loss.item():.5g}", f"random={errors}/{total}",
                  f"structured={structured_errors}/{structured_total}", f"seconds={elapsed:.1f}", flush=True)
            score = errors + structured_errors
            if score <= best:
                best = score
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "step": step, "score": score}, checkpoint)


if __name__ == "__main__":
    main()
