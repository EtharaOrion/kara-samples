import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from submission import AdditionTransformer


DEVICE = "cuda"
LIMIT = 100_000_000_000_000
POW10 = torch.tensor([10**i for i in range(15)], device=DEVICE, dtype=torch.long)


def columns(values, count=14):
    return (values[:, None] // POW10[None, :count]) % 10


def make_pairs(batch, structured=0.25):
    a = torch.randint(LIMIT, (batch,), device=DEVICE)
    b = torch.randint(LIMIT, (batch,), device=DEVICE)
    amount = int(batch * structured)
    if not amount:
        return a, b
    kind = torch.randint(6, (amount,), device=DEVICE)
    starts = torch.randint(14, (amount,), device=DEVICE)
    lengths = torch.minimum(torch.randint(1, 15, (amount,), device=DEVICE), 14 - starts)
    p = POW10[starts]
    run = (POW10[lengths] - 1) * p

    # Exact carry runs, with arbitrary independent higher and lower context.
    mask = kind == 0
    a[:amount][mask] = run[mask]
    b[:amount][mask] = p[mask]

    # Near-carry contrasts: long 9-runs that must not propagate.
    mask = kind == 1
    a[:amount][mask] = run[mask] - p[mask]
    b[:amount][mask] = p[mask]

    # Sparse increments into otherwise random operands.
    mask = kind == 2
    b[:amount][mask] = p[mask]

    # Complementary low runs causing carries of varied lengths.
    mask = kind == 3
    cap = POW10[lengths[mask]]
    low = 1 + a[:amount][mask] % (cap - 1)
    a[:amount][mask] = low * p[mask]
    b[:amount][mask] = (cap - low) * p[mask]

    # Repeated digits.
    mask = kind == 4
    digit_a = torch.randint(10, (int(mask.sum()),), device=DEVICE)
    digit_b = torch.randint(10, (int(mask.sum()),), device=DEVICE)
    rep = (POW10[14] - 1) // 9
    a[:amount][mask] = digit_a * rep
    b[:amount][mask] = digit_b * rep

    # Boundary values and full-width overflow.
    mask = kind == 5
    count = int(mask.sum())
    high = torch.randint(1, 10, (count,), device=DEVICE)
    a[:amount][mask] = LIMIT - high
    b[:amount][mask] = high
    return a, b


def batch_tokens(a, b):
    batch = a.shape[0]
    answer = columns(a + b, 15)
    sentinel = torch.full((batch, 1), 10, device=DEVICE)
    operand_a = torch.cat((columns(a), sentinel, torch.full((batch, 14), 10, device=DEVICE)), 1)
    operand_b = torch.cat((columns(b), sentinel, torch.full((batch, 14), 10, device=DEVICE)), 1)
    output = torch.cat((torch.full((batch, 15), 10, device=DEVICE), answer[:, :-1]), 1)
    return operand_a, operand_b, output, answer


@torch.inference_mode()
def evaluate(model, total, structured=0.0, chunk=8192):
    model.eval()
    errors = 0
    digits_wrong = 0
    for _ in range(math.ceil(total / chunk)):
        size = min(chunk, total)
        total -= size
        a, b = make_pairs(size, structured)
        target = columns(a + b, 15)
        operand_a = torch.cat((columns(a), torch.full((size, 1), 10, device=DEVICE)), 1)
        operand_b = torch.cat((columns(b), torch.full((size, 1), 10, device=DEVICE)), 1)
        output = torch.full((size, 15), 10, device=DEVICE)
        prediction = []
        for _place in range(15):
            digit = model(operand_a, operand_b, output)[:, -1].argmax(1)
            prediction.append(digit)
            operand_a = torch.cat((operand_a, torch.full((size, 1), 10, device=DEVICE)), 1)
            operand_b = torch.cat((operand_b, torch.full((size, 1), 10, device=DEVICE)), 1)
            output = torch.cat((output, digit[:, None]), 1)
        prediction = torch.stack(prediction, 1)
        wrong = prediction != target
        errors += int(wrong.any(1).sum())
        digits_wrong += int(wrong.sum())
    model.train()
    return errors, digits_wrong


def main():
    torch.manual_seed(2301)
    random.seed(2301)
    torch.set_float32_matmul_precision("high")
    model = AdditionTransformer().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.98), weight_decay=0.003, fused=True)
    batch = 4096
    phases = [(5000, 3e-3, 0.20), (5000, 1e-3, 0.30), (4000, 3e-4, 0.40), (3000, 1e-4, 0.40)]
    step = 0
    started = time.time()
    best = None
    for steps, lr, structured in phases:
        for group in optimizer.param_groups:
            group["lr"] = lr
        for _ in range(steps):
            step += 1
            a, b = make_pairs(batch, structured)
            operand_a, operand_b, output, answer = batch_tokens(a, b)
            logits = model(operand_a, operand_b, output)[:, 14:]
            loss = F.cross_entropy(logits.reshape(-1, 10), answer.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % 1000 == 0:
                elapsed = time.time() - started
                print(f"step={step} loss={loss.item():.6g} elapsed={elapsed:.1f}s", flush=True)
            if step % 4000 == 0:
                random_err, _ = evaluate(model, 65536, 0.0)
                struct_err, _ = evaluate(model, 32768, 1.0)
                print(f"validation step={step} random={random_err}/65536 structured={struct_err}/32768", flush=True)
                score = random_err * 4 + struct_err
                if best is None or score <= best:
                    best = score
                    torch.save(model.state_dict(), "/workspace/model.pt")
                    print("saved", flush=True)
    torch.save(model.state_dict(), "/workspace/model_final.pt")
    model.load_state_dict(torch.load("/workspace/model.pt", weights_only=True))
    print("final random", evaluate(model, 262144, 0.0), flush=True)
    print("final structured", evaluate(model, 131072, 1.0), flush=True)


if __name__ == "__main__":
    main()
