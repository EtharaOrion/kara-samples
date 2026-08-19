import random
import os
import time
import torch
import torch.nn.functional as F
from submission import AdditionTransformer

DEVICE = "cuda"
torch.manual_seed(927)
random.seed(927)
torch.set_float32_matmul_precision("high")


def batch_data(batch, structured=0.0):
    a = torch.randint(0, 10, (batch, 14), device=DEVICE)
    b = torch.randint(0, 10, (batch, 14), device=DEVICE)
    if structured and random.random() < structured:
        kind = random.randrange(3)
        if kind == 0:
            a[:] = torch.randint(0, 10, (batch, 1), device=DEVICE)
            b[:] = torch.randint(0, 10, (batch, 1), device=DEVICE)
        elif kind == 1:
            start = random.randrange(0, 10)
            length = random.randrange(3, 15 - start)
            a[:, start:start + length] = 9
            b[:, start] = torch.randint(1, 10, (batch,), device=DEVICE)
        else:
            a[:, ::2] = 9
            b[:, 1::2] = 9
    operands = torch.zeros((batch, 15, 2), dtype=torch.long, device=DEVICE)
    operands[:, :14, 0] = a
    operands[:, :14, 1] = b
    target = torch.empty((batch, 15), dtype=torch.long, device=DEVICE)
    carry = torch.zeros(batch, dtype=torch.long, device=DEVICE)
    for place in range(14):
        total = a[:, place] + b[:, place] + carry
        target[:, place] = total.remainder(10)
        carry = total.div(10, rounding_mode="floor")
    target[:, 14] = carry
    return operands, target


@torch.no_grad()
def evaluate(model, batches=20, batch=1000):
    model.eval()
    exact = digits = total = 0
    for _ in range(batches):
        x, y = batch_data(batch)
        pred = model(x).argmax(-1)
        exact += pred.eq(y).all(1).sum().item()
        digits += pred.eq(y).sum().item()
        total += batch
    model.train()
    return exact / total, digits / (15 * total)


def main():
    width = int(os.environ.get("WIDTH", "16"))
    rounds = int(os.environ.get("ROUNDS", "6"))
    model = AdditionTransformer(width=width, rounds=rounds).to(DEVICE)
    print("parameters", sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.003)
    steps = int(os.environ.get("STEPS", "12000"))
    started = time.time()
    best = 0.0
    for step in range(1, steps + 1):
        if step == int(steps * .35):
            for group in optimizer.param_groups: group["lr"] = 1e-3
        if step == int(steps * .60):
            for group in optimizer.param_groups: group["lr"] = 3e-4
        if step == int(steps * .80):
            for group in optimizer.param_groups: group["lr"] = 1e-4
        x, y = batch_data(4096, structured=0.35)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, 10), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 250 == 0:
            accuracy, digit_accuracy = evaluate(model, 10)
            print(step, f"loss={loss.item():.5f}", f"exact={accuracy:.5f}", f"digit={digit_accuracy:.7f}", f"seconds={time.time()-started:.1f}", flush=True)
            if accuracy >= best:
                best = accuracy
                torch.save(model.state_dict(), f"/workspace/model_w{width}_r{rounds}.pt")
    print("final", evaluate(model, 100), "best", best)


if __name__ == "__main__":
    main()
