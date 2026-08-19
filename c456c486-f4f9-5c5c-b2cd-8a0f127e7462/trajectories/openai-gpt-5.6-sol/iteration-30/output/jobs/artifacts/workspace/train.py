import os
import time
import torch
import torch.nn.functional as F

from submission import AdditionTransformer


DEVICE = "cuda"
BATCH = 4096
DIGITS = 14


def targets(a, b):
    out = torch.empty((a.shape[0], 15), dtype=torch.long, device=a.device)
    carry = torch.zeros(a.shape[0], dtype=torch.long, device=a.device)
    for column in range(DIGITS):
        total = a[:, column] + b[:, column] + carry
        out[:, column] = total.remainder(10)
        carry = total.div(10, rounding_mode="floor")
    out[:, 14] = carry
    return out


def make_batch(size, structured_fraction=0.4):
    a = torch.randint(10, (size, DIGITS), device=DEVICE)
    b = torch.randint(10, (size, DIGITS), device=DEVICE)
    count = int(size * structured_fraction)
    if not count:
        return a, b, targets(a, b)
    groups = torch.arange(count, device=DEVICE).remainder(7)

    # Randomized initiated carry chains: a+b >= 10, followed by columns summing to 9.
    rows = torch.where(groups == 0)[0]
    if rows.numel():
        start = torch.randint(0, DIGITS, (rows.numel(),), device=DEVICE)
        length = torch.randint(1, DIGITS + 1, (rows.numel(),), device=DEVICE)
        col = torch.arange(DIGITS, device=DEVICE)[None]
        first = col == start[:, None]
        continuation = (col > start[:, None]) & (col < (start + length)[:, None])
        av = torch.randint(1, 10, (rows.numel(), DIGITS), device=DEVICE)
        bv = 10 - av
        a[rows] = torch.where(first, av, a[rows])
        b[rows] = torch.where(first, bv, b[rows])
        av = torch.randint(0, 10, (rows.numel(), DIGITS), device=DEVICE)
        bv = 9 - av
        a[rows] = torch.where(continuation, av, a[rows])
        b[rows] = torch.where(continuation, bv, b[rows])

    # Matched non-carry 9-runs, with lower columns cleared to exclude incoming carry.
    rows = torch.where(groups == 1)[0]
    if rows.numel():
        start = torch.randint(0, DIGITS, (rows.numel(),), device=DEVICE)
        length = torch.randint(1, DIGITS + 1, (rows.numel(),), device=DEVICE)
        col = torch.arange(DIGITS, device=DEVICE)[None]
        below = col < start[:, None]
        run = (col >= start[:, None]) & (col < (start + length)[:, None])
        av = torch.randint(0, 10, (rows.numel(), DIGITS), device=DEVICE)
        a[rows] = torch.where(below, 0, a[rows])
        b[rows] = torch.where(below, 0, b[rows])
        a[rows] = torch.where(run, av, a[rows])
        b[rows] = torch.where(run, 9 - av, b[rows])

    # Sparse exact boundaries, heavily covering the historically difficult 5+5 case.
    rows = torch.where(groups == 2)[0]
    if rows.numel():
        a[rows] = 0
        b[rows] = 0
        pos = torch.randint(0, DIGITS, (rows.numel(),), device=DEVICE)
        value = torch.randint(1, 10, (rows.numel(),), device=DEVICE)
        value[::2] = 5
        a[rows, pos] = value
        b[rows, pos] = 10 - value
        extra = torch.randint(0, DIGITS, (rows.numel(),), device=DEVICE)
        b[rows, extra] = torch.where(extra == pos, b[rows, extra], torch.randint(0, 10, (rows.numel(),), device=DEVICE))

    # Long 9-runs incremented by one at a random boundary.
    rows = torch.where(groups == 3)[0]
    if rows.numel():
        a[rows] = 0
        b[rows] = 0
        start = torch.randint(0, DIGITS, (rows.numel(),), device=DEVICE)
        end = start + torch.randint(1, DIGITS + 1, (rows.numel(),), device=DEVICE)
        end.clamp_(max=DIGITS)
        col = torch.arange(DIGITS, device=DEVICE)[None]
        run = (col >= start[:, None]) & (col < end[:, None])
        a[rows] = torch.where(run, 9, a[rows])
        b[rows, start] = 1

    # Repeated operands and block patterns.
    rows = torch.where(groups == 4)[0]
    if rows.numel():
        da = torch.randint(0, 10, (rows.numel(), 1), device=DEVICE)
        db = torch.randint(0, 10, (rows.numel(), 1), device=DEVICE)
        a[rows] = da
        b[rows] = db
        split = torch.randint(1, DIGITS, (rows.numel(),), device=DEVICE)
        col = torch.arange(DIGITS, device=DEVICE)[None]
        a[rows] = torch.where(col >= split[:, None], torch.randint(0, 10, (rows.numel(), 1), device=DEVICE), a[rows])

    # Complementary columns include exact carry and non-carry boundary contrasts.
    rows = torch.where(groups == 5)[0]
    if rows.numel():
        av = torch.randint(0, 10, (rows.numel(), DIGITS), device=DEVICE)
        sums = torch.where(torch.rand((rows.numel(), DIGITS), device=DEVICE) < 0.5, 9, 10)
        av = torch.minimum(av, sums)
        a[rows] = av
        b[rows] = sums - av

    # Maximum overflow and top-position carry boundaries.
    rows = torch.where(groups == 6)[0]
    if rows.numel():
        a[rows] = 9
        b[rows] = 0
        pos = torch.randint(0, DIGITS, (rows.numel(),), device=DEVICE)
        b[rows, pos] = 1
        half = rows[::2]
        if half.numel():
            a[half, :13] = torch.randint(0, 10, (half.numel(), 13), device=DEVICE)
            a[half, 13] = 5
            b[half, 13] = 5

    return a, b, targets(a, b)


def logits_for(model, a, b, answer):
    pad = torch.full((a.shape[0], 1), 10, dtype=torch.long, device=DEVICE)
    aa = torch.cat((a, pad), 1)
    bb = torch.cat((b, pad), 1)
    previous = torch.cat((pad, answer[:, :-1]), 1)
    return model(aa, bb, previous)[:, 15:]


@torch.no_grad()
def evaluate(model, batches, structured_fraction=0.0, batch_size=16384):
    model.eval()
    errors = 0
    total = 0
    for _ in range(batches):
        a, b, expected = make_batch(batch_size, structured_fraction)
        pad = torch.full((batch_size, 1), 10, dtype=torch.long, device=DEVICE)
        aa = torch.cat((a, pad), 1)
        bb = torch.cat((b, pad), 1)
        previous = pad
        generated = []
        for __ in range(15):
            digit = model(aa, bb, previous)[:, -1].argmax(-1, keepdim=True)
            generated.append(digit)
            previous = torch.cat((previous, digit), 1)
        predicted = torch.cat(generated, 1)
        errors += (predicted != expected).any(1).sum().item()
        total += batch_size
    model.train()
    return errors, total


def main():
    torch.manual_seed(30030)
    model = AdditionTransformer().to(DEVICE)
    print("parameters", sum(p.numel() for p in model.parameters()), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.98), weight_decay=0.005)
    start_step = 0
    if os.path.exists("/workspace/checkpoint.pt"):
        checkpoint = torch.load("/workspace/checkpoint.pt", map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"]
        print("resuming", start_step, flush=True)

    phases = [(8000, 3e-3, 0.30), (16000, 1e-3, 0.40), (24000, 3e-4, 0.45), (30000, 1e-4, 0.50)]
    begin = time.time()
    for step in range(start_step, phases[-1][0]):
        for end, lr, fraction in phases:
            if step < end:
                break
        for group in optimizer.param_groups:
            group["lr"] = lr
        a, b, answer = make_batch(BATCH, fraction)
        logits = logits_for(model, a, b, answer)
        loss = F.cross_entropy(logits.reshape(-1, 10), answer.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        done = step + 1
        if done % 1000 == 0:
            elapsed = time.time() - begin
            print(done, f"loss={loss.item():.6g}", f"lr={lr:g}", f"sec={elapsed:.1f}", flush=True)
        if done % 5000 == 0:
            error, total = evaluate(model, 4, 0.0)
            structured_error, structured_total = evaluate(model, 4, 1.0)
            print("eval", done, error, "/", total, "structured", structured_error, "/", structured_total, flush=True)
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": done}, "/workspace/checkpoint.pt")

    error, total = evaluate(model, 32, 0.0)
    structured_error, structured_total = evaluate(model, 32, 1.0)
    print("FINAL", error, "/", total, "structured", structured_error, "/", structured_total, flush=True)
    torch.save(model.state_dict(), "/workspace/model.pt")


if __name__ == "__main__":
    main()
