import argparse
import importlib.util
import math
import random
import re
import time
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
MAXIMUM = 100_000_000_000_000
POWERS = torch.tensor([10 ** i for i in range(15)], dtype=torch.long)

spec = importlib.util.spec_from_file_location("submission", ROOT / "submission.py")
submission = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submission)


def integer_digits(values, places=15):
    powers = POWERS[:places].to(values.device)
    return values[:, None].div(powers, rounding_mode="floor").remainder(10)


def structured_digits(n, device):
    da = torch.randint(0, 10, (n, 14), device=device)
    db = torch.randint(0, 10, (n, 14), device=device)
    rows = torch.arange(n, device=device)
    kind = torch.randint(0, 8, (n,), device=device)
    start = torch.randint(0, 14, (n,), device=device)
    length = torch.randint(1, 15, (n,), device=device)
    end = torch.minimum(start + length, torch.full_like(start, 14))

    # Carry chains: initiating column >=10, following columns exactly 9.
    mask = kind <= 2
    r = rows[mask]
    s = start[mask]
    if r.numel():
        left = torch.randint(1, 10, (r.numel(),), device=device)
        da[r, s] = left
        db[r, s] = 10 - left + torch.randint(0, 10, (r.numel(),), device=device).remainder(left)
        positions = torch.arange(14, device=device)[None]
        chain = (positions >= s[:, None] + 1) & (positions < end[mask, None])
        vals = torch.randint(0, 10, (r.numel(), 14), device=device)
        da[r[:, None], positions.expand(r.numel(), -1)] = torch.where(chain, vals, da[r])
        db[r[:, None], positions.expand(r.numel(), -1)] = torch.where(chain, 9 - vals, db[r])
        stops = end[mask]
        valid = stops < 14
        rr, ss = r[valid], stops[valid]
        if rr.numel():
            x = torch.randint(0, 9, (rr.numel(),), device=device)
            da[rr, ss] = x
            db[rr, ss] = torch.randint(0, 9, (rr.numel(),), device=device).remainder(9 - x)

    # Matched non-carry runs are close counterexamples to carry propagation.
    mask = kind == 3
    r = rows[mask]
    s = start[mask]
    if r.numel():
        positions = torch.arange(14, device=device)[None]
        run = (positions >= s[:, None]) & (positions < end[mask, None])
        vals = torch.randint(0, 10, (r.numel(), 14), device=device)
        da[r[:, None], positions.expand(r.numel(), -1)] = torch.where(run, vals, da[r])
        db[r[:, None], positions.expand(r.numel(), -1)] = torch.where(run, 9 - vals, db[r])

    # Sparse isolated boundary columns, emphasizing 5+5 and 9+9 at all heights.
    mask = kind == 4
    r, s = rows[mask], start[mask]
    if r.numel():
        da[r] = 0
        db[r] = 0
        choice = torch.randint(0, 4, (r.numel(),), device=device)
        da[r, s] = torch.where(choice < 2, 5, torch.where(choice == 2, 9, torch.randint(0, 10, (r.numel(),), device=device)))
        db[r, s] = torch.where(choice < 2, 5, torch.where(choice == 2, 9, torch.randint(0, 10, (r.numel(),), device=device)))

    # Complements, repeated digits, and block patterns.
    mask = kind == 5
    r = rows[mask]
    if r.numel():
        db[r] = 9 - da[r]
        s = start[mask]
        left = torch.randint(1, 10, (r.numel(),), device=device)
        da[r, s] = left
        db[r, s] = 10 - left
    mask = kind == 6
    r = rows[mask]
    if r.numel():
        da[r] = torch.randint(0, 10, (r.numel(), 1), device=device)
        db[r] = torch.randint(0, 10, (r.numel(), 1), device=device)
    mask = kind == 7
    r = rows[mask]
    if r.numel():
        da[r] = 0
        db[r] = 0
        p = torch.arange(14, device=device)[None]
        run = (p >= start[mask, None]) & (p < end[mask, None])
        da[r[:, None], p.expand(r.numel(), -1)] = torch.where(run, 9, da[r])
        db[r[:, None], p.expand(r.numel(), -1)] = torch.where(run, 9, db[r])
    return da, db


def batch_data(batch, device, structured_fraction):
    uniform = batch - int(batch * structured_fraction)
    a = torch.randint(0, MAXIMUM, (uniform,), device=device)
    b = torch.randint(0, MAXIMUM, (uniform,), device=device)
    da = integer_digits(a, 14)
    db = integer_digits(b, 14)
    rest = batch - uniform
    if rest:
        sa, sb = structured_digits(rest, device)
        da = torch.cat((da, sa))
        db = torch.cat((db, sb))
        powers = POWERS[:14].to(device)
        a = torch.cat((a, (sa * powers).sum(1)))
        b = torch.cat((b, (sb * powers).sum(1)))
    targets = integer_digits(a + b, 15)
    return da, db, targets


@torch.no_grad()
def evaluate(model, count, mode="uniform", batch=8192):
    model.eval()
    errors = 0
    digits_wrong = 0
    for begin in range(0, count, batch):
        n = min(batch, count - begin)
        da, db, target = batch_data(n, "cuda", 0.0 if mode == "uniform" else 1.0)
        previous = torch.empty((n, 0), dtype=torch.long, device="cuda")
        for _ in range(15):
            digit = model(da, db, previous)[:, -1].argmax(-1, keepdim=True)
            previous = torch.cat((previous, digit), 1)
        bad = previous.ne(target)
        errors += bad.any(1).sum().item()
        digits_wrong += bad.sum().item()
    model.train()
    return errors, digits_wrong


def export(model):
    flat = torch.cat([p.detach().float().cpu().reshape(-1) for p in model.parameters()])
    values = ",".join(format(x, ".9g") for x in flat.tolist())
    path = ROOT / "submission.py"
    text = path.read_text()
    text = re.sub(r"_FLAT = \[[^\]]*\]", "_FLAT = [" + values + "]", text, count=1)
    path.write_text(text)
    print("exported", flat.numel(), "parameters to", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=96000)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(3701)
    random.seed(3701)
    torch.set_float32_matmul_precision("high")
    model = submission.AdditionTransformer().cuda().train()
    checkpoint = ROOT / "checkpoint.pt"
    start = 0
    if args.resume and checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cuda", weights_only=True)
        model.load_state_dict(saved["model"])
        start = saved["step"]
    print("parameters", sum(p.numel() for p in model.parameters()), "start", start)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.002, fused=True)
    schedule = [(12000, 3e-3, 0.0), (30000, 1e-3, 0.35), (52000, 3e-4, 0.48),
                (72000, 1e-4, 0.48), (88000, 3e-5, 0.48), (args.steps, 1e-5, 0.48)]
    then = time.time()
    for step in range(start, args.steps):
        for limit, lr, fraction in schedule:
            if step < limit:
                break
        for group in optimizer.param_groups:
            group["lr"] = lr
        da, db, target = batch_data(args.batch, "cuda", fraction)
        logits = model(da, db, target[:, :14])
        loss = F.cross_entropy(logits.reshape(-1, 10), target.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        if (step + 1) % 1000 == 0:
            elapsed = time.time() - then
            with torch.no_grad():
                accuracy = logits.argmax(-1).eq(target).float().mean().item()
            print(step + 1, f"loss={loss.item():.6g}", f"tf={accuracy:.7f}",
                  f"lr={lr:g}", f"{elapsed:.1f}s", flush=True)
            then = time.time()
        if (step + 1) % 8000 == 0:
            torch.save({"model": model.state_dict(), "step": step + 1}, checkpoint)
    torch.save({"model": model.state_dict(), "step": args.steps}, checkpoint)
    for mode in ("uniform", "structured"):
        result = evaluate(model, 131072, mode)
        print(mode, result, flush=True)
    export(model)


if __name__ == "__main__":
    main()
