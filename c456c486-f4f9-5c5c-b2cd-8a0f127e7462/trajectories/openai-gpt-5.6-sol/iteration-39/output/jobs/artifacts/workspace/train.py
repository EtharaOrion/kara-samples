import argparse
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace")
from submission import AdditionTransformer

DIGITS = 14
MAX_VALUE = 100_000_000_000_000


def numbers_to_digits(values, count=15):
    columns = []
    work = values
    for _ in range(count):
        columns.append(work.remainder(10))
        work = torch.div(work, 10, rounding_mode="floor")
    return torch.stack(columns, dim=1).long()


def digits_to_numbers(digits, powers):
    return (digits * powers[:digits.shape[1]]).sum(dim=1)


def uniform_batch(batch, device, powers):
    a = torch.randint(MAX_VALUE, (batch,), device=device)
    b = torch.randint(MAX_VALUE, (batch,), device=device)
    return make_tokens(a, b)


def make_tokens(a, b):
    ad = numbers_to_digits(a, DIGITS)
    bd = numbers_to_digits(b, DIGITS)
    sentinel = torch.full((a.shape[0], 1), 10, dtype=torch.long, device=a.device)
    return torch.cat((ad, sentinel), 1), torch.cat((bd, sentinel), 1), numbers_to_digits(a + b, 15)


def structured_batch(batch, device, powers, mode=None):
    ad = torch.randint(10, (batch, DIGITS), device=device)
    bd = torch.randint(10, (batch, DIGITS), device=device)
    rows = torch.arange(batch, device=device)
    family = torch.randint(6, (batch,), device=device) if mode is None else torch.full((batch,), mode, device=device)

    # Carry chains and matched non-carry chains at arbitrary positions.
    start = torch.randint(DIGITS, (batch,), device=device)
    max_len = DIGITS - start
    length = 1 + (torch.rand(batch, device=device) * max_len).long()
    col = torch.arange(DIGITS, device=device).unsqueeze(0)
    active = (col >= start[:, None]) & (col < (start + length)[:, None])
    continuation = active & (col > start[:, None])
    carry_rows = family == 0
    first_a = torch.randint(1, 10, (batch,), device=device)
    first_b = 10 - first_a + torch.randint(0, 10, (batch,), device=device)
    first_b.clamp_(max=9)
    ad[rows[carry_rows], start[carry_rows]] = first_a[carry_rows]
    bd[rows[carry_rows], start[carry_rows]] = first_b[carry_rows]
    ca = torch.randint(10, (batch, DIGITS), device=device)
    cb = 9 - ca
    mask = continuation & carry_rows[:, None]
    ad[mask], bd[mask] = ca[mask], cb[mask]

    non_rows = family == 1
    na = torch.randint(10, (batch, DIGITS), device=device)
    nb = 9 - na
    mask = active & non_rows[:, None]
    ad[mask], bd[mask] = na[mask], nb[mask]

    # Sparse exact boundaries, heavily including 5+5 and 9+9.
    sparse = family == 2
    ad[sparse] = 0
    bd[sparse] = 0
    pos = torch.randint(DIGITS, (batch,), device=device)
    choice = torch.randint(4, (batch,), device=device)
    sa = torch.where(choice < 2, torch.full_like(choice, 5), torch.where(choice == 2, torch.full_like(choice, 9), torch.randint(10, (batch,), device=device)))
    sb = torch.where(choice < 2, torch.full_like(choice, 5), torch.where(choice == 2, torch.full_like(choice, 9), 9 - sa))
    ad[rows[sparse], pos[sparse]] = sa[sparse]
    bd[rows[sparse], pos[sparse]] = sb[sparse]

    # Shifted all-nine runs plus one, the explicit late calibration family.
    nine = family == 3
    ad[nine] = 0
    bd[nine] = 0
    run_start = torch.randint(DIGITS, (batch,), device=device)
    run_max = DIGITS - run_start
    run_len = 1 + (torch.rand(batch, device=device) * run_max).long()
    run_mask = (col >= run_start[:, None]) & (col < (run_start + run_len)[:, None]) & nine[:, None]
    ad[run_mask] = 9
    bd[rows[nine], run_start[nine]] = 1

    # Repeated/block patterns.
    repeated = family == 4
    ra = torch.randint(10, (batch, 1), device=device).expand(-1, DIGITS)
    rb = torch.randint(10, (batch, 1), device=device).expand(-1, DIGITS)
    ad[repeated], bd[repeated] = ra[repeated], rb[repeated]

    # Complementary random columns, a mixture of exact 9 and 10 boundaries.
    comp = family == 5
    xa = torch.randint(10, (batch, DIGITS), device=device)
    carry_bit = torch.randint(2, (batch, DIGITS), device=device)
    xb = (9 + carry_bit - xa).clamp(0, 9)
    ad[comp], bd[comp] = xa[comp], xb[comp]

    a = digits_to_numbers(ad, powers)
    b = digits_to_numbers(bd, powers)
    return make_tokens(a, b)


def mixed_batch(batch, device, powers, structured_fraction, all_nines_fraction=0.0):
    uniform_count = int(batch * (1.0 - structured_fraction))
    structured_count = batch - uniform_count
    parts = []
    if uniform_count:
        parts.append(uniform_batch(uniform_count, device, powers))
    if structured_count:
        if all_nines_fraction and random.random() < all_nines_fraction:
            parts.append(structured_batch(structured_count, device, powers, mode=3))
        else:
            parts.append(structured_batch(structured_count, device, powers))
    if len(parts) == 1:
        return parts[0]
    return tuple(torch.cat(items, 0) for items in zip(*parts))


def loss_for(model, batch):
    ad, bd, target = batch
    logits = model(ad, bd, target[:, :-1])[:, 14:]
    return F.cross_entropy(logits.reshape(-1, 10), target.reshape(-1))


@torch.no_grad()
def autoregressive_errors(model, batches, device, powers, kind="uniform"):
    model.eval()
    errors = 0
    examples = 0
    for _ in range(batches):
        data = uniform_batch(4096, device, powers) if kind == "uniform" else structured_batch(4096, device, powers)
        ad, bd, target = data
        generated = torch.empty((ad.shape[0], 0), dtype=torch.long, device=device)
        for _ in range(15):
            generated = torch.cat((generated, model(ad, bd, generated)[:, -1].argmax(-1, keepdim=True)), 1)
        errors += (generated != target).any(1).sum().item()
        examples += ad.shape[0]
    model.train()
    return errors, examples


def export_submission(model, path):
    template = Path("/workspace/submission.py").read_text()
    marker = "\n_TRAINED_WEIGHTS = ["
    if marker in template:
        template = template.split(marker)[0].rstrip() + "\n"
    values = torch.cat([p.detach().float().cpu().reshape(-1) for p in model.parameters()]).tolist()
    literal = ",".join(format(v, ".9g") for v in values)
    loader = f"\n_TRAINED_WEIGHTS = [{literal}]\n\n_original_build_model = build_model\ndef build_model():\n    model, metadata = _original_build_model()\n    offset = 0\n    with torch.no_grad():\n        for parameter in model.parameters():\n            count = parameter.numel()\n            parameter.copy_(torch.tensor(_TRAINED_WEIGHTS[offset:offset + count], dtype=parameter.dtype).view_as(parameter))\n            offset += count\n    return model, metadata\n"
    Path(path).write_text(template + loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=106000)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=39)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda")
    powers = (10 ** torch.arange(DIGITS, device=device, dtype=torch.long))
    model = AdditionTransformer().to(device)
    checkpoint = Path("/workspace/checkpoint.pt")
    start = 0
    if args.resume and checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(saved["model"])
        start = saved["step"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.002, fused=True)
    phases = [(12000, 3e-3, 0.0, 0.0), (32000, 1e-3, 0.35, 0.0), (56000, 3e-4, 0.5, 0.05), (80000, 1e-4, 0.5, 0.15), (90000, 3e-5, 0.5, 0.25), (106000, 1e-5, 0.5, 0.60), (136000, 1e-5, 0.25, 0.10)]
    began = time.time()
    model.train()
    for step in range(start, args.steps):
        for end, lr, fraction, nines in phases:
            if step < end:
                break
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for(model, mixed_batch(args.batch, device, powers, fraction, nines))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        if (step + 1) % 1000 == 0:
            elapsed = time.time() - began
            print(f"step {step+1} loss {loss.item():.6f} lr {lr:g} elapsed {elapsed:.1f}s", flush=True)
        if (step + 1) % 10000 == 0 or step + 1 == args.steps:
            e1, n1 = autoregressive_errors(model, 2, device, powers, "uniform")
            e2, n2 = autoregressive_errors(model, 2, device, powers, "structured")
            print(f"validation step {step+1}: uniform {e1}/{n1}, structured {e2}/{n2}", flush=True)
            torch.save({"model": model.state_dict(), "step": step + 1}, checkpoint)
            export_submission(model, "/workspace/submission.py")
    export_submission(model, "/workspace/submission.py")


if __name__ == "__main__":
    main()
