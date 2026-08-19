import argparse
import copy
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from submission import AdditionTransformer

LIMIT = 100_000_000_000_000
POW10 = torch.tensor([10 ** i for i in range(15)], device="cuda", dtype=torch.long)


def digits(x, width=14):
    return (x[:, None] // POW10[None, :width]) % 10


def uniform(n):
    return torch.randint(0, LIMIT, (n,), device="cuda"), torch.randint(0, LIMIT, (n,), device="cuda")


def structured(n):
    # Start from uniform so every family retains realistic unrelated columns.
    a, b = uniform(n)
    kind = torch.randint(0, 8, (n,), device="cuda")
    place = torch.randint(0, 14, (n,), device="cuda")
    power = POW10[place]

    # Sparse isolated boundaries, especially the historically difficult 5+5/9+9.
    mask = kind == 0
    vals = torch.where(torch.rand(n, device="cuda") < .5, 5, 9)
    a[mask] = (vals * power)[mask]
    b[mask] = (vals * power)[mask]

    # Sparse increments into a random lower prefix.
    mask = kind == 1
    low = torch.remainder(torch.randint(0, LIMIT, (n,), device="cuda"), power)
    a[mask] = low[mask]
    b[mask] = power[mask]

    # Exact shifted runs of nines plus one, spanning random lengths.
    mask = kind == 2
    length = torch.minimum(torch.randint(1, 15, (n,), device="cuda"), 14 - place)
    highp = POW10[place + length]
    run = highp - power
    a[mask] = run[mask]
    b[mask] = power[mask]

    # Near-carry contrasts: same run but add one below the triggering threshold.
    mask = kind == 3
    length = torch.minimum(torch.randint(1, 15, (n,), device="cuda"), 14 - place)
    highp = POW10[place + length]
    run = highp - power
    a[mask] = run[mask]
    b[mask] = torch.clamp(power[mask] - 1, min=0)

    # Repeated operands.
    mask = kind == 4
    da = torch.randint(0, 10, (n,), device="cuda")
    db = torch.randint(0, 10, (n,), device="cuda")
    rep = (LIMIT - 1) // 9
    a[mask] = (da * rep)[mask]
    b[mask] = (db * rep)[mask]

    # Complementary columns with random length and shift.
    mask = kind == 5
    d = torch.randint(1, 10, (n,), device="cuda")
    a[mask] = (d * power)[mask]
    b[mask] = ((10 - d) * power)[mask]

    # Maximum overflow and close non-overflow.
    mask = kind == 6
    a[mask] = LIMIT - 1
    b[mask] = torch.where(torch.rand(n, device="cuda") < .5, 1, LIMIT - 1)[mask]

    # Operand-symmetric sparse random columns.
    mask = kind == 7
    d1 = torch.randint(0, 10, (n,), device="cuda")
    d2 = torch.randint(0, 10, (n,), device="cuda")
    a[mask] = (d1 * power)[mask]
    b[mask] = (d2 * power)[mask]
    swap = torch.rand(n, device="cuda") < .5
    aa = torch.where(swap, b, a)
    bb = torch.where(swap, a, b)
    return aa, bb


def batch(n, structured_fraction):
    a, b = uniform(n)
    count = int(n * structured_fraction)
    if count:
        sa, sb = structured(count)
        a[:count], b[:count] = sa, sb
    ad, bd = digits(a), digits(b)
    target = digits(a + b, 15)
    return ad, bd, target


def loss_for(model, ad, bd, target):
    logits = model(ad, bd, target[:, :-1])
    return F.cross_entropy(logits[:, 14:].reshape(-1, 10), target.reshape(-1))

@torch.inference_mode()
def exact(model, n, mode="uniform", chunk=8192):
    wrong = 0
    done = 0
    while done < n:
        size = min(chunk, n - done)
        a, b = uniform(size) if mode == "uniform" else structured(size)
        ad, bd = digits(a), digits(b)
        prior = torch.empty((size, 0), dtype=torch.long, device="cuda")
        for _ in range(15):
            pred = model(ad, bd, prior)[:, -1].argmax(-1, keepdim=True)
            prior = torch.cat((prior, pred), 1)
        truth = digits(a + b, 15)
        wrong += (prior != truth).any(1).sum().item()
        done += size
    return wrong


def export(model, path):
    values = []
    for p in model.parameters():
        values.append(p.detach().float().cpu().flatten().tolist())
    text = path.read_text()
    marker = "_WEIGHTS = None"
    literal = "_WEIGHTS = " + repr(values)
    path.write_text(text.replace(marker, literal))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120000)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--resume")
    parser.add_argument("--refine", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(43)
    random.seed(43)
    model = AdditionTransformer().cuda()
    if args.resume:
        model.load_state_dict(torch.load(args.resume, weights_only=True))
    print("parameters", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=.002)
    # Uniform-first, then never exceed half structured data.
    phases = ([(args.steps, 1e-5, .25)] if args.refine else
              [(12000, 3e-3, 0.0), (18000, 1e-3, .25), (24000, 3e-4, .5),
               (30000, 1e-4, .5), (36000, 3e-5, .5)])
    boundaries = []
    total = 0
    for length, lr, frac in phases:
        total += length
        boundaries.append((total, lr, frac))
    if args.steps > total:
        boundaries.append((args.steps, 1e-5, .5))
    compiled = model
    step = 0
    for end, lr, frac in boundaries:
        if step >= args.steps: break
        for group in opt.param_groups: group["lr"] = lr
        while step < min(end, args.steps):
            ad, bd, target = batch(args.batch, frac)
            opt.zero_grad(set_to_none=True)
            loss = loss_for(compiled, ad, bd, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 2000 == 0:
                print(step, float(loss), lr, frac, flush=True)
            if step % 12000 == 0:
                torch.save(model.state_dict(), f"/workspace/checkpoint_{step}.pt")
    model.eval()
    torch.save(model.state_dict(), "/workspace/final.pt")
    for mode in ("uniform", "structured"):
        print(mode, "errors", exact(model, 131072, mode), flush=True)
    export(model, Path("/workspace/submission.py"))

if __name__ == "__main__":
    main()
