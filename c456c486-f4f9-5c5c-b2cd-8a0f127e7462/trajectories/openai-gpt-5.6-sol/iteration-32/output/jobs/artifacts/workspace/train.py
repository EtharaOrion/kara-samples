import argparse
import importlib.util
import math
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
LIMIT = 100_000_000_000_000
POWERS = torch.tensor([10 ** i for i in range(15)], dtype=torch.long)


def load_submission():
    spec = importlib.util.spec_from_file_location("submission", ROOT / "submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digits(values, width=15):
    powers = POWERS[:width].to(values.device)
    return (values[:, None] // powers[None, :]) % 10


def values_from_digits(ds):
    return (ds * POWERS[:14].to(ds.device)).sum(1)


def carry_examples(n, device):
    da = torch.randint(0, 10, (n, 14), device=device)
    db = torch.randint(0, 10, (n, 14), device=device)
    start = torch.randint(0, 14, (n,), device=device)
    length = torch.randint(1, 15, (n,), device=device)
    end = torch.minimum(start + length, torch.full_like(start, 14))
    cols = torch.arange(14, device=device)[None, :]
    trigger = cols == start[:, None]
    run = (cols > start[:, None]) & (cols < end[:, None])
    stop = (cols == end[:, None]) & (end[:, None] < 14)
    x = torch.randint(1, 10, (n, 14), device=device)
    da = torch.where(trigger, x, da)
    db = torch.where(trigger, 10 - x, db)
    x = torch.randint(0, 10, (n, 14), device=device)
    da = torch.where(run, x, da)
    db = torch.where(run, 9 - x, db)
    x = torch.randint(0, 9, (n, 14), device=device)
    da = torch.where(stop, x, da)
    db = torch.where(stop, torch.randint(0, 9, (n, 14), device=device) % (9 - x), db)
    return values_from_digits(da), values_from_digits(db)


def noncarry_examples(n, device):
    da = torch.randint(0, 10, (n, 14), device=device)
    db = torch.randint(0, 10, (n, 14), device=device)
    start = torch.randint(0, 14, (n,), device=device)
    length = torch.randint(1, 15, (n,), device=device)
    end = torch.minimum(start + length, torch.full_like(start, 14))
    cols = torch.arange(14, device=device)[None, :]
    run = (cols >= start[:, None]) & (cols < end[:, None])
    x = torch.randint(0, 10, (n, 14), device=device)
    da = torch.where(run, x, da)
    db = torch.where(run, 9 - x, db)
    return values_from_digits(da), values_from_digits(db)


def boundary_examples(n, device):
    a = torch.zeros(n, dtype=torch.long, device=device)
    b = torch.zeros_like(a)
    pos = torch.randint(0, 14, (n,), device=device)
    place = torch.tensor(10, device=device, dtype=torch.long).pow(pos)
    kind = torch.randint(0, 4, (n,), device=device)
    x = torch.randint(0, 10, (n,), device=device)
    y = torch.randint(0, 10, (n,), device=device)
    x = torch.where(kind == 0, torch.full_like(x, 5), x)
    y = torch.where(kind == 0, torch.full_like(y, 5), y)
    x = torch.where(kind == 1, torch.full_like(x, 9), x)
    y = torch.where(kind == 1, torch.full_like(y, 9), y)
    # Add occasional lower random columns while keeping the selected column isolated often.
    low_mask = torch.randint(0, 2, (n,), device=device).bool() & (pos > 0)
    low_limit = torch.where(pos > 0, place, torch.ones_like(place))
    low_a = torch.floor(torch.rand(n, device=device) * low_limit).long()
    low_b = torch.floor(torch.rand(n, device=device) * low_limit).long()
    a = x * place + torch.where(low_mask, low_a, 0)
    b = y * place + torch.where(low_mask, low_b, 0)
    return a, b


def patterned_examples(n, device):
    kind = torch.randint(0, 4, (n,), device=device)
    digit_a = torch.randint(0, 10, (n,), device=device)
    digit_b = torch.randint(0, 10, (n,), device=device)
    rep = torch.full((n,), 11_111_111_111_111, dtype=torch.long, device=device)
    a = digit_a * rep
    b = digit_b * rep
    random_a = torch.randint(0, LIMIT, (n,), device=device)
    # Complement columns and sparse increments.
    comp_b = (torch.full_like(random_a, LIMIT - 1) - random_a)
    pos = torch.randint(0, 14, (n,), device=device)
    sparse = torch.tensor(10, device=device, dtype=torch.long).pow(pos)
    a = torch.where(kind == 1, random_a, a)
    b = torch.where(kind == 1, comp_b, b)
    a = torch.where(kind == 2, random_a, a)
    b = torch.where(kind == 2, sparse, b)
    a = torch.where(kind == 3, torch.full_like(a, LIMIT - 1), a)
    b = torch.where(kind == 3, sparse, b)
    b = torch.minimum(b, torch.full_like(b, LIMIT - 1))
    return a, b


def batch_pairs(batch, device, phase="mixed"):
    if phase == "random":
        return (torch.randint(0, LIMIT, (batch,), device=device),
                torch.randint(0, LIMIT, (batch,), device=device))
    q = batch // 4
    a0 = torch.randint(0, LIMIT, (q,), device=device)
    b0 = torch.randint(0, LIMIT, (q,), device=device)
    if phase == "boundary":
        a1, b1 = boundary_examples(q, device)
        a2, b2 = boundary_examples(q, device)
        a3, b3 = carry_examples(batch - 3 * q, device)
    else:
        a1, b1 = carry_examples(q, device)
        a2, b2 = noncarry_examples(q, device)
        if phase == "broad":
            a3, b3 = patterned_examples(batch - 3 * q, device)
        else:
            a3, b3 = boundary_examples(batch - 3 * q, device)
    return torch.cat((a0, a1, a2, a3)), torch.cat((b0, b1, b2, b3))


def model_inputs(a, b):
    ad = digits(a, 14)
    bd = digits(b, 14)
    sentinel = torch.full((a.shape[0], 1), 10, dtype=torch.long, device=a.device)
    ad = torch.cat((ad, sentinel), 1)
    bd = torch.cat((bd, sentinel), 1)
    target = digits(a + b, 15)
    return ad, bd, target


def train_loss(model, a, b):
    ad, bd, target = model_inputs(a, b)
    logits = model(ad, bd, target[:, :14])[:, 14:29]
    return F.cross_entropy(logits.reshape(-1, 10), target.reshape(-1))


@torch.no_grad()
def predict(model, a, b):
    ad, bd, target = model_inputs(a, b)
    previous = target[:, :0]
    output = []
    for _ in range(15):
        logits = model(ad, bd, previous)
        token = logits[:, -1].argmax(1)
        output.append(token)
        previous = torch.stack(output, 1)
    return previous


@torch.no_grad()
def evaluate(model, count, kind, device, chunk=16384):
    model.eval()
    errors = 0
    remaining = count
    while remaining:
        n = min(chunk, remaining)
        a, b = batch_pairs(n, device, kind)
        target = model_inputs(a, b)[2]
        errors += (predict(model, a, b) != target).any(1).sum().item()
        remaining -= n
    model.train()
    return errors


def systematic_pairs(device):
    pairs = {(0, 0), (LIMIT - 1, 0), (LIMIT - 1, 1), (LIMIT - 1, LIMIT - 1)}
    for p in range(14):
        place = 10 ** p
        for x in range(10):
            for y in range(10):
                pairs.add((x * place, y * place))
        for length in range(1, 15 - p):
            run = (10 ** length - 1) * place
            pairs.add((run, place))
            pairs.add((run - place if run >= place else 0, place))
            pairs.add((5 * place, 5 * place))
            pairs.add((9 * place, 9 * place))
    vals = list(pairs)
    return (torch.tensor([x[0] for x in vals], device=device),
            torch.tensor([x[1] for x in vals], device=device))


@torch.no_grad()
def systematic_errors(model, device):
    a, b = systematic_pairs(device)
    return (predict(model, a, b) != model_inputs(a, b)[2]).any(1).sum().item(), len(a)


def export(model, submission_path):
    values = torch.cat([p.detach().float().cpu().reshape(-1) for p in model.parameters()]).tolist()
    text = submission_path.read_text()
    start = text.index("_WEIGHTS = ")
    end = text.index("\n\n\ndef _load_weights", start)
    literal = "_WEIGHTS = [\n"
    for i in range(0, len(values), 12):
        literal += "    " + ", ".join(format(v, ".9g") for v in values[i:i + 12]) + ",\n"
    literal += "]"
    submission_path.write_text(text[:start] + literal + text[end:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=38000)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(918273)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    sub = load_submission()
    model = sub.AdditionTransformer().to(device)
    checkpoint = ROOT / "best.pt"
    if args.resume and checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.003, fused=True)
    best_score = math.inf
    schedule = ((12000, 3e-3, "mixed"), (22000, 1e-3, "broad"),
                (32000, 3e-4, "mixed"), (38000, 8e-5, "boundary"))
    for step in range(1, args.steps + 1):
        for boundary, lr, phase in schedule:
            if step <= boundary:
                break
        for group in optimizer.param_groups:
            group["lr"] = lr
        a, b = batch_pairs(args.batch, device, phase)
        loss = train_loss(model, a, b)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 1000 == 0 or step == args.steps:
            random_errors = evaluate(model, 65536, "random", device)
            structured_errors = evaluate(model, 65536, "mixed", device)
            edge_errors, edge_count = systematic_errors(model, device)
            score = random_errors * 4 + structured_errors * 2 + edge_errors * 100
            print(f"step={step} loss={loss.item():.6f} random={random_errors}/65536 "
                  f"mixed={structured_errors}/65536 edge={edge_errors}/{edge_count} lr={lr:g}", flush=True)
            if score <= best_score or step > 32000:
                best_score = min(best_score, score)
                torch.save(model.state_dict(), checkpoint)
                export(model, ROOT / "submission.py")
    print("saved", checkpoint, "parameters", sum(p.numel() for p in model.parameters()))


if __name__ == "__main__":
    main()
