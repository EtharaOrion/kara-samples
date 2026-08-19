import argparse
import importlib.util
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path('/workspace')
BASE = 100_000_000_000_000
POWERS = torch.tensor([10 ** i for i in range(15)], device='cuda', dtype=torch.long)


def load_submission():
    spec = importlib.util.spec_from_file_location('submission_train', ROOT / 'submission.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digits(values):
    return (values[:, None] // POWERS[None, :]) % 10


def random_pairs(n):
    return torch.randint(BASE, (n, 2), device='cuda', dtype=torch.long)


def digits_to_values(d):
    return (d * POWERS).sum(1)


def carry_pairs(n):
    # Explicit isolated carry chains with randomized start and length.
    da = torch.randint(10, (n, 15), device='cuda')
    db = torch.randint(10, (n, 15), device='cuda')
    da[:, 14] = 0
    db[:, 14] = 0
    starts = torch.randint(14, (n,), device='cuda')
    lengths = torch.randint(1, 15, (n,), device='cuda')
    lengths = torch.minimum(lengths, 14 - starts)
    col = torch.arange(14, device='cuda')[None, :]
    before = col < starts[:, None]
    chain = (col >= starts[:, None]) & (col < (starts + lengths)[:, None])
    first = col == starts[:, None]
    end = col == (starts + lengths)[:, None]
    # Prevent an accidental incoming carry before the designed chain.
    target = torch.randint(9, (n, 14), device='cuda')
    aa = torch.minimum(da[:, :14], target)
    bb = target - aa
    da[:, :14] = torch.where(before, aa, da[:, :14])
    db[:, :14] = torch.where(before, bb, db[:, :14])
    comp_a = torch.randint(10, (n, 14), device='cuda')
    comp_b = 9 - comp_a
    da[:, :14] = torch.where(chain, comp_a, da[:, :14])
    db[:, :14] = torch.where(chain, comp_b, db[:, :14])
    start_a = torch.randint(1, 10, (n, 14), device='cuda')
    start_b = 10 - start_a
    da[:, :14] = torch.where(first, start_a, da[:, :14])
    db[:, :14] = torch.where(first, start_b, db[:, :14])
    end_target = torch.randint(9, (n, 14), device='cuda')
    end_a = torch.minimum(torch.randint(10, (n, 14), device='cuda'), end_target)
    end_b = end_target - end_a
    da[:, :14] = torch.where(end, end_a, da[:, :14])
    db[:, :14] = torch.where(end, end_b, db[:, :14])
    return torch.stack((digits_to_values(da), digits_to_values(db)), 1)


def pattern_pairs(n):
    da = torch.randint(10, (n, 15), device='cuda')
    db = torch.randint(10, (n, 15), device='cuda')
    da[:, 14] = db[:, 14] = 0
    kind = torch.randint(5, (n,), device='cuda')
    cols = torch.arange(14, device='cuda')[None, :]

    # Repeated digits in either or both operands.
    rep = kind == 0
    ra = torch.randint(10, (n, 1), device='cuda').expand(-1, 14)
    rb = torch.randint(10, (n, 1), device='cuda').expand(-1, 14)
    da[:, :14] = torch.where(rep[:, None], ra, da[:, :14])
    db[:, :14] = torch.where(rep[:, None], rb, db[:, :14])

    # Sparse operands and increments at arbitrary columns.
    sparse = kind == 1
    keep_a = torch.rand(n, 14, device='cuda') < 0.18
    keep_b = torch.rand(n, 14, device='cuda') < 0.18
    da[:, :14] = torch.where(sparse[:, None] & ~keep_a, 0, da[:, :14])
    db[:, :14] = torch.where(sparse[:, None] & ~keep_b, 0, db[:, :14])

    # Complementary columns, both carry and no-carry boundaries.
    comp = kind == 2
    target = torch.randint(8, 11, (n, 14), device='cuda')
    ca = torch.randint(10, (n, 14), device='cuda')
    ca = torch.minimum(ca, target)
    cb = target - ca
    valid = cb < 10
    da[:, :14] = torch.where(comp[:, None] & valid, ca, da[:, :14])
    db[:, :14] = torch.where(comp[:, None] & valid, cb, db[:, :14])

    # Long 9-runs without an initiating carry (anti-hallucination).
    boundary = kind == 3
    start = torch.randint(14, (n,), device='cuda')
    length = torch.randint(1, 15, (n,), device='cuda')
    run = (cols >= start[:, None]) & (cols < (start + length)[:, None])
    side = torch.rand(n, 1, device='cuda') < .5
    da[:, :14] = torch.where(boundary[:, None] & run & side, 9, da[:, :14])
    db[:, :14] = torch.where(boundary[:, None] & run & side, 0, db[:, :14])
    db[:, :14] = torch.where(boundary[:, None] & run & ~side, 9, db[:, :14])
    da[:, :14] = torch.where(boundary[:, None] & run & ~side, 0, da[:, :14])

    # Blockwise constant digits.
    block = kind == 4
    split = torch.randint(1, 14, (n, 1), device='cuda')
    a1, a2 = torch.randint(10, (n, 1), device='cuda'), torch.randint(10, (n, 1), device='cuda')
    b1, b2 = torch.randint(10, (n, 1), device='cuda'), torch.randint(10, (n, 1), device='cuda')
    ba = torch.where(cols < split, a1, a2)
    bb = torch.where(cols < split, b1, b2)
    da[:, :14] = torch.where(block[:, None], ba, da[:, :14])
    db[:, :14] = torch.where(block[:, None], bb, db[:, :14])
    return torch.stack((digits_to_values(da), digits_to_values(db)), 1)


def batch_pairs(n, phase):
    if phase == 0:
        nr, nc = n * 3 // 4, n // 8
    elif phase == 1:
        nr, nc = n // 2, n // 4
    else:
        nr, nc = n * 5 // 8, n // 4
    chunks = [random_pairs(nr), carry_pairs(nc), pattern_pairs(n - nr - nc)]
    pairs = torch.cat(chunks)
    return pairs[torch.randperm(n, device='cuda')]


def make_io(pairs):
    a, b = pairs[:, 0], pairs[:, 1]
    x = torch.stack((digits(a), digits(b)), 1)
    y = digits(a + b)
    return x, y

@torch.no_grad()
def evaluate(model, n, generator=random_pairs, batch=65536):
    model.eval()
    errors = 0
    digit_errors = 0
    for _ in range(math.ceil(n / batch)):
        m = min(batch, n)
        p = generator(m)
        x, y = make_io(p)
        pred = model(x).argmax(-1)
        bad = pred.ne(y)
        errors += bad.any(1).sum().item()
        digit_errors += bad.sum().item()
        n -= m
    model.train()
    return errors, digit_errors


def export(model, hidden, path=ROOT / 'submission.py'):
    values = torch.nn.utils.parameters_to_vector(model.parameters()).detach().float().cpu().tolist()
    text = path.read_text()
    start = text.index('_WEIGHTS = ')
    end = text.index('\n\n\ndef build_model', start)
    literal = '_WEIGHTS = ' + repr(values)
    path.write_text(text[:start] + literal + text[end:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden', type=int, default=7)
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--seed', type=int, default=2207)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    sub = load_submission()
    model = sub.AdditionTransformer(hidden=args.hidden).cuda().train()
    print('parameters', sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.003)
    best = None
    for step in range(1, args.steps + 1):
        frac = step / args.steps
        phase = 0 if frac < .55 else (1 if frac < .82 else 2)
        if frac < .45: lr = 3e-3
        elif frac < .68: lr = 1e-3
        elif frac < .86: lr = 3e-4
        else: lr = 1e-4
        for g in opt.param_groups: g['lr'] = lr
        x, y = make_io(batch_pairs(args.batch, phase))
        loss = F.cross_entropy(model(x).reshape(-1, 10), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0 or step == args.steps:
            er, de = evaluate(model, 131072)
            ec, dc = evaluate(model, 65536, carry_pairs)
            ep, dp = evaluate(model, 65536, pattern_pairs)
            score = er * 4 + ec + ep
            print(step, f'loss={loss.item():.6g}', f'random={er}/{131072}', f'carry={ec}/{65536}', f'pattern={ep}/{65536}', 'lr', lr, flush=True)
            torch.save({'model': model.state_dict(), 'step': step, 'score': score}, ROOT / f'checkpoint_h{args.hidden}.pt')
            if best is None or score <= best:
                best = score
                export(model, args.hidden)
                torch.save(model.state_dict(), ROOT / f'best_h{args.hidden}.pt')
    model.load_state_dict(torch.load(ROOT / f'best_h{args.hidden}.pt', weights_only=True))
    export(model, args.hidden)

if __name__ == '__main__':
    main()
