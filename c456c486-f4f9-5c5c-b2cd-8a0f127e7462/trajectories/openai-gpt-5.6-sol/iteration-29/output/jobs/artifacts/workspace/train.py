import argparse
import math
import random
import time
from pathlib import Path

import torch
from torch import nn

from submission import AdditionTransformer

MAX = 100_000_000_000_000
PLACES = torch.tensor([10 ** i for i in range(14)], dtype=torch.long)


def digits(x):
    return (x[:, None] // PLACES.to(x.device)[None, :]) % 10


def values(d):
    return (d * PLACES.to(d.device)[None, :]).sum(1)


def make_batch(batch, device, structured=0.45):
    n_struct = int(batch * structured)
    n_uniform = batch - n_struct
    a = torch.randint(0, MAX, (n_uniform,), device=device)
    b = torch.randint(0, MAX, (n_uniform,), device=device)
    if n_struct:
        ad = torch.randint(0, 10, (n_struct, 14), device=device)
        bd = torch.randint(0, 10, (n_struct, 14), device=device)
        kind = torch.arange(n_struct, device=device) % 5

        # Exact carry chains with randomized start and length.
        ids = (kind == 0).nonzero().flatten()
        if ids.numel():
            starts = torch.randint(0, 14, (ids.numel(),), device=device)
            lengths = torch.randint(1, 15, (ids.numel(),), device=device)
            ends = torch.minimum(starts + lengths, torch.full_like(starts, 14))
            for col in range(14):
                active = (starts <= col) & (col < ends)
                begin = starts == col
                rows = ids[active]
                if rows.numel():
                    is_begin = begin[active]
                    av = torch.randint(0, 10, (rows.numel(),), device=device)
                    av = torch.where(is_begin & (av == 0), torch.ones_like(av), av)
                    ad[rows, col] = av
                    target = torch.where(is_begin, torch.full_like(av, 10), torch.full_like(av, 9))
                    bd[rows, col] = target - av
                stop = (ends == col) & (ends < 14)
                rows = ids[stop]
                if rows.numel():
                    av = torch.randint(0, 9, (rows.numel(),), device=device)
                    ad[rows, col] = av
                    bd[rows, col] = torch.randint(0, 9, (rows.numel(),), device=device) % (9 - av)

        # Matched long non-carry contrasts (column sums 8 or 9, no incoming carry).
        ids = (kind == 1).nonzero().flatten()
        if ids.numel():
            starts = torch.randint(0, 14, (ids.numel(),), device=device)
            lengths = torch.randint(1, 15, (ids.numel(),), device=device)
            ends = torch.minimum(starts + lengths, torch.full_like(starts, 14))
            totals = torch.randint(8, 10, (ids.numel(),), device=device)
            for col in range(14):
                active = (starts <= col) & (col < ends)
                rows = ids[active]
                if rows.numel():
                    target = totals[active]
                    av = (torch.rand(rows.numel(), device=device) * (target + 1)).long()
                    ad[rows, col] = av
                    bd[rows, col] = target - av

        # Sparse increments and exact 5+5 boundaries at random columns.
        ids = (kind == 2).nonzero().flatten()
        if ids.numel():
            ad[ids] = 0
            bd[ids] = 0
            cols = torch.randint(0, 14, (ids.numel(),), device=device)
            half = torch.arange(ids.numel(), device=device) % 2 == 0
            ad[ids, cols] = torch.where(half, torch.full_like(cols, 5), torch.randint(1, 10, cols.shape, device=device))
            bd[ids, cols] = torch.where(half, torch.full_like(cols, 5), torch.randint(0, 10, cols.shape, device=device))
            extra = torch.randint(0, 14, (ids.numel(),), device=device)
            ad[ids, extra] = torch.where(half, torch.zeros_like(extra), torch.randint(0, 10, extra.shape, device=device))

        # Repeated/block patterns.
        ids = (kind == 3).nonzero().flatten()
        if ids.numel():
            ra = torch.randint(0, 10, (ids.numel(), 1), device=device)
            rb = torch.randint(0, 10, (ids.numel(), 1), device=device)
            ad[ids] = ra
            bd[ids] = rb
            split = torch.randint(1, 14, (ids.numel(),), device=device)
            alt_a = torch.randint(0, 10, (ids.numel(),), device=device)
            alt_b = torch.randint(0, 10, (ids.numel(),), device=device)
            for col in range(14):
                mask = col >= split
                ad[ids[mask], col] = alt_a[mask]
                bd[ids[mask], col] = alt_b[mask]

        # Complementary columns, including overflow and near-boundary contrasts.
        ids = (kind == 4).nonzero().flatten()
        if ids.numel():
            ad[ids] = torch.randint(0, 10, (ids.numel(), 14), device=device)
            totals = torch.randint(8, 11, (ids.numel(), 1), device=device)
            bd[ids] = (totals - ad[ids]).clamp(0, 9)

        a = torch.cat((a, values(ad)))
        b = torch.cat((b, values(bd)))

    y = a + b
    ad = digits(a)
    bd = digits(b)
    yd = torch.cat((digits(y), (y // MAX)[:, None]), dim=1)
    pad = torch.full((batch, 1), 10, dtype=torch.long, device=device)
    return torch.cat((ad, pad), 1), torch.cat((bd, pad), 1), yd, a, b


def loss_for(model, ad, bd, yd):
    logits = model(ad, bd, yd[:, :-1])[:, 14:]
    return nn.functional.cross_entropy(logits.reshape(-1, 10), yd.reshape(-1))


@torch.no_grad()
def autoregressive(model, ad, bd):
    out = torch.empty((ad.shape[0], 0), dtype=torch.long, device=ad.device)
    for _ in range(15):
        out = torch.cat((out, model(ad, bd, out)[:, -1].argmax(1, keepdim=True)), 1)
    return out


@torch.no_grad()
def evaluate(model, total, device, structured, chunk=16384):
    model.eval()
    errors = 0
    digit_errors = 0
    seen = 0
    while seen < total:
        n = min(chunk, total - seen)
        ad, bd, yd, _, _ = make_batch(n, device, structured)
        pred = autoregressive(model, ad, bd)
        bad = pred.ne(yd)
        errors += bad.any(1).sum().item()
        digit_errors += bad.sum().item()
        seen += n
    model.train()
    return errors, digit_errors


def systematic(model, device):
    cases = set()
    top = MAX - 1
    constants = [0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 99, 101, top]
    for x in constants:
        for y in constants:
            if x < MAX and y < MAX:
                cases.add((x, y))
    for k in range(14):
        p = 10 ** k
        for d in range(1, 10):
            vals = [p, d * p, max(0, p - 1), min(top, p + 5), min(top, 5 * p)]
            for x in vals:
                for y in [1, 5, p, 5 * p if 5 * p < MAX else top]:
                    if x < MAX and y < MAX:
                        cases.add((x, y)); cases.add((y, x))
        if k:
            run = p - 1
            for inc in [1, 2, 5, 9, 10, p, top - run]:
                if 0 <= inc < MAX:
                    cases.add((run, inc)); cases.add((inc, run))
    aa = torch.tensor([x for x, _ in cases], device=device)
    bb = torch.tensor([y for _, y in cases], device=device)
    yy = aa + bb
    pad = torch.full((len(cases), 1), 10, device=device, dtype=torch.long)
    ad = torch.cat((digits(aa), pad), 1)
    bd = torch.cat((digits(bb), pad), 1)
    yd = torch.cat((digits(yy), (yy // MAX)[:, None]), 1)
    pred = autoregressive(model, ad, bd)
    return pred.ne(yd).any(1).sum().item(), len(cases)


def export(model, path):
    template = Path('/workspace/submission.py').read_text()
    start = template.index('_INITIAL_STATE =')
    end = template.index('\n\n\ndef build_model', start)
    arrays = []
    for p in model.parameters():
        vals = ','.join(format(float(v), '.9g') for v in p.detach().cpu().flatten())
        arrays.append('[' + vals + ']')
    replacement = '_INITIAL_STATE = [\n' + ',\n'.join(arrays) + '\n]'
    Path(path).write_text(template[:start] + replacement + template[end:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=30000)
    parser.add_argument('--batch', type=int, default=4096)
    parser.add_argument('--resume')
    args = parser.parse_args()
    torch.manual_seed(2901); random.seed(2901)
    device = torch.device('cuda')
    model = AdditionTransformer().to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True))
    print('parameters', sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.002, fused=True)
    best = 10**9
    began = time.time()
    for step in range(1, args.steps + 1):
        if step <= 9000: lr = 3e-3
        elif step <= 18000: lr = 1e-3
        elif step <= 25000: lr = 3e-4
        else: lr = 1e-4
        for group in opt.param_groups: group['lr'] = lr
        structured = 0.35 if step <= 9000 else (0.45 if step <= 25000 else 0.55)
        ad, bd, yd, _, _ = make_batch(args.batch, device, structured)
        opt.zero_grad(set_to_none=True)
        loss = loss_for(model, ad, bd, yd)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0 or step == 1:
            with torch.no_grad():
                pred = model(ad, bd, yd[:, :-1])[:, 14:].argmax(-1)
                seqerr = pred.ne(yd).any(1).float().mean().item()
            print(f'{step} loss={loss.item():.6g} teacher_seqerr={seqerr:.6g} lr={lr:g} elapsed={time.time()-began:.1f}', flush=True)
        if step >= 18000 and step % 2000 == 0:
            er, de = evaluate(model, 65536, device, 0.0)
            es, ds = evaluate(model, 65536, device, 1.0)
            sy, sn = systematic(model, device)
            score = er + es + sy * 100
            print(f'VALID {step} random={er}/65536 structured={es}/65536 systematic={sy}/{sn}', flush=True)
            torch.save(model.state_dict(), f'/workspace/checkpoint_{step}.pt')
            if score <= best:
                best = score
                torch.save(model.state_dict(), '/workspace/best.pt')
                export(model, '/workspace/submission.py')
                print('exported best', score, flush=True)
    print('training complete', flush=True)


if __name__ == '__main__':
    main()
