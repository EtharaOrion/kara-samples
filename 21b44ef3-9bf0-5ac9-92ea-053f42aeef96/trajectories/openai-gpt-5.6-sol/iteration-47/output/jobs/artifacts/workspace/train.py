import argparse
import copy
import importlib.util
import math
import os
import random
import time
import torch
import torch.nn.functional as F

os.chdir('/workspace')
torch.set_float32_matmul_precision('high')
from submission import AdditionTransformer

DEVICE = 'cuda'
BATCH = 8192
LOW = 10_000_000
HIGH = 99_999_999
POW10 = torch.tensor([10, 100, 1000, 10000, 100000, 1000000, 10000000], device=DEVICE)


def make_pairs(n, structured=0.18):
    a = torch.randint(LOW, HIGH + 1, (n,), device=DEVICE)
    b = torch.randint(LOW, HIGH + 1, (n,), device=DEVICE)
    m = int(n * structured)
    if not m:
        return a, b
    kind = torch.randint(0, 7, (m,), device=DEVICE)
    aa = torch.randint(LOW, HIGH + 1, (m,), device=DEVICE)
    bb = torch.randint(LOW, HIGH + 1, (m,), device=DEVICE)

    # Exact complements exercise uninterrupted carries through every output column.
    ix = kind == 0
    if ix.any():
        aa[ix] = torch.randint(LOW, 90_000_001, (int(ix.sum()),), device=DEVICE)
        bb[ix] = 100_000_000 - aa[ix]

    # Long asymmetric suffixes of nines or zeros.
    for typ, nines_a in ((1, True), (2, False)):
        ix = kind == typ
        c = int(ix.sum())
        if c:
            p = POW10[torch.randint(0, len(POW10), (c,), device=DEVICE)]
            base = torch.randint(LOW, HIGH + 1, (c,), device=DEVICE)
            patterned = torch.div(base, p, rounding_mode='floor') * p
            if nines_a:
                patterned += p - 1
            patterned.clamp_(LOW, HIGH)
            aa[ix] = patterned
            bb[ix] = torch.randint(LOW, HIGH + 1, (c,), device=DEVICE)

    # Decimal boundaries with perturbations on either side.
    ix = kind == 3
    c = int(ix.sum())
    if c:
        p = POW10[torch.randint(0, len(POW10), (c,), device=DEVICE)]
        base = torch.randint(1, 100_000_000, (c,), device=DEVICE)
        aa[ix] = (torch.div(base, p, rounding_mode='floor') * p + torch.randint(-2, 3, (c,), device=DEVICE)).clamp(LOW, HIGH)
        bb[ix] = torch.randint(LOW, HIGH + 1, (c,), device=DEVICE)

    # Repeated digits.
    ix = kind == 4
    c = int(ix.sum())
    if c:
        rep = torch.tensor([11_111_111,22_222_222,33_333_333,44_444_444,55_555_555,66_666_666,77_777_777,88_888_888,99_999_999], device=DEVICE)
        aa[ix] = rep[torch.randint(0, 9, (c,), device=DEVICE)]

    # Sparse zero/nine digits while retaining a nonzero leading digit.
    ix = kind == 5
    c = int(ix.sum())
    if c:
        lead = torch.randint(1, 10, (c, 1), device=DEVICE)
        tail = torch.randint(0, 2, (c, 7), device=DEVICE) * 9
        digs = torch.cat((lead, tail), 1)
        place = torch.tensor([10_000_000,1_000_000,100_000,10_000,1000,100,10,1], device=DEVICE)
        aa[ix] = (digs * place).sum(1)

    # Near extremes.
    ix = kind == 6
    c = int(ix.sum())
    if c:
        aa[ix] = torch.where(torch.rand(c, device=DEVICE) < .5,
                             LOW + torch.randint(0, 10000, (c,), device=DEVICE),
                             HIGH - torch.randint(0, 10000, (c,), device=DEVICE))
        bb[ix] = torch.where(torch.rand(c, device=DEVICE) < .5,
                             LOW + torch.randint(0, 10000, (c,), device=DEVICE),
                             HIGH - torch.randint(0, 10000, (c,), device=DEVICE))
    a[:m], b[:m] = aa, bb
    return a, b


def digits_lsf(x, count):
    cols = []
    for _ in range(count):
        cols.append(x.remainder(10))
        x = torch.div(x, 10, rounding_mode='floor')
    return torch.stack(cols, 1)


def batch(n=BATCH, structured=.18):
    a, b = make_pairs(n, structured)
    da, db = digits_lsf(a, 8), digits_lsf(b, 8)
    operands = torch.stack((da, db), 2).reshape(n, 16)
    target = digits_lsf(a + b, 9)
    inp = torch.cat((operands, torch.full((n, 1), 10, device=DEVICE), target[:, :-1]), 1)
    return inp, target, a, b


@torch.no_grad()
def evaluate(model, n=100000, structured=.0, chunk=20000):
    model.eval()
    good = total = 0
    min_margin = 1e9
    while total < n:
        q = min(chunk, n-total)
        inp, target, _, _ = batch(q, structured)
        prefix = inp[:, :17]
        preds = []
        for _ in range(9):
            logits = model(prefix)[:, -1]
            vals, inds = logits.topk(2, 1)
            preds.append(inds[:, 0])
            min_margin = min(min_margin, float((vals[:, 0]-vals[:, 1]).min()))
            prefix = torch.cat((prefix, inds[:, :1]), 1)
        pred = torch.stack(preds, 1)
        good += int((pred == target).all(1).sum())
        total += q
    model.train()
    return good, total, min_margin


def prune(model, new_ff):
    old = model.ff1.out_features
    assert new_ff == old - 1
    importance = model.ff1.weight.detach().norm(dim=1) * model.ff2.weight.detach().norm(dim=0)
    keep = [i for i in range(old) if i != int(importance.argmin())]
    result = AdditionTransformer(new_ff).to(DEVICE)
    state = model.state_dict()
    own = result.state_dict()
    for name in own:
        if name == 'ff1.weight': own[name].copy_(state[name][keep])
        elif name == 'ff2.weight': own[name].copy_(state[name][:, keep])
        else: own[name].copy_(state[name])
    return result


def run_phase(model, steps, lr, structured, label, eval_every=2000):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(.9, .98), weight_decay=.01)
    start = time.time()
    for step in range(1, steps+1):
        inp, target, _, _ = batch(structured=structured)
        logits = model(inp)[:, 16:25]
        loss = F.cross_entropy(logits.reshape(-1, 10), target.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % eval_every == 0 or step == steps:
            g, t, margin = evaluate(model, 10000, 0, 10000)
            gs, ts, _ = evaluate(model, 10000, .75, 10000)
            print(f'{label} {step}/{steps} loss={loss.item():.5f} random={g}/{t} struct={gs}/{ts} margin={margin:.3f} elapsed={time.time()-start:.1f}', flush=True)
            torch.save({'model': model.state_dict(), 'ff': model.ff1.out_features, 'phase': label, 'step': step}, f'/workspace/{label}.pt')
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    torch.manual_seed(2025); random.seed(2025)
    model = AdditionTransformer(4).to(DEVICE)
    # Keep PyTorch's default initialization, especially LayerNorm weights.
    teacher_steps = 12000 if args.quick else 36000
    model = run_phase(model, teacher_steps, 2e-3, .18, 'teacher4')
    model = run_phase(model, 6000, 5e-5, .30, 'stable4')
    g, t, _ = evaluate(model, 100000, 0)
    gs, ts, _ = evaluate(model, 100000, .75)
    print('teacher-final', g, t, gs, ts, flush=True)
    if g < 99990 or gs < 99990:
        print('Teacher insufficient; refusing to prune', flush=True)
        return
    model = prune(model, 3)
    model = run_phase(model, 18000, 2e-5, .30, 'recover3')
    model = prune(model, 2)
    model = run_phase(model, 30000, 1.2e-5, .35, 'recover2')
    model = run_phase(model, 12000, 4e-6, .40, 'polish2')
    torch.save({'model': model.state_dict(), 'ff': 2}, '/workspace/final.pt')
    for s in (0., .75):
        print('FINAL', s, evaluate(model, 500000, s), flush=True)


if __name__ == '__main__':
    main()
