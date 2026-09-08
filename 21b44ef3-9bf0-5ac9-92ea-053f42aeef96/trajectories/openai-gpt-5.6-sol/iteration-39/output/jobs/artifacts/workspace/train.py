import argparse
import math
import random
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from submission import AdditionTransformer

LO = 10_000_000
HI = 99_999_999
BASE = 100_000_000
DEVICE = 'cuda'


def digits(n):
    places = torch.tensor([10 ** i for i in range(9)], device=n.device)
    return (n[:, None] // places % 10).long()


def make_batch(batch, structured=0.18):
    a = torch.randint(LO, HI + 1, (batch,), device=DEVICE)
    b = torch.randint(LO, HI + 1, (batch,), device=DEVICE)
    n = int(batch * structured)
    if n:
        mode = torch.randint(0, 7, (n,), device=DEVICE)
        aa = torch.randint(LO, HI + 1, (n,), device=DEVICE)
        bb = torch.randint(LO, HI + 1, (n,), device=DEVICE)
        # Near-complements exercise carries that reach the ninth output digit.
        ix = mode == 0
        delta = torch.randint(-20, 21, (n,), device=DEVICE)
        bb[ix] = (BASE - aa[ix] + delta[ix]).clamp(LO, HI)
        # Arbitrary-length asymmetric suffixes of nines and zeros.
        ix = mode == 1
        run = torch.randint(1, 8, (n,), device=DEVICE)
        p10 = 10 ** run
        prefix = torch.randint(1_000_000, 10_000_000, (n,), device=DEVICE)
        candidate = (prefix // p10 * p10 + p10 - 1).clamp(LO, HI)
        aa[ix] = candidate[ix]
        bb[ix] = torch.randint(LO, HI + 1, (n,), device=DEVICE)[ix]
        ix = mode == 2
        run = torch.randint(1, 8, (n,), device=DEVICE)
        p10 = 10 ** run
        aa[ix] = (aa[ix] // p10[ix] * p10[ix]).clamp(LO, HI)
        bb[ix] = (p10[ix] - 1 + (bb[ix] // p10[ix]) * p10[ix]).clamp(LO, HI)
        # Decimal boundaries with perturbations on both sides.
        ix = mode == 3
        power = 10 ** torch.randint(1, 8, (n,), device=DEVICE)
        center = (torch.randint(1, 100_000_000, (n,), device=DEVICE) // power) * power
        jitter = torch.randint(-100, 101, (n,), device=DEVICE)
        aa[ix] = (center[ix] + jitter[ix]).clamp(LO, HI)
        # Repeated digits.
        ix = mode == 4
        repa = torch.randint(1, 10, (n,), device=DEVICE) * 11_111_111
        repb = torch.randint(1, 10, (n,), device=DEVICE) * 11_111_111
        aa[ix], bb[ix] = repa[ix], repb[ix]
        # Sparse lower digits and near extrema.
        ix = mode == 5
        lead = torch.randint(1, 10, (n,), device=DEVICE) * 10_000_000
        tail = torch.randint(0, 1000, (n,), device=DEVICE)
        aa[ix] = (lead[ix] + tail[ix]).clamp(LO, HI)
        bb[ix] = (HI - torch.randint(0, 10000, (n,), device=DEVICE))[ix]
        ix = mode == 6
        aa[ix] = (LO + torch.randint(0, 10000, (n,), device=DEVICE))[ix]
        bb[ix] = (HI - torch.randint(0, 10000, (n,), device=DEVICE))[ix]
        a[:n], b[:n] = aa, bb
    ad, bd, target = digits(a)[:, :8], digits(b)[:, :8], digits(a + b)
    inputs = torch.empty(batch, 25, dtype=torch.long, device=DEVICE)
    inputs[:, 0:16:2], inputs[:, 1:16:2] = ad, bd
    inputs[:, 16] = 10
    inputs[:, 17:] = target[:, :8]
    return inputs, target


@torch.no_grad()
def evaluate(model, batches=10, batch=10000, structured=0.0):
    model.eval()
    correct = total = 0
    min_margin = 100.0
    for _ in range(batches):
        x, y = make_batch(batch, structured)
        logits = model(x)[:, 16:25]
        pred = logits.argmax(-1)
        correct += (pred == y).all(1).sum().item()
        total += batch
        vals = logits.gather(2, y.unsqueeze(-1)).squeeze(-1)
        wrong = logits.scatter(2, y.unsqueeze(-1), -1e9).amax(-1)
        min_margin = min(min_margin, (vals - wrong).amin().item())
    model.train()
    return correct, total, min_margin


def train_phase(model, steps, lr, structured, name, batch=8192):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.01)
    scheduler = None
    if name == 'teacher':
        def scale(step):
            if step < 500: return (step + 1) / 500
            return max(0.08, 0.5 * (1 + math.cos(math.pi * (step - 500) / (steps - 500))))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
    started = time.time()
    model.train()
    for step in range(1, steps + 1):
        x, y = make_batch(batch, structured)
        logits = model(x)[:, 16:25]
        loss = F.cross_entropy(logits.reshape(-1, 10), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler: scheduler.step()
        if step == 1 or step % 1000 == 0:
            c, t, margin = evaluate(model, batches=2, batch=10000, structured=0.5)
            print(f'{name} {step}/{steps} loss={loss.item():.5f} val={c}/{t} margin={margin:.3f} sec={time.time()-started:.1f}', flush=True)
            torch.save(model.state_dict(), f'/workspace/{name}.pt')
    torch.save(model.state_dict(), f'/workspace/{name}.pt')


def prune_ff(model, width):
    old = model.ff_in.out_features
    assert width == old - 1
    importance = model.ff_in.weight.norm(dim=1) * model.ff_out.weight.norm(dim=0)
    keep = torch.argsort(importance, descending=True)[:width].sort().values
    result = AdditionTransformer(width).to(DEVICE)
    state = model.state_dict()
    new = result.state_dict()
    for key in new:
        if key == 'ff_in.weight': new[key].copy_(state[key][keep])
        elif key == 'ff_out.weight': new[key].copy_(state[key][:, keep])
        else: new[key].copy_(state[key])
    return result


def export_standard(model, output='/workspace/submission.py'):
    template = Path('/workspace/submission.py').read_text()
    marker = '_TRAINED_STATE = None'
    state = {k: v.detach().cpu().float().tolist() for k, v in model.state_dict().items()}
    literal = '_TRAINED_STATE = {\n' + ''.join(f'    {k!r}: torch.tensor({v!r}),\n' for k, v in state.items()) + '}'
    if marker not in template:
        start = template.index('_TRAINED_STATE = {')
        end = template.index('\n\n\ndef build_model', start)
        template = template[:start] + marker + template[end:]
    Path(output).write_text(template.replace(marker, literal))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['teacher', 'width3', 'width2', 'all'], default='all')
    args = parser.parse_args()
    torch.manual_seed(2025); random.seed(2025)
    torch.backends.cuda.matmul.allow_tf32 = True
    if args.stage in ('teacher', 'all'):
        model = AdditionTransformer(4).to(DEVICE)
        train_phase(model, 36000, 2e-3, 0.18, 'teacher')
        train_phase(model, 6000, 5e-5, 0.30, 'teacher_stable')
    else:
        model = AdditionTransformer(4).to(DEVICE); model.load_state_dict(torch.load('/workspace/teacher_stable.pt'))
    if args.stage == 'teacher': export_standard(model); return
    if args.stage in ('width3', 'all'):
        model = prune_ff(model, 3)
        train_phase(model, 18000, 2e-5, 0.30, 'width3')
    else:
        model = AdditionTransformer(3).to(DEVICE); model.load_state_dict(torch.load('/workspace/width3.pt'))
    if args.stage == 'width3': export_standard(model); return
    model = prune_ff(model, 2)
    train_phase(model, 30000, 1.2e-5, 0.32, 'width2')
    train_phase(model, 12000, 4e-6, 0.35, 'width2_polish')
    train_phase(model, 6000, 2e-6, 0.55, 'width2_edge')
    c1, t1, m1 = evaluate(model, 50, 10000, 0.0)
    c2, t2, m2 = evaluate(model, 50, 10000, 0.75)
    print('FINAL', c1, t1, m1, c2, t2, m2, flush=True)
    torch.save(model.state_dict(), '/workspace/final.pt')
    export_standard(model)


if __name__ == '__main__':
    main()
