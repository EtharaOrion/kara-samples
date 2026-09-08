import copy
import math
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, "/workspace")
from submission import AdditionTransformer

DEVICE = "cuda"
BATCH = 8192
LO = 10_000_000
HI = 100_000_000
POW10 = torch.tensor([10**i for i in range(9)], device=DEVICE, dtype=torch.long)


def digits(x, n=8):
    return (x[:, None] // POW10[None, :n]) % 10


def make_batch(batch=BATCH, structured=0.18):
    a = torch.randint(LO, HI, (batch,), device=DEVICE)
    b = torch.randint(LO, HI, (batch,), device=DEVICE)
    n = int(batch * structured)
    if n:
        kind = torch.randint(0, 8, (n,), device=DEVICE)
        aa = torch.randint(LO, HI, (n,), device=DEVICE)
        bb = torch.randint(LO, HI, (n,), device=DEVICE)
        # Exact and near complements around 10^8.
        m = kind == 0
        if m.any():
            delta = torch.randint(-20, 21, (int(m.sum()),), device=DEVICE)
            bb[m] = (HI - aa[m] + delta).clamp(LO, HI - 1)
        # Long unequal suffixes of 9 and 0.
        m = kind == 1
        if m.any():
            c = int(m.sum()); run = torch.randint(1, 8, (c,), device=DEVICE)
            p = POW10[run]
            prefix = torch.randint(1, 100_000_000, (c,), device=DEVICE)
            aa[m] = ((prefix // p) * p + p - 1).clamp(LO, HI - 1)
            bb[m] = torch.randint(LO, HI, (c,), device=DEVICE)
        # Decimal boundaries with nearby offsets.
        m = kind == 2
        if m.any():
            c = int(m.sum()); run = torch.randint(1, 8, (c,), device=DEVICE); p = POW10[run]
            base = torch.randint(LO, HI, (c,), device=DEVICE)
            aa[m] = ((base // p) * p + torch.randint(-9, 10, (c,), device=DEVICE)).clamp(LO, HI - 1)
        # Repeated decimal digits.
        m = kind == 3
        if m.any():
            c = int(m.sum()); d = torch.randint(0, 10, (c,), device=DEVICE)
            aa[m] = (d * 11_111_111).clamp(LO, HI - 1)
            leadzero = aa[m] < LO
            if leadzero.any(): aa[m][leadzero] = 10_000_000
        # Sparse / dense zero-nine patterns, always full width.
        m = kind == 4
        if m.any():
            c = int(m.sum()); da = torch.randint(0, 2, (c, 8), device=DEVICE) * 9
            da[:, 7] = torch.randint(1, 10, (c,), device=DEVICE)
            aa[m] = (da * POW10[:8]).sum(1)
        # Near extrema.
        m = kind == 5
        if m.any():
            c = int(m.sum()); side = torch.randint(0, 2, (c,), device=DEVICE)
            aa[m] = torch.where(side.bool(), LO + torch.randint(0, 10000, (c,), device=DEVICE), HI - 1 - torch.randint(0, 10000, (c,), device=DEVICE))
        # Force carry beginning at varied columns.
        m = kind == 6
        if m.any():
            c = int(m.sum()); col = torch.randint(0, 8, (c,), device=DEVICE); p = POW10[col]
            ad = digits(aa[m]); bd = digits(bb[m]); idx = torch.arange(c, device=DEVICE)
            bd[idx, col] = 9 - ad[idx, col] + torch.randint(0, 2, (c,), device=DEVICE)
            bd[:, 7] = bd[:, 7].clamp_min(1)
            bb[m] = (bd * POW10[:8]).sum(1).clamp(LO, HI - 1)
        # Asymmetric high prefixes plus complementary suffixes.
        m = kind == 7
        if m.any():
            c = int(m.sum()); run = torch.randint(1, 8, (c,), device=DEVICE); p = POW10[run]
            x = torch.randint(LO, HI, (c,), device=DEVICE)
            aa[m] = ((x // p) * p + torch.randint(0, 10, (c,), device=DEVICE)).clamp(LO, HI - 1)
            suffix = p - 1 - (aa[m] % p)
            y = torch.randint(LO, HI, (c,), device=DEVICE)
            bb[m] = ((y // p) * p + suffix).clamp(LO, HI - 1)
        a[:n], b[:n] = aa, bb
    ad, bd = digits(a), digits(b)
    target = digits(a + b, 9)
    inp = torch.empty(batch, 25, dtype=torch.long, device=DEVICE)
    inp[:, 0:16:2], inp[:, 1:16:2] = ad, bd
    inp[:, 16] = 10
    inp[:, 17:] = target[:, :8]
    return inp, target


def autoregressive(model, a, b):
    ad, bd = digits(a), digits(b)
    x = torch.empty(a.shape[0], 17, dtype=torch.long, device=DEVICE)
    x[:, 0:16:2], x[:, 1:16:2], x[:, 16] = ad, bd, 10
    out = []
    for _ in range(9):
        d = model(x)[:, -1].argmax(-1)
        out.append(d)
        x = torch.cat((x, d[:, None]), 1)
    got = (torch.stack(out, 1) * POW10).sum(1)
    return got


@torch.no_grad()
def validate(model, count=100000, structured=0.0, chunk=20000):
    model.eval(); good = total = 0
    for _ in range((count + chunk - 1) // chunk):
        n = min(chunk, count-total)
        inp, target = make_batch(n, structured)
        # recover operands from input only for comparison
        a = (inp[:,0:16:2] * POW10[:8]).sum(1)
        b = (inp[:,1:16:2] * POW10[:8]).sum(1)
        good += int((autoregressive(model, a, b) == a+b).sum())
        total += n
    model.train(); return good, total


def train_stage(model, steps, lr, structured, label, validate_every=2000):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9,0.98), weight_decay=0.01, fused=True)
    model.train(); start=time.time()
    best = None; best_score=-1
    for step in range(1, steps+1):
        inp, target = make_batch(BATCH, structured)
        logits = model(inp)[:,16:25]
        loss = nn.functional.cross_entropy(logits.reshape(-1,10), target.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % validate_every == 0 or step == steps:
            vr=validate(model, 20000, 0.0); vs=validate(model, 20000, 0.65)
            score=vr[0]+vs[0]
            if score>best_score: best_score=score; best=copy.deepcopy(model.state_dict())
            print(label, step, f"loss={loss.item():.5f}", vr, vs, f"sec={time.time()-start:.1f}", flush=True)
    model.load_state_dict(best)
    return model


def save_checkpoint(model, name):
    torch.save(model.state_dict(), f"/workspace/{name}.pt")

def export(model):
    source=Path('/workspace/submission.py').read_text()
    vals=[]
    for p in model.parameters():
        flat=p.detach().float().cpu().reshape(-1).tolist()
        vals.append('['+','.join(format(x,'.9g') for x in flat)+']')
    source=source.replace('_TRAINED_STATE = None', '_TRAINED_STATE = [\n'+',\n'.join(vals)+'\n]')
    Path('/workspace/submission.py').write_text(source)

def main():
    torch.manual_seed(2025); random.seed(2025)
    torch.backends.cuda.matmul.allow_tf32=True
    model=AdditionTransformer().to(DEVICE)
    model=train_stage(model,18000,2e-3,0.18,'base',1000)
    model=train_stage(model,8000,5e-5,0.30,'polish',1000)
    save_checkpoint(model,'final'); print('FINAL',validate(model,500000,0.0),validate(model,500000,0.65),flush=True)
    export(model)

if __name__=='__main__': main()
