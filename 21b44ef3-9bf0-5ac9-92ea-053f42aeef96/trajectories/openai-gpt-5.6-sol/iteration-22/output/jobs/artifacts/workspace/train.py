import math
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, "/workspace")
from submission import AdditionTransformer

DEVICE = "cuda"
BATCH = 8192
LOW = 10_000_000
HIGH = 99_999_999
POW8 = torch.tensor([10 ** i for i in range(8)], device=DEVICE, dtype=torch.long)
POW9 = torch.tensor([10 ** i for i in range(9)], device=DEVICE, dtype=torch.long)


def structured(n):
    a = torch.randint(LOW, HIGH + 1, (n,), device=DEVICE)
    b = torch.randint(LOW, HIGH + 1, (n,), device=DEVICE)
    mode = torch.randint(0, 8, (n,), device=DEVICE)

    # Exact and near complements exercise carries through all eight columns.
    ix = mode == 0
    if ix.any():
        aa = torch.randint(LOW, 90_000_001, (int(ix.sum()),), device=DEVICE)
        delta = torch.randint(-9, 10, aa.shape, device=DEVICE)
        a[ix], b[ix] = aa, (100_000_000 - aa + delta).clamp(LOW, HIGH)

    # Asymmetric suffixes of zeroes/nines at every possible length.
    ix = mode == 1
    if ix.any():
        m = int(ix.sum()); run = torch.randint(1, 8, (m,), device=DEVICE)
        p = 10 ** run
        aa = torch.randint(LOW, HIGH + 1, (m,), device=DEVICE)
        bb = torch.randint(LOW, HIGH + 1, (m,), device=DEVICE)
        choose = torch.rand(m, device=DEVICE) < .5
        aa = torch.where(choose, (aa // p) * p + p - 1, (aa // p) * p)
        bb = torch.where(choose, (bb // p) * p, (bb // p) * p + p - 1)
        a[ix], b[ix] = aa.clamp(LOW, HIGH), bb.clamp(LOW, HIGH)

    # Values around decimal boundaries.
    ix = mode == 2
    if ix.any():
        m = int(ix.sum()); run = torch.randint(1, 8, (m,), device=DEVICE)
        p = 10 ** run
        base_a = torch.randint(1, 100_000_000, (m,), device=DEVICE) // p * p
        base_b = torch.randint(1, 100_000_000, (m,), device=DEVICE) // p * p
        da = torch.randint(-12, 13, (m,), device=DEVICE)
        db = torch.randint(-12, 13, (m,), device=DEVICE)
        a[ix], b[ix] = (base_a + da).clamp(LOW, HIGH), (base_b + db).clamp(LOW, HIGH)

    # Repeated digit operands.
    ix = mode == 3
    if ix.any():
        m = int(ix.sum()); d1 = torch.randint(1, 10, (m,), device=DEVICE); d2 = torch.randint(1, 10, (m,), device=DEVICE)
        a[ix], b[ix] = d1 * 11_111_111, d2 * 11_111_111

    # Sparse internal digits with valid nonzero leading digit.
    ix = mode == 4
    if ix.any():
        m = int(ix.sum()); lead1 = torch.randint(1, 10, (m,), device=DEVICE); lead2 = torch.randint(1, 10, (m,), device=DEVICE)
        pos1 = torch.randint(0, 7, (m,), device=DEVICE); pos2 = torch.randint(0, 7, (m,), device=DEVICE)
        dig1 = torch.randint(0, 10, (m,), device=DEVICE); dig2 = torch.randint(0, 10, (m,), device=DEVICE)
        a[ix] = lead1 * 10_000_000 + dig1 * (10 ** pos1)
        b[ix] = lead2 * 10_000_000 + dig2 * (10 ** pos2)

    # Near extrema.
    ix = mode == 5
    if ix.any():
        m = int(ix.sum()); span = torch.randint(0, 100_000, (m,), device=DEVICE)
        side = torch.rand(m, device=DEVICE) < .5
        a[ix] = torch.where(side, LOW + span, HIGH - span)
        b[ix] = torch.randint(LOW, HIGH + 1, (m,), device=DEVICE)

    # Force a carry to begin at an arbitrary column and propagate through nines.
    ix = mode == 6
    if ix.any():
        m = int(ix.sum()); run = torch.randint(1, 8, (m,), device=DEVICE); p = 10 ** run
        aa = torch.randint(LOW, HIGH + 1, (m,), device=DEVICE)
        lowpart = torch.randint(1, 10, (m,), device=DEVICE)
        aa = (aa // p) * p + p - lowpart
        bb = torch.randint(LOW, HIGH + 1, (m,), device=DEVICE)
        bb = (bb // p) * p + lowpart
        a[ix], b[ix] = aa.clamp(LOW, HIGH), bb.clamp(LOW, HIGH)

    # Same/symmetric operands and high sums.
    ix = mode == 7
    if ix.any():
        m = int(ix.sum()); aa = torch.randint(45_000_000, HIGH + 1, (m,), device=DEVICE)
        noise = torch.randint(-100, 101, (m,), device=DEVICE)
        a[ix], b[ix] = aa, (aa + noise).clamp(LOW, HIGH)
    return a, b


def batch(structured_fraction):
    n = BATCH
    a = torch.randint(LOW, HIGH + 1, (n,), device=DEVICE)
    b = torch.randint(LOW, HIGH + 1, (n,), device=DEVICE)
    count = int(n * structured_fraction)
    if count:
        sa, sb = structured(count)
        a[:count], b[:count] = sa, sb
    ad = (a[:, None] // POW8) % 10
    bd = (b[:, None] // POW8) % 10
    out = (a + b)[:, None] // POW9 % 10
    tokens = torch.empty(n, 25, dtype=torch.long, device=DEVICE)
    tokens[:, :16:2], tokens[:, 1:16:2] = ad, bd
    tokens[:, 16] = 10
    tokens[:, 17:] = out[:, :8]
    return tokens, out


def accuracy(model, total, kind="random"):
    model.eval(); good = 0; margin = 100.0
    with torch.no_grad():
        for start in range(0, total, BATCH):
            n = min(BATCH, total-start)
            if kind == "random":
                a = torch.randint(LOW, HIGH + 1, (n,), device=DEVICE); b = torch.randint(LOW, HIGH + 1, (n,), device=DEVICE)
            else:
                a, b = structured(n)
            ad=(a[:,None]//POW8)%10; bd=(b[:,None]//POW8)%10; truth=(a+b)[:,None]//POW9%10
            seq=torch.empty(n,17,dtype=torch.long,device=DEVICE); seq[:,:16:2]=ad; seq[:,1:16:2]=bd; seq[:,16]=10
            pred=[]
            for _ in range(9):
                logits=model(seq)[:,-1]
                top=logits.topk(2,dim=-1).values
                margin=min(margin, float((top[:,0]-top[:,1]).min()))
                digit=logits.argmax(-1); pred.append(digit); seq=torch.cat((seq,digit[:,None]),1)
            guess=torch.stack(pred,1)
            good += int((guess == truth).all(1).sum())
    model.train(); return good, total, margin


def main():
    torch.manual_seed(220022)
    torch.set_float32_matmul_precision("high")
    model = AdditionTransformer().to(DEVICE)
    print("parameters", sum(p.numel() for p in model.parameters()), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(.9,.98), weight_decay=.01, fused=True)
    start = time.time(); best = 0
    steps = 44000
    for step in range(1, steps+1):
        if step <= 1000: lr=2e-3*step/1000
        elif step <= 34000: lr=2e-3 * (0.05 + .95*.5*(1+math.cos(math.pi*(step-1000)/33000)))
        elif step <= 40000: lr=5e-5
        else: lr=2e-5
        for group in optimizer.param_groups: group["lr"] = lr
        frac = .30 if step < 30000 else .50
        x,y=batch(frac)
        logits=model(x)[:,16:25]
        loss=F.cross_entropy(logits.reshape(-1,10), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step % 1000 == 0:
            print(step, f"loss={loss.item():.6g}", f"lr={lr:.2g}", f"sec={time.time()-start:.1f}", flush=True)
        if step in (30000, 34000, 38000, 42000, 44000):
            r=accuracy(model,50000,"random"); s=accuracy(model,50000,"structured")
            score=min(r[0]/r[1],s[0]/s[1])
            print("VALID",step,r,s,flush=True)
            torch.save({"model":model.state_dict(),"step":step,"random":r,"structured":s},f"/workspace/checkpoint_{step}.pt")
            if score >= best:
                best=score; torch.save(model.state_dict(),"/workspace/best.pt")
    r=accuracy(model,500000,"random"); s=accuracy(model,500000,"structured")
    print("FINAL",r,s,flush=True)
    torch.save(model.state_dict(),"/workspace/final.pt")

if __name__ == "__main__": main()
