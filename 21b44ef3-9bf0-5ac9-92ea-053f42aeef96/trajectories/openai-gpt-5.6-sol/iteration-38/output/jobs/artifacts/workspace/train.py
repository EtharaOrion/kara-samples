import math
import random
import sys
from pathlib import Path

sys.path.append("/usr/local/lib/python3.11/dist-packages")
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, "/workspace")
from submission import AdditionTransformer

DEVICE = "cuda"
BATCH = 8192
LO = 10_000_000
HI = 99_999_999
BASE = 100_000_000


def digits(x):
    places = torch.tensor([10 ** i for i in range(9)], device=x.device)
    return (x[:, None] // places % 10).long()


def make_uniform(n):
    a = torch.randint(LO, HI + 1, (n,), device=DEVICE)
    b = torch.randint(LO, HI + 1, (n,), device=DEVICE)
    return a, b


def make_structured(n):
    # Broad mixture emphasizing every carry length without sacrificing diversity.
    a, b = make_uniform(n)
    kind = torch.randint(0, 8, (n,), device=DEVICE)
    k = torch.randint(1, 9, (n,), device=DEVICE)
    p10 = torch.tensor([10 ** i for i in range(10)], device=DEVICE)
    p = p10[k]

    idx = kind == 0  # exact/near complements
    if idx.any():
        aa = torch.randint(LO, 90_000_001, (int(idx.sum()),), device=DEVICE)
        delta = torch.randint(-25, 26, aa.shape, device=DEVICE)
        bb = BASE - aa + delta
        a[idx], b[idx] = aa, bb.clamp(LO, HI)

    idx = kind == 1  # controlled trailing-nine run in one operand
    if idx.any():
        m = int(idx.sum()); pp = p[idx]
        prefix = torch.randint(1, 100_000_000, (m,), device=DEVICE)
        aa = (prefix // pp) * pp + pp - 1
        aa = aa.clamp(LO, HI)
        bb = torch.randint(LO, HI + 1, (m,), device=DEVICE)
        a[idx], b[idx] = aa, bb

    idx = kind == 2  # suffixes adding exactly to a power of ten
    if idx.any():
        m = int(idx.sum()); pp = p[idx]
        suffix = torch.minimum((torch.rand(m, device=DEVICE) * pp.float()).long(), pp - 1)
        pa = torch.randint(1, 10_000_000, (m,), device=DEVICE)
        pb = torch.randint(1, 10_000_000, (m,), device=DEVICE)
        aa = (pa * pp + suffix)
        bb = (pb * pp + (pp - suffix) % pp)
        good = (aa >= LO) & (aa <= HI) & (bb >= LO) & (bb <= HI)
        olda, oldb = a[idx], b[idx]
        a[idx], b[idx] = torch.where(good, aa, olda), torch.where(good, bb, oldb)

    idx = kind == 3  # decimal boundaries plus small offsets
    if idx.any():
        m = int(idx.sum()); pp = p[idx]
        mult = torch.randint(1, 100_000_000, (m,), device=DEVICE)
        off = torch.randint(-50, 51, (m,), device=DEVICE)
        aa = (mult // pp) * pp + off
        a[idx] = aa.clamp(LO, HI)

    idx = kind == 4  # repeated digits
    if idx.any():
        m = int(idx.sum())
        da = torch.randint(1, 10, (m,), device=DEVICE)
        db = torch.randint(1, 10, (m,), device=DEVICE)
        a[idx], b[idx] = da * 11_111_111, db * 11_111_111

    idx = kind == 5  # sparse internal digits
    if idx.any():
        m = int(idx.sum())
        lead1 = torch.randint(1, 10, (m,), device=DEVICE) * 10_000_000
        lead2 = torch.randint(1, 10, (m,), device=DEVICE) * 10_000_000
        place1 = torch.randint(0, 7, (m,), device=DEVICE)
        place2 = torch.randint(0, 7, (m,), device=DEVICE)
        a[idx] = lead1 + torch.randint(0, 10, (m,), device=DEVICE) * p10[place1]
        b[idx] = lead2 + torch.randint(0, 10, (m,), device=DEVICE) * p10[place2]

    idx = kind == 6  # extrema neighborhoods
    if idx.any():
        m = int(idx.sum())
        side = torch.randint(0, 2, (m,), device=DEVICE).bool()
        vals = torch.randint(0, 10000, (m,), device=DEVICE)
        a[idx] = torch.where(side, LO + vals, HI - vals)

    return a, b


def batch(structured_fraction=0.18):
    a, b = make_uniform(BATCH)
    n = int(BATCH * structured_fraction)
    if n:
        a[:n], b[:n] = make_structured(n)
    da, db, ds = digits(a), digits(b), digits(a + b)
    x = torch.empty((BATCH, 25), dtype=torch.long, device=DEVICE)
    x[:, 0:16:2] = da[:, :8]
    x[:, 1:16:2] = db[:, :8]
    x[:, 16] = 10
    x[:, 17:] = ds[:, :8]
    return x, ds


def accuracy(model, n=100000, structured=False, chunk=10000):
    model.eval(); correct = total = 0; min_margin = 1e9
    with torch.no_grad():
        for _ in range((n + chunk - 1) // chunk):
            m = min(chunk, n-total)
            a, b = make_structured(m) if structured else make_uniform(m)
            da, db = digits(a), digits(b)
            seq = torch.empty((m, 17), dtype=torch.long, device=DEVICE)
            seq[:, 0:16:2], seq[:, 1:16:2], seq[:, 16] = da[:,:8], db[:,:8], 10
            outs=[]
            for j in range(9):
                logits=model(seq)[:,-1]
                top=logits.topk(2,dim=-1).values
                min_margin=min(min_margin,float((top[:,0]-top[:,1]).min()))
                d=logits.argmax(-1); outs.append(d)
                seq=torch.cat((seq,d[:,None]),1)
            pred=torch.stack(outs,1)
            correct += int((pred == digits(a+b)).all(1).sum()); total += m
    model.train()
    return correct, total, min_margin


def train_stage(model, steps, lr, structured, tag):
    model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=lr,betas=(0.9,0.98),weight_decay=0.01, fused=True)
    for step in range(1,steps+1):
        x,y=batch(structured)
        logits=model(x)[:,16:25]
        loss=F.cross_entropy(logits.reshape(-1,10),y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step % 1000 == 0 or step == steps:
            print(tag,step,float(loss),flush=True)
        if step % 12000 == 0 or step == steps:
            print(" eval",accuracy(model,20000,False),accuracy(model,20000,True),flush=True)
            torch.save(model.state_dict(),f"/workspace/{tag}_latest.pt")
    return model


def prune(model, width):
    old=model.ff1.out_features
    if old == width: return model
    score=model.ff1.weight.norm(dim=1)*model.ff2.weight.norm(dim=0)
    keep=score.topk(width).indices.sort().values
    new=AdditionTransformer(width).to(DEVICE)
    state=model.state_dict(); ns=new.state_dict()
    for k in ns:
        if k.startswith("ff1") or k.startswith("ff2"): continue
        ns[k].copy_(state[k])
    new.ff1=nn.Linear(20,width,bias=False).to(DEVICE)
    new.ff2=nn.Linear(width,20,bias=False).to(DEVICE)
    with torch.no_grad():
        new.ff1.weight.copy_(model.ff1.weight[keep]); new.ff2.weight.copy_(model.ff2.weight[:,keep])
    return new


def export(model):
    vals=[]
    for p in model.parameters(): vals.append(p.detach().cpu().flatten().tolist())
    path=Path('/workspace/submission.py')
    text=path.read_text(); text=text.replace('_TRAINED_STATE = None','_TRAINED_STATE = '+repr(vals))
    path.write_text(text)


def main():
    torch.manual_seed(2025); random.seed(2025)
    model=AdditionTransformer(4).to(DEVICE)
    model.load_state_dict(torch.load('/workspace/teacher.pt', weights_only=True))
    print('TEACHER',accuracy(model,100000,False),accuracy(model,100000,True),flush=True)
    model=prune(model,3)
    train_stage(model,18000,2e-5,0.30,'w3')
    model=prune(model,2)
    train_stage(model,30000,1.2e-5,0.30,'w2')
    train_stage(model,12000,4e-6,0.40,'w2_polish')
    print('FINAL',accuracy(model,500000,False),accuracy(model,500000,True),flush=True)
    torch.save(model.state_dict(),'/workspace/final.pt'); export(model)

if __name__=='__main__': main()
