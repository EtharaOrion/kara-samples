import argparse
import math
import os
import random
import sys
import time
import torch
from torch import nn

D, H, HD = 20, 4, 5
class AdditionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token = nn.Embedding(11, D)
        self.position = nn.Parameter(torch.empty(25, D))
        self.attn_norm = nn.ModuleList([nn.LayerNorm(D) for _ in range(2)])
        self.ff_norm = nn.ModuleList([nn.LayerNorm(D) for _ in range(2)])
        self.q = nn.ModuleList([nn.Linear(D, D, bias=False) for _ in range(2)])
        self.k = nn.ModuleList([nn.Linear(D, D, bias=False) for _ in range(2)])
        self.v = nn.ModuleList([nn.Linear(D, D, bias=False) for _ in range(2)])
        self.o = nn.ModuleList([nn.Linear(D, D, bias=False) for _ in range(2)])
        self.ff1 = nn.ModuleList([nn.Linear(D, 4, bias=False) for _ in range(2)])
        self.ff2 = nn.ModuleList([nn.Linear(4, D, bias=False) for _ in range(2)])
        self.final_norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, 10, bias=False)
        nn.init.normal_(self.position, std=0.02)
    def forward(self, tokens):
        length=tokens.shape[1]
        x=self.token(tokens)+self.position[:length]
        mask=torch.ones(length,length,device=x.device,dtype=torch.bool).tril()
        for layer in range(2):
            z=self.attn_norm[layer](x)
            q=self.q[layer](z).view(*z.shape[:2],H,HD).transpose(1,2)
            k=self.k[layer](z).view(*z.shape[:2],H,HD).transpose(1,2)
            v=self.v[layer](z).view(*z.shape[:2],H,HD).transpose(1,2)
            scores=(q@k.transpose(-2,-1))*(HD**-0.5)
            scores=scores.masked_fill(~mask,-torch.inf)
            attended=(scores.softmax(-1)@v).transpose(1,2).reshape_as(x)
            x=x+self.o[layer](attended)
            x=x+self.ff2[layer](nn.functional.gelu(self.ff1[layer](self.ff_norm[layer](x))))
        return self.head(self.final_norm(x))

MINV = 10_000_000
MAXV = 99_999_999
POW = torch.tensor([10 ** i for i in range(9)], device="cuda", dtype=torch.long)


def digits(x, n):
    return (x[:, None] // POW[None, :n]) % 10


def make_batch(batch, structured=0.18):
    a = torch.randint(MINV, MAXV + 1, (batch,), device="cuda")
    b = torch.randint(MINV, MAXV + 1, (batch,), device="cuda")
    n = int(batch * structured)
    if not n:
        return encode(a, b)
    # Equal-size slices from diverse carry and boundary families.
    kind = torch.randint(0, 8, (n,), device="cuda")
    aa = torch.randint(MINV, MAXV + 1, (n,), device="cuda")
    bb = torch.randint(MINV, MAXV + 1, (n,), device="cuda")
    for k in range(8):
        ix = kind == k
        m = int(ix.sum())
        if not m:
            continue
        x = torch.randint(MINV, MAXV + 1, (m,), device="cuda")
        y = torch.randint(MINV, MAXV + 1, (m,), device="cuda")
        if k == 0:  # exact and near complements to 100m
            delta = torch.randint(-20, 21, (m,), device="cuda")
            y = 100_000_000 - x + delta
        elif k == 1:  # long suffix of nines
            p = torch.randint(1, 8, (m,), device="cuda")
            base = POW[p]
            x = (x // base) * base + base - 1
            y = torch.randint(MINV, MAXV + 1, (m,), device="cuda")
        elif k == 2:  # rounded operand plus nearby boundary
            p = torch.randint(1, 8, (m,), device="cuda")
            base = POW[p]
            x = (x // base) * base
            y = torch.randint(MINV, MAXV + 1, (m,), device="cuda")
            y = y + torch.randint(-2, 3, (m,), device="cuda")
        elif k == 3:  # asymmetric complementary low suffixes
            p = torch.randint(1, 8, (m,), device="cuda")
            base = POW[p]
            low = torch.randint(0, 20, (m,), device="cuda")
            x = (x // base) * base + low
            y = (y // base) * base + (base - 1 - low)
        elif k == 4:  # repeated digits
            d1 = torch.randint(0, 10, (m,), device="cuda")
            d2 = torch.randint(0, 10, (m,), device="cuda")
            x = d1 * 11_111_111
            y = d2 * 11_111_111
            x = torch.where(x < MINV, x + 11_111_111, x)
            y = torch.where(y < MINV, y + 11_111_111, y)
        elif k == 5:  # sparse decimal values
            lead1 = torch.randint(1, 10, (m,), device="cuda")
            lead2 = torch.randint(1, 10, (m,), device="cuda")
            q1 = torch.randint(0, 7, (m,), device="cuda")
            q2 = torch.randint(0, 7, (m,), device="cuda")
            x = lead1 * 10_000_000 + POW[q1] * torch.randint(0, 10, (m,), device="cuda")
            y = lead2 * 10_000_000 + POW[q2] * torch.randint(0, 10, (m,), device="cuda")
        elif k == 6:  # extrema neighborhoods
            side = torch.randint(0, 2, (m,), device="cuda")
            off = torch.randint(0, 10001, (m,), device="cuda")
            x = torch.where(side.bool(), MAXV - off, MINV + off)
            y = torch.randint(MINV, MAXV + 1, (m,), device="cuda")
        else:  # all-nine runs on both sides, arbitrary leading digits
            p = torch.randint(2, 8, (m,), device="cuda")
            base = POW[p]
            x = (x // base) * base + base - torch.randint(1, 11, (m,), device="cuda")
            y = (y // base) * base + torch.randint(1, 11, (m,), device="cuda")
        x = x.clamp(MINV, MAXV)
        y = y.clamp(MINV, MAXV)
        aa[ix], bb[ix] = x, y
    a[:n], b[:n] = aa, bb
    return encode(a, b)


def encode(a, b):
    batch = a.shape[0]
    ab = torch.stack((digits(a, 8), digits(b, 8)), dim=2).reshape(batch, 16)
    ans = digits(a + b, 9)
    inp = torch.cat((ab, torch.full((batch, 1), 10, device="cuda", dtype=torch.long), ans[:, :8]), 1)
    return inp, ans


@torch.no_grad()
def evaluate(model, count=100000, structured=0.0, chunk=10000, autoregressive=True):
    model.eval()
    good = total = 0
    min_margin = 1e9
    while total < count:
        n = min(chunk, count-total)
        inp, target = make_batch(n, structured)
        if autoregressive:
            toks = inp[:, :17]
            preds = []
            for _ in range(9):
                log = model(toks)[:, -1]
                top = log.topk(2, 1).values
                min_margin = min(min_margin, float((top[:,0]-top[:,1]).min()))
                d = log.argmax(1)
                preds.append(d)
                toks = torch.cat((toks, d[:,None]), 1)
            pred = torch.stack(preds, 1)
        else:
            pred = model(inp)[:, 16:25].argmax(2)
        good += int((pred == target).all(1).sum())
        total += n
    model.train()
    return good, total, min_margin


def run_stage(model, steps, lr, structured, label, batch=8192, eval_every=2000):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9,0.98), weight_decay=0.01)
    start = time.time()
    for step in range(1, steps+1):
        inp, target = make_batch(batch, structured)
        logits = model(inp)[:,16:25]
        loss = nn.functional.cross_entropy(logits.reshape(-1,10), target.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % eval_every == 0 or step == steps:
            # Teacher-forced diagnostic; autoregressive check at key points.
            g,t,_ = evaluate(model, 20000, 0.0, autoregressive=False)
            msg=f"{label} {step}/{steps} loss={loss.item():.5f} tf={g}/{t} elapsed={time.time()-start:.1f}s"
            if step == steps or g == t:
                ag,at,margin=evaluate(model, 50000, 0.0, autoregressive=True)
                sg,st,_=evaluate(model, 50000, 0.7, autoregressive=True)
                msg += f" ar={ag}/{at} struct={sg}/{st} margin={margin:.3f}"
            print(msg, flush=True)
            torch.save({"model":model.state_dict(),"width":4,"stage":label,"step":step}, f"/workspace/{label}.pt")
    return model


def export(model):
    path='/workspace/submission.py'
    text=open(path).read()
    state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    literal="{\n"+",\n".join(repr(k)+": torch.tensor("+repr(v.tolist())+")" for k,v in state.items())+"\n}"
    text=text.replace('_TRAINED_STATE = None', '_TRAINED_STATE = '+literal)
    with open(path+'.new','w') as f: f.write(text)
    os.replace(path+'.new',path)
    print('exported',path,'parameters',sum(p.numel() for p in model.parameters()),'bytes',os.path.getsize(path),flush=True)


def main():
    torch.manual_seed(2025); random.seed(2025)
    torch.backends.cuda.matmul.allow_tf32=True
    model=AdditionTransformer().cuda()
    print('initial params',sum(p.numel() for p in model.parameters()),flush=True)
    model=run_stage(model,18000,2e-3,0.18,'fallback',eval_every=2000)
    model=run_stage(model,8000,5e-5,0.30,'fallback_polish',eval_every=2000)
    export(model)
    g,t,m=evaluate(model,500000,0.0); s,u,_=evaluate(model,500000,0.7)
    print('FINAL',g,t,s,u,'margin',m,flush=True)
    torch.save({'model':model.state_dict()},'/workspace/final.pt')

if __name__=='__main__': main()
