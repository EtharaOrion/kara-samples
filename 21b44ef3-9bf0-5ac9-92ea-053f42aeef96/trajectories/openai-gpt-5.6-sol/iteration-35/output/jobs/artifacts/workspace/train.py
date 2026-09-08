import math
import random
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

DEVICE = "cuda"
D, H, DH, N = 20, 4, 5, 25
BATCH = 8192
LOW, HIGH = 10_000_000, 99_999_999
POW = torch.tensor([10 ** i for i in range(9)], device=DEVICE)


class Model(nn.Module):
    def __init__(self, ff=4):
        super().__init__()
        self.ff = ff
        self.token = nn.Embedding(11, D)
        self.pos_a = nn.Parameter(torch.randn(N, 2) * .08)
        self.pos_b = nn.Parameter(torch.randn(2, D) * .08)
        self.q = nn.ParameterList([nn.Parameter(torch.randn(D, D) * .08) for _ in range(2)])
        self.o = nn.ParameterList([nn.Parameter(torch.randn(D, D) * .08) for _ in range(2)])
        self.k = nn.Parameter(torch.randn(D, DH) * .08)
        self.v = nn.Parameter(torch.randn(D, DH) * .08)
        self.attn_norm = nn.LayerNorm(D)
        self.ff_norm = nn.LayerNorm(D)
        self.ff1 = nn.Linear(D, ff)
        self.ff2 = nn.Linear(ff, D, bias=False)
        self.final_norm = nn.LayerNorm(D)
        self.classifier = nn.Linear(D, 10)
        self.register_buffer("mask", torch.triu(torch.ones(N, N, dtype=torch.bool), 1), persistent=False)

    def forward(self, t):
        L = t.shape[1]
        x = self.token(t) + self.pos_a[:L] @ self.pos_b
        for i in range(2):
            z = self.attn_norm(x)
            q = (z @ self.q[i]).view(-1, L, H, DH).transpose(1, 2)
            k, v = (z @ self.k).unsqueeze(1), (z @ self.v).unsqueeze(1)
            s = (q @ k.transpose(-1, -2)) * (DH ** -.5)
            s = s.masked_fill(self.mask[:L, :L], -1e4)
            x = x + (s.softmax(-1) @ v).transpose(1, 2).reshape(-1, L, D) @ self.o[i]
            x = x + self.ff2(F.gelu(self.ff1(self.ff_norm(x))))
        return self.classifier(self.final_norm(x))


def digits(x, n=8):
    return (x[:, None] // POW[:n]) % 10


def uniform(n):
    return (torch.randint(LOW, HIGH + 1, (n,), device=DEVICE),
            torch.randint(LOW, HIGH + 1, (n,), device=DEVICE))


def batch(n=BATCH, structured=.40):
    a, b = uniform(n)
    m = int(n * structured)
    if not m:
        return a, b
    typ = torch.randint(0, 7, (m,), device=DEVICE)
    aa, bb = uniform(m)
    # Complements to 1e8 with a modest signed displacement.
    ix = typ == 0
    x = torch.randint(LOW, HIGH + 1, (int(ix.sum()),), device=DEVICE)
    delta = torch.randint(-1000, 1001, x.shape, device=DEVICE)
    aa[ix] = x
    bb[ix] = (100_000_000 - x + delta).clamp(LOW, HIGH)
    # Long asymmetric 0/9 suffixes, with arbitrary prefixes.
    for code, fill_a, fill_b in [(1, 9, 0), (2, 0, 9)]:
        ix = typ == code
        count = int(ix.sum())
        run = torch.randint(1, 8, (count,), device=DEVICE)
        p10 = 10 ** run
        xa = torch.randint(LOW, HIGH + 1, (count,), device=DEVICE)
        xb = torch.randint(LOW, HIGH + 1, (count,), device=DEVICE)
        if fill_a == 9:
            xa = (xa // p10) * p10 + p10 - 1
            xb = (xb // p10) * p10
        else:
            xa = (xa // p10) * p10
            xb = (xb // p10) * p10 + p10 - 1
        aa[ix], bb[ix] = xa.clamp(LOW, HIGH), xb.clamp(LOW, HIGH)
    # Near decimal boundaries at arbitrary scales.
    ix = typ == 3
    count = int(ix.sum())
    scale = 10 ** torch.randint(1, 8, (count,), device=DEVICE)
    x = torch.randint(LOW, HIGH + 1, (count,), device=DEVICE)
    aa[ix] = ((x // scale) * scale + torch.randint(-12, 13, (count,), device=DEVICE)).clamp(LOW, HIGH)
    # Repeated digits.
    ix = typ == 4
    count = int(ix.sum())
    rep = torch.randint(1, 10, (count,), device=DEVICE) * 11_111_111
    aa[ix] = rep
    bb[ix] = torch.randint(1, 10, (count,), device=DEVICE) * 11_111_111
    # Sparse interior digit patterns.
    ix = typ == 5
    count = int(ix.sum())
    da = torch.zeros(count, 8, dtype=torch.long, device=DEVICE)
    db = torch.zeros_like(da)
    da[:, 7] = torch.randint(1, 10, (count,), device=DEVICE)
    db[:, 7] = torch.randint(1, 10, (count,), device=DEVICE)
    for ds in (da, db):
        cols = torch.randint(0, 7, (count, 2), device=DEVICE)
        vals = torch.randint(0, 10, (count, 2), device=DEVICE)
        ds.scatter_(1, cols, vals)
    aa[ix], bb[ix] = (da * POW[:8]).sum(1), (db * POW[:8]).sum(1)
    # Near extrema.
    ix = typ == 6
    count = int(ix.sum())
    side = torch.randint(0, 2, (count,), device=DEVICE)
    aa[ix] = torch.where(side.bool(), HIGH - torch.randint(0, 10000, (count,), device=DEVICE), LOW + torch.randint(0, 10000, (count,), device=DEVICE))
    a[:m], b[:m] = aa, bb
    return a, b


def encode(a, b):
    da, db = digits(a), digits(b)
    y = digits(a + b, 9)
    x = torch.empty(a.shape[0], 25, dtype=torch.long, device=DEVICE)
    x[:, :16:2], x[:, 1:16:2] = da, db
    x[:, 16] = 10
    x[:, 17:] = y[:, :8]
    return x, y


@torch.no_grad()
def accuracy(model, count=100000, structured=.0, chunk=10000):
    model.eval(); good = total = 0; minmargin = 1e9
    for _ in range((count + chunk - 1) // chunk):
        n = min(chunk, count-total)
        a,b = batch(n, structured)
        da,db = digits(a),digits(b)
        seq=torch.empty(n,17,dtype=torch.long,device=DEVICE)
        seq[:,:16:2],seq[:,1:16:2],seq[:,16]=da,db,10
        out=[]
        for j in range(9):
            z=model(seq)[:,-1]
            top=z.topk(2,1).values
            minmargin=min(minmargin,float((top[:,0]-top[:,1]).min()))
            d=z.argmax(1); out.append(d); seq=torch.cat((seq,d[:,None]),1)
        pred=(torch.stack(out,1)*torch.tensor([10**i for i in range(9)],device=DEVICE)).sum(1)
        good += int((pred==a+b).sum()); total += n
    model.train(); return good,total,minmargin


def train_stage(model, steps, lr, structured, name, warmup=300):
    model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=lr,betas=(.9,.98),weight_decay=.01)
    for step in range(1,steps+1):
        a,b=batch(BATCH,structured); x,y=encode(a,b)
        frac=min(1.,step/warmup) * (0.15 + .85*.5*(1+math.cos(math.pi*step/steps)))
        for g in opt.param_groups:g['lr']=lr*frac
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(x)[:,16:25]
            loss=F.cross_entropy(logits.reshape(-1,10),y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step%2000==0 or step==steps:
            print(name,step,float(loss),flush=True)
            torch.save(model.state_dict(),f'/workspace/{name}.pt')
    return model


def prune(model):
    old=model
    score=old.ff1.weight.norm(dim=1)*old.ff2.weight.norm(dim=0)
    keep=torch.argsort(score,descending=True)[:-1].sort().values
    new=Model(old.ff-1).to(DEVICE)
    sd=old.state_dict(); ns=new.state_dict()
    for k in ns:
        if k=='ff1.weight': ns[k].copy_(sd[k][keep])
        elif k=='ff1.bias': ns[k].copy_(sd[k][keep])
        elif k=='ff2.weight': ns[k].copy_(sd[k][:,keep])
        else: ns[k].copy_(sd[k])
    return new


def export(model):
    sd={k:v.detach().float().cpu() for k,v in model.state_dict().items()}
    # Fold affine LayerNorms into their following linear maps. The inference
    # module retains the normalization itself but needs no learned affine.
    fw = sd['ff1.weight'] * sd['ff_norm.weight'][None, :]
    fb = sd['ff1.bias'] + sd['ff1.weight'] @ sd['ff_norm.bias']
    cw = sd['classifier.weight'] * sd['final_norm.weight'][None, :]
    cb = sd['classifier.bias'] + sd['classifier.weight'] @ sd['final_norm.bias']
    del sd['ff_norm.weight'], sd['ff_norm.bias'], sd['final_norm.weight'], sd['final_norm.bias']
    sd['ff1.weight'], sd['ff1.bias'] = fw, fb
    sd['classifier.weight'], sd['classifier.bias'] = cw, cb
    base=Path('/workspace/submission.py').read_text()
    base=base[:base.index('_STATE =')]
    def lit(t): return 'torch.tensor('+repr(t.tolist())+', dtype=torch.float32)'
    state='_STATE = {\n'+''.join('    '+repr(k)+': '+lit(v)+',\n' for k,v in sd.items())+'}\n'
    temp=Path('/workspace/submission.new.py'); temp.write_text(base+state); temp.replace('/workspace/submission.py')


if __name__=='__main__':
    torch.manual_seed(350034); random.seed(350034)
    torch.set_float32_matmul_precision('high')
    m=Model(4).to(DEVICE)
    m=train_stage(m,36000,2e-3,.42,'teacher4')
    print('teacher val',accuracy(m,100000,0),accuracy(m,100000,.7),flush=True)
    m=prune(m); m=train_stage(m,18000,3e-5,.48,'width3')
    print('w3 val',accuracy(m,100000,0),accuracy(m,100000,.7),flush=True)
    m=prune(m); m=train_stage(m,30000,1.5e-5,.50,'width2a')
    m=train_stage(m,16000,4e-6,.58,'width2b',100)
    print('FINAL',accuracy(m,500000,0),accuracy(m,500000,.75),flush=True)
    torch.save(m.state_dict(),'/workspace/final.pt'); export(m)
