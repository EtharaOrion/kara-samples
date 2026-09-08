import argparse
import math
import random
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

D = 32
FF = 32
HEADS = 4
LAYERS = 2
SEQ = 25
LO = 10_000_000
HI = 99_999_999
POW10 = torch.tensor([10 ** i for i in range(9)], dtype=torch.long)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(D, HEADS, batch_first=True)
        self.n2 = nn.LayerNorm(D)
        self.ff1 = nn.Linear(D, FF)
        self.ff2 = nn.Linear(FF, D)

    def forward(self, x):
        y = self.n1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        return x + self.ff2(F.gelu(self.ff1(self.n2(x))))


class AdditionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token = nn.Embedding(11, D)
        self.position = nn.Parameter(torch.empty(SEQ, D))
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, 10, bias=False)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, tokens):
        x = self.token(tokens) + self.position
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x[:, 16:]))


def digits(x, n=8):
    p = POW10[:n].to(x.device)
    return (x[:, None] // p[None, :]) % 10


def make_batch(n, device, structured=0.25):
    a = torch.randint(LO, HI + 1, (n,), device=device)
    b = torch.randint(LO, HI + 1, (n,), device=device)
    k = int(n * structured)
    if k:
        kind = torch.randint(0, 8, (k,), device=device)
        x = torch.randint(LO, HI + 1, (k,), device=device)
        sa, sb = x.clone(), torch.randint(LO, HI + 1, (k,), device=device)

        m = kind == 0  # exact and nearby complements to 1e8
        delta = torch.randint(-20, 21, (k,), device=device)
        sb[m] = (100_000_000 - sa[m] + delta[m]).clamp(LO, HI)

        m = kind == 1  # long trailing runs of 9 plus small perturbations
        run = torch.randint(1, 8, (k,), device=device)
        scale = (10 ** run).long()
        sa[m] = ((x[m] // scale[m]) * scale[m] + scale[m] - 1).clamp(LO, HI)
        sb[m] = torch.randint(LO, HI + 1, (int(m.sum()),), device=device)

        m = kind == 2  # decimal boundaries
        run = torch.randint(1, 8, (k,), device=device)
        scale = (10 ** run).long()
        off = torch.randint(-3, 4, (k,), device=device)
        sa[m] = ((x[m] // scale[m]) * scale[m] + off[m]).clamp(LO, HI)

        m = kind == 3  # repeated digits
        da = torch.randint(1, 10, (k,), device=device)
        db = torch.randint(1, 10, (k,), device=device)
        sa[m] = (da[m] * 11_111_111).clamp(LO, HI)
        sb[m] = (db[m] * 11_111_111).clamp(LO, HI)

        m = kind == 4  # near extrema
        choose = torch.randint(0, 2, (k,), device=device)
        edge = torch.where(choose == 0, LO + torch.randint(0, 10000, (k,), device=device), HI - torch.randint(0, 10000, (k,), device=device))
        sa[m] = edge[m]

        m = kind == 5  # sparse round numbers
        run = torch.randint(1, 8, (k,), device=device)
        scale = (10 ** run).long()
        sa[m] = ((x[m] // scale[m]) * scale[m]).clamp(LO, HI)
        sb[m] = ((sb[m] // scale[m]) * scale[m]).clamp(LO, HI)

        m = kind == 6  # asymmetric runs of 0 and 9
        run = torch.randint(1, 8, (k,), device=device)
        scale = (10 ** run).long()
        sa[m] = ((x[m] // scale[m]) * scale[m]).clamp(LO, HI)
        sb[m] = ((sb[m] // scale[m]) * scale[m] + scale[m] - 1).clamp(LO, HI)

        m = kind == 7  # identical / nearly identical operands
        sb[m] = (sa[m] + torch.randint(-10, 11, (k,), device=device)[m]).clamp(LO, HI)
        a[:k], b[:k] = sa, sb

    ad, bd = digits(a), digits(b)
    tokens = torch.full((n, SEQ), 10, dtype=torch.long, device=device)
    tokens[:, 0:16:2] = ad
    tokens[:, 1:16:2] = bd
    target = digits(a + b, 9)
    return tokens, target, a, b


@torch.no_grad()
def evaluate(model, n, batch=16384, structured=0.0):
    model.eval()
    exact = total = digit_ok = 0
    worst_margin = float('inf')
    for _ in range(math.ceil(n / batch)):
        q = min(batch, n - total)
        t, y, _, _ = make_batch(q, next(model.parameters()).device, structured)
        logits = model(t)
        pred = logits.argmax(-1)
        exact += (pred == y).all(1).sum().item()
        digit_ok += (pred == y).sum().item()
        correct = logits.gather(2, y.unsqueeze(-1)).squeeze(-1)
        other = logits.masked_fill(F.one_hot(y, 10).bool(), -torch.inf).amax(-1)
        worst_margin = min(worst_margin, (correct - other).amin().item())
        total += q
    model.train()
    return exact / total, digit_ok / (total * 9), worst_margin


def curated(device):
    vals = {LO, LO+1, LO+9, LO+10, LO+99, 11_111_111, 20_000_000, 50_000_000,
            88_888_888, 89_999_999, 90_000_000, 90_000_001, 98_999_999,
            99_000_000, 99_900_000, 99_990_000, 99_999_000, 99_999_900,
            99_999_990, 99_999_998, HI}
    for p in range(1, 8):
        s = 10 ** p
        for lead in range(1, 10):
            base = lead * 10_000_000
            for z in (-1, 0, 1):
                v = base + s + z
                if LO <= v <= HI: vals.add(v)
    pairs = [(a,b) for a in vals for b in vals]
    # Add all complement neighborhoods and long-carry asymmetries.
    for a in sorted(vals):
        for d in range(-10, 11):
            b = 100_000_000 - a + d
            if LO <= b <= HI: pairs.append((a,b))
    a = torch.tensor([x for x,_ in pairs], device=device)
    b = torch.tensor([y for _,y in pairs], device=device)
    t = torch.full((len(pairs), SEQ), 10, dtype=torch.long, device=device)
    t[:,0:16:2], t[:,1:16:2] = digits(a), digits(b)
    return t, digits(a+b, 9), pairs


@torch.no_grad()
def eval_curated(model):
    model.eval()
    t, y, pairs = curated(next(model.parameters()).device)
    logits = model(t)
    pred = logits.argmax(-1)
    good = (pred == y).all(1)
    failures = [(pairs[i], pred[i].tolist(), y[i].tolist()) for i in (~good).nonzero().flatten().tolist()[:10]]
    model.train()
    return good.float().mean().item(), failures


def tensor_literal(t):
    # 8 significant digits is ample for faithful float32 inference and keeps source manageable.
    return repr(t.detach().cpu().tolist())


def export(model, path):
    state = {k: v.detach().cpu() for k,v in model.state_dict().items()}
    source = '''import torch\nfrom torch import nn\nimport torch.nn.functional as F\n\nD=32\nFF=32\n\nclass Block(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.n1=nn.LayerNorm(D)\n        self.attn=nn.MultiheadAttention(D,4,batch_first=True)\n        self.n2=nn.LayerNorm(D)\n        self.ff1=nn.Linear(D,FF)\n        self.ff2=nn.Linear(FF,D)\n    def forward(self,x):\n        y=self.n1(x)\n        x=x+self.attn(y,y,y,need_weights=False)[0]\n        return x+self.ff2(F.gelu(self.ff1(self.n2(x))))\n\nclass AdditionTransformer(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.token=nn.Embedding(11,D)\n        self.position=nn.Parameter(torch.empty(25,D))\n        self.blocks=nn.ModuleList([Block(),Block()])\n        self.norm=nn.LayerNorm(D)\n        self.head=nn.Linear(D,10,bias=False)\n    def forward(self,tokens):\n        x=self.token(tokens)+self.position\n        for block in self.blocks:x=block(x)\n        return self.head(self.norm(x[:,16:]))\n\n_STATE = {\n'''
    for k,v in state.items():
        # repr(list) emits ordinary source-level floating point literals.
        source += repr(k) + ':torch.tensor(' + tensor_literal(v) + '),\n'
    source += '''}\n\ndef build_model():\n    model=AdditionTransformer()\n    model.load_state_dict(_STATE)\n    model.eval()\n    return model,{"architecture":"joint-output bidirectional transformer","parameters":sum(p.numel() for p in model.parameters()),"digit_order":"least-significant-first"}\n\ndef add(model,a:int,b:int)->int:\n    sa=f"{a:08d}"[::-1]\n    sb=f"{b:08d}"[::-1]\n    values=[]\n    for x,y in zip(sa,sb):\n        values.extend((ord(x)-48,ord(y)-48))\n    tokens=torch.tensor([values+[10]*9],dtype=torch.long,device=next(model.parameters()).device)\n    with torch.no_grad():\n        out=model(tokens).argmax(-1)[0].tolist()\n    return int("".join(str(d) for d in reversed(out)))\n'''
    Path(path).write_text(source)
    print('exported', path, Path(path).stat().st_size, 'bytes')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--steps',type=int,default=18000)
    ap.add_argument('--batch',type=int,default=8192)
    ap.add_argument('--resume',type=str)
    ap.add_argument('--init-only',action='store_true')
    args=ap.parse_args()
    torch.manual_seed(1401); random.seed(1401)
    torch.backends.cuda.matmul.allow_tf32=True
    device=torch.device('cuda')
    model=AdditionTransformer().to(device)
    print('parameters',sum(p.numel() for p in model.parameters()))
    if args.resume: model.load_state_dict(torch.load(args.resume,weights_only=True))
    export(model,'/workspace/submission.py')
    if args.init_only:return
    opt=torch.optim.AdamW(model.parameters(),lr=2e-3,betas=(0.9,0.98),weight_decay=0.01)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.steps,eta_min=1e-4)
    best=0.0
    for step in range(1,args.steps+1):
        frac=0.20 if step < int(args.steps*.72) else 0.45
        t,y,_,_=make_batch(args.batch,device,frac)
        opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(t).flatten(0,1),y.flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step(); sched.step()
        if step%500==0 or step==1:
            acc,dacc,margin=evaluate(model,50000,structured=0.3)
            edge,fail=eval_curated(model)
            print(f'{step:5d} loss={loss.item():.5f} seq={acc:.6f} digit={dacc:.6f} edge={edge:.6f} margin={margin:.3f} lr={sched.get_last_lr()[0]:.2g}',flush=True)
            score=min(acc,edge)
            if score>best:
                best=score; torch.save(model.state_dict(),'/workspace/best.pt'); export(model,'/workspace/submission.py')
    # Robust low-rate edge-biased stabilization.
    for g in opt.param_groups:g['lr']=2e-5;g['weight_decay']=0.0
    for step in range(1,4001):
        t,y,_,_=make_batch(args.batch,device,0.60)
        opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(t).flatten(0,1),y.flatten())
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step%500==0:
            acc,dacc,margin=evaluate(model,100000,structured=0.4)
            edge,fail=eval_curated(model)
            print(f'fine {step:4d} loss={loss.item():.5f} seq={acc:.6f} edge={edge:.6f} margin={margin:.3f}',flush=True)
            if min(acc,edge)>best:
                best=min(acc,edge);torch.save(model.state_dict(),'/workspace/best.pt');export(model,'/workspace/submission.py')
    model.load_state_dict(torch.load('/workspace/best.pt',weights_only=True))
    export(model,'/workspace/submission.py')
    torch.save(model.state_dict(),'/workspace/final.pt')
    print('final random',evaluate(model,500000,structured=0.0),'structured',evaluate(model,500000,structured=0.7),'curated',eval_curated(model))

if __name__=='__main__':main()
