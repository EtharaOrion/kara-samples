import argparse
import random
from pathlib import Path
import torch
from torch import nn

WIDTH = 32
HEADS = 4
FF = 32
LAYERS = 2
LO = 10_000_000
HI = 99_999_999
POW10 = torch.tensor([10 ** i for i in range(9)], device='cuda', dtype=torch.long)


class Block(nn.Module):
    def __init__(self, width, heads, ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, ff), nn.GELU(), nn.Linear(ff, width))
    def forward(self, x):
        y = self.norm1(x)
        x = x + self.attention(y, y, y, need_weights=False)[0]
        return x + self.ff(self.norm2(x))


class Model(nn.Module):
    def __init__(self, width=WIDTH, heads=HEADS, ff=FF, layers=LAYERS):
        super().__init__()
        self.token = nn.Embedding(11, width)
        self.position = nn.Parameter(torch.empty(25, width))
        self.blocks = nn.ModuleList([Block(width, heads, ff) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, 10, bias=False)
        nn.init.normal_(self.position, std=.02)
    def forward(self, tokens):
        x = self.token(tokens) + self.position
        for block in self.blocks: x = block(x)
        return self.output(self.norm(x[:, 16:]))


def random_pairs(n, structured=.30):
    a = torch.randint(LO, HI + 1, (n,), device='cuda')
    b = torch.randint(LO, HI + 1, (n,), device='cuda')
    k = int(n * structured)
    if not k: return a, b
    mode = torch.randint(0, 6, (k,), device='cuda')
    idx = torch.arange(k, device='cuda')
    x = torch.randint(LO, HI + 1, (k,), device='cuda')
    # Complements force carries through many columns.
    mask = mode == 0
    a[idx[mask]] = x[mask]
    b[idx[mask]] = 100_000_000 - x[mask]
    # Long suffixes of 9 plus small perturbations.
    mask = mode == 1
    places = 10 ** torch.randint(1, 8, (k,), device='cuda')
    aa = (x // places) * places + places - 1
    a[idx[mask]] = aa[mask].clamp(LO, HI)
    b[idx[mask]] = torch.randint(LO, HI + 1, (k,), device='cuda')[mask]
    # Decimal boundaries / rounded operands.
    mask = mode == 2
    places = 10 ** torch.randint(1, 8, (k,), device='cuda')
    aa = (x // places) * places
    a[idx[mask]] = aa[mask].clamp(LO, HI)
    # Near extrema.
    mask = mode == 3
    a[idx[mask]] = (LO + torch.randint(0, 10000, (k,), device='cuda'))[mask]
    b[idx[mask]] = (HI - torch.randint(0, 10000, (k,), device='cuda'))[mask]
    # Repeated digits.
    mask = mode == 4
    digit = torch.randint(1, 10, (k,), device='cuda')
    rep = digit * 11_111_111
    a[idx[mask]] = rep[mask]
    # Carry/no-carry column patterns by complementary random values around 1e8.
    mask = mode == 5
    delta = torch.randint(-9999, 10000, (k,), device='cuda')
    a[idx[mask]] = x[mask]
    b[idx[mask]] = (100_000_000 - x + delta)[mask].clamp(LO, HI)
    return a, b


def encode(a, b):
    da = (a[:, None] // POW10[:8]) % 10
    db = (b[:, None] // POW10[:8]) % 10
    inp = torch.empty((a.shape[0], 25), dtype=torch.long, device='cuda')
    inp[:, 0:16:2] = da
    inp[:, 1:16:2] = db
    inp[:, 16:] = 10
    target = ((a + b)[:, None] // POW10) % 10
    return inp, target


@torch.no_grad()
def evaluate(model, n=100000, structured=.0, batch=8192):
    model.eval(); good = total = 0; token_good = 0
    for _ in range((n + batch - 1) // batch):
        m = min(batch, n-total)
        a,b = random_pairs(m, structured)
        x,y = encode(a,b)
        pred = model(x).argmax(-1)
        good += (pred == y).all(1).sum().item()
        token_good += (pred == y).sum().item(); total += m
    model.train()
    return good/total, token_good/(total*9)


def export(model, path):
    state = {k:v.detach().cpu().float() for k,v in model.state_dict().items()}
    arch = f'''import torch\nfrom torch import nn\n\n\nclass Block(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.norm1=nn.LayerNorm({WIDTH})\n        self.attention=nn.MultiheadAttention({WIDTH},{HEADS},batch_first=True)\n        self.norm2=nn.LayerNorm({WIDTH})\n        self.ff=nn.Sequential(nn.Linear({WIDTH},{FF}),nn.GELU(),nn.Linear({FF},{WIDTH}))\n    def forward(self,x):\n        y=self.norm1(x)\n        x=x+self.attention(y,y,y,need_weights=False)[0]\n        return x+self.ff(self.norm2(x))\n\n\nclass AddTransformer(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.token=nn.Embedding(11,{WIDTH})\n        self.position=nn.Parameter(torch.empty(25,{WIDTH}))\n        self.blocks=nn.ModuleList([Block() for _ in range({LAYERS})])\n        self.norm=nn.LayerNorm({WIDTH})\n        self.output=nn.Linear({WIDTH},10,bias=False)\n    def forward(self,tokens):\n        x=self.token(tokens)+self.position\n        for block in self.blocks:x=block(x)\n        return self.output(self.norm(x[:,16:]))\n\n\n_STATE={{\n'''
    for key, value in state.items():
        arch += repr(key) + ':torch.tensor(' + repr(value.tolist()) + '),\n'
    arch += '''}\n\ndef build_model():\n    model=AddTransformer()\n    model.load_state_dict(_STATE)\n    model.eval()\n    return model,{"architecture":"joint-output transformer","digit_order":"least-significant-first"}\n\n\ndef add(model,a:int,b:int)->int:\n    sa,sb=str(a)[::-1],str(b)[::-1]\n    tokens=[int(d) for pair in zip(sa,sb) for d in pair]+[10]*9\n    device=next(model.parameters()).device\n    with torch.no_grad():\n        digits=model(torch.tensor([tokens],dtype=torch.long,device=device)).argmax(-1)[0].tolist()\n    return int(''.join(str(d) for d in digits[::-1]))\n'''
    Path(path).write_text(arch)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--steps',type=int,default=30000); p.add_argument('--batch',type=int,default=4096); p.add_argument('--seed',type=int,default=13); args=p.parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed)
    model=Model().cuda().train()
    print('parameters',sum(x.numel() for x in model.parameters()),flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-3,betas=(.9,.98),weight_decay=.01)
    lossfn=nn.CrossEntropyLoss()
    best=0
    for step in range(1,args.steps+1):
        frac=.15 if step < 10000 else (.30 if step < 22000 else .50)
        a,b=random_pairs(args.batch,frac); x,y=encode(a,b)
        logits=model(x); loss=lossfn(logits.reshape(-1,10),y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step in (18000,24000):
            for g in opt.param_groups:g['lr'] *= .2
        if step%1000==0:
            acc,tok=evaluate(model,20000,.3)
            print(step,float(loss),acc,tok,flush=True)
            torch.save({'model':model.state_dict(),'opt':opt.state_dict(),'step':step},'/workspace/checkpoint.pt')
            if acc>=best:
                best=acc; export(model,'/workspace/submission.py')
        if step>=15000 and best>=.99995:
            break
    acc,tok=evaluate(model,200000,0); edge,etok=evaluate(model,200000,.9)
    print('final',acc,tok,edge,etok,flush=True)
    if min(acc,edge)>=.99: export(model,'/workspace/submission.py')

if __name__=='__main__':main()
