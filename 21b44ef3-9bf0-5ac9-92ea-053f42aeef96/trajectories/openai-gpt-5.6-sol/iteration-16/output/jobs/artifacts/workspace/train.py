import math
import random
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

D = 20
RANK = 9
FF = 4
SEQ = 25
BATCH = 4096
DEVICE = "cuda"


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(D, 4, batch_first=True)
        self.n2 = nn.LayerNorm(D)
        self.ff1 = nn.Linear(D, FF)
        self.ff2 = nn.Linear(FF, D)

    def forward(self, x, mask):
        z = self.n1(x)
        x = x + self.attn(z, z, z, attn_mask=mask, need_weights=False)[0]
        x = x + self.ff2(F.gelu(self.ff1(self.n2(x))))
        return x


class Adder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(11, D)
        self.pos_a = nn.Parameter(torch.empty(SEQ, RANK))
        self.pos_b = nn.Parameter(torch.empty(RANK, D))
        self.blocks = nn.ModuleList([Block(), Block()])
        self.norm = nn.LayerNorm(D)
        self.out = nn.Linear(D, 10, bias=False)
        self.register_buffer("mask", torch.triu(torch.ones(SEQ, SEQ, dtype=torch.bool), 1), persistent=False)
        nn.init.normal_(self.pos_a, std=.2)
        nn.init.normal_(self.pos_b, std=.2)

    def forward(self, tokens):
        n = tokens.shape[1]
        x = self.tok(tokens) + (self.pos_a[:n] @ self.pos_b)
        mask = self.mask[:n, :n]
        for block in self.blocks:
            x = block(x, mask)
        return self.out(self.norm(x))


def digits(x, n=8):
    powers = 10 ** torch.arange(n, device=x.device)
    return (x[:, None] // powers % 10).long()


def uniform(n):
    a = torch.randint(10_000_000, 100_000_000, (n,), device=DEVICE)
    b = torch.randint(10_000_000, 100_000_000, (n,), device=DEVICE)
    return a, b


def structured(n):
    # Diverse arithmetic families, randomized independently each batch.
    kind = torch.randint(0, 8, (n,), device=DEVICE)
    a, b = uniform(n)
    pows = (10 ** torch.randint(1, 8, (n,), device=DEVICE)).long()

    m = kind == 0  # exact and near complements to 100M
    aa = torch.randint(10_000_000, 90_000_001, (n,), device=DEVICE)
    delta = torch.randint(-3, 4, (n,), device=DEVICE)
    bb = (100_000_000 - aa + delta).clamp(10_000_000, 99_999_999)
    a[m], b[m] = aa[m], bb[m]

    m = kind == 1  # long runs of trailing nines against a nonzero suffix
    prefix = torch.randint(1, 10, (n,), device=DEVICE)
    aa = (torch.randint(10_000_000, 100_000_000, (n,), device=DEVICE) // pows) * pows + pows - 1
    aa = aa.clamp(10_000_000, 99_999_999)
    bb = (torch.randint(10_000_000, 100_000_000, (n,), device=DEVICE) // pows) * pows + prefix
    bb = bb.clamp(10_000_000, 99_999_999)
    a[m], b[m] = aa[m], bb[m]

    m = kind == 2  # round boundaries and nearby values
    aa = (a // pows) * pows + torch.randint(-2, 3, (n,), device=DEVICE)
    bb = (b // pows) * pows + torch.randint(-2, 3, (n,), device=DEVICE)
    aa.clamp_(10_000_000, 99_999_999); bb.clamp_(10_000_000, 99_999_999)
    a[m], b[m] = aa[m], bb[m]

    m = kind == 3  # repeated digits
    da = torch.randint(1, 10, (n,), device=DEVICE)
    db = torch.randint(1, 10, (n,), device=DEVICE)
    aa = da * 11_111_111; bb = db * 11_111_111
    a[m], b[m] = aa[m], bb[m]

    m = kind == 4  # sparse values
    place1 = 10 ** torch.randint(0, 7, (n,), device=DEVICE)
    place2 = 10 ** torch.randint(0, 7, (n,), device=DEVICE)
    aa = 10_000_000 + torch.randint(0, 9, (n,), device=DEVICE) * place1
    bb = 10_000_000 + torch.randint(0, 9, (n,), device=DEVICE) * place2
    a[m], b[m] = aa[m], bb[m]

    m = kind == 5  # near extrema
    side = torch.randint(0, 2, (n,), device=DEVICE)
    aa = torch.where(side == 0, 10_000_000 + torch.randint(0, 1000, (n,), device=DEVICE),
                     99_999_999 - torch.randint(0, 1000, (n,), device=DEVICE))
    side = torch.randint(0, 2, (n,), device=DEVICE)
    bb = torch.where(side == 0, 10_000_000 + torch.randint(0, 1000, (n,), device=DEVICE),
                     99_999_999 - torch.randint(0, 1000, (n,), device=DEVICE))
    a[m], b[m] = aa[m], bb[m]

    m = kind == 6  # force chosen carry start, with random upper digits
    k = torch.randint(0, 8, (n,), device=DEVICE)
    pw = 10 ** k
    low = torch.maximum(pw, torch.ones_like(pw))
    aa = (a // (pw * 10)) * (pw * 10) + 9 * pw + (a % pw)
    bb = (b // (pw * 10)) * (pw * 10) + torch.randint(1, 10, (n,), device=DEVICE) * pw + (b % pw)
    aa.clamp_(10_000_000, 99_999_999); bb.clamp_(10_000_000, 99_999_999)
    a[m], b[m] = aa[m], bb[m]
    return a, b


def batch(step, structured_rate=.35):
    a, b = uniform(BATCH)
    take = int(BATCH * structured_rate)
    if take:
        sa, sb = structured(take)
        a[:take], b[:take] = sa, sb
    ad, bd = digits(a), digits(b)
    target = digits(a + b, 9)
    tokens = torch.full((BATCH, SEQ), 10, dtype=torch.long, device=DEVICE)
    tokens[:, 0:16:2] = ad
    tokens[:, 1:16:2] = bd
    tokens[:, 17:] = target[:, :-1]
    return tokens, target


@torch.inference_mode()
def accuracy(model, n, structured_data=False, chunk=4096):
    model.eval()
    good = total = 0
    min_margin = 100.0
    while total < n:
        bs = min(chunk, n-total)
        a, b = structured(bs) if structured_data else uniform(bs)
        ad, bd = digits(a), digits(b)
        target = digits(a+b, 9)
        tok = torch.full((bs, SEQ), 10, dtype=torch.long, device=DEVICE)
        tok[:, 0:16:2] = ad; tok[:, 1:16:2] = bd
        pred = []
        for j in range(9):
            logits = model(tok[:, :17+j])[:, -1]
            top = logits.topk(2, dim=-1)
            d = top.indices[:, 0]
            pred.append(d)
            min_margin = min(min_margin, (top.values[:,0]-top.values[:,1]).min().item())
            if j < 8: tok[:, 17+j] = d
        pred = torch.stack(pred, 1)
        good += (pred == target).all(1).sum().item(); total += bs
    model.train()
    return good / total, min_margin


def export(model):
    state = {k: v.detach().float().cpu() for k,v in model.state_dict().items()}
    # mask is non-persistent and absent.
    def literal(t):
        return "torch.tensor(" + repr(t.tolist()) + ",dtype=torch.float32)"
    entries = ",\n            ".join(repr(k)+": "+literal(v) for k,v in state.items())
    source = '''import torch
from torch import nn
import torch.nn.functional as F

D=20
RANK=9
SEQ=25

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1=nn.LayerNorm(D)
        self.attn=nn.MultiheadAttention(D,4,batch_first=True)
        self.n2=nn.LayerNorm(D)
        self.ff1=nn.Linear(D,4)
        self.ff2=nn.Linear(4,D)
    def forward(self,x,mask):
        z=self.n1(x)
        x=x+self.attn(z,z,z,attn_mask=mask,need_weights=False)[0]
        return x+self.ff2(F.gelu(self.ff1(self.n2(x))))

class AdditionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok=nn.Embedding(11,D)
        self.pos_a=nn.Parameter(torch.empty(SEQ,RANK))
        self.pos_b=nn.Parameter(torch.empty(RANK,D))
        self.blocks=nn.ModuleList([Block(),Block()])
        self.norm=nn.LayerNorm(D)
        self.out=nn.Linear(D,10,bias=False)
        self.register_buffer("mask",torch.triu(torch.ones(SEQ,SEQ,dtype=torch.bool),1),persistent=False)
        self.load_state_dict({
            STATE
        })
    def forward(self,tokens):
        n=tokens.shape[1]
        x=self.tok(tokens)+(self.pos_a[:n]@self.pos_b)
        mask=self.mask[:n,:n]
        for block in self.blocks:
            x=block(x,mask)
        return self.out(self.norm(x))

def build_model():
    model=AdditionTransformer()
    model.eval()
    return model,{"architecture":"two-layer causal autoregressive transformer","digit_order":"least-significant-first","position_rank":9}

def add(model,a:int,b:int)->int:
    sa=f"{a:08d}"[::-1]
    sb=f"{b:08d}"[::-1]
    values=[]
    for x,y in zip(sa,sb):
        values.extend((ord(x)-48,ord(y)-48))
    values.append(10)
    device=next(model.parameters()).device
    tokens=torch.tensor([values],dtype=torch.long,device=device)
    output=[]
    with torch.inference_mode():
        for _ in range(9):
            digit=int(model(tokens)[0,-1].argmax().item())
            output.append(chr(48+digit))
            tokens=torch.cat((tokens,torch.tensor([[digit]],dtype=torch.long,device=device)),dim=1)
    return int("".join(reversed(output)))
'''.replace("STATE", entries)
    Path("/workspace/submission.py").write_text(source)


def main():
    torch.manual_seed(1601)
    torch.set_float32_matmul_precision("high")
    model = Adder().to(DEVICE)
    print("parameters", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(.9,.98), weight_decay=.01)
    best = 0.0
    for step in range(1, 36001):
        if step <= 1000: lr = 2e-3 * step / 1000
        elif step <= 28000: lr = 2e-3 * (0.1 + .9 * .5 * (1 + math.cos(math.pi*(step-1000)/27000)))
        elif step <= 32000: lr = 8e-5
        else: lr = 2e-5
        for g in opt.param_groups: g['lr'] = lr
        rate = .30 if step <= 28000 else .55
        tok, target = batch(step, rate)
        logits = model(tok)[:, 16:]
        loss = F.cross_entropy(logits.reshape(-1,10), target.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 1000 == 0:
            print(step, f"loss={loss.item():.6g}", f"lr={lr:.2g}", flush=True)
        if step in (20000,24000,28000,30000,32000,34000,36000):
            ar, mr = accuracy(model, 20000, False)
            ae, me = accuracy(model, 20000, True)
            print("VAL",step,ar,ae,"margin",mr,me,flush=True)
            if min(ar,ae) >= best:
                best=min(ar,ae); export(model)
                torch.save(model.state_dict(), "/workspace/model.pt")
                print("EXPORTED",best,flush=True)
    print("final larger validation",accuracy(model,100000,False),accuracy(model,100000,True),flush=True)

if __name__ == "__main__": main()
