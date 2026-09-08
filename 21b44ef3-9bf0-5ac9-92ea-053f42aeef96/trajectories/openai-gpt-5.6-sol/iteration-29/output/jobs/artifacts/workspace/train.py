import math
import os
import random
import sys
import time

import torch
from torch import nn
import torch.nn.functional as F

os.chdir('/workspace')
torch.set_float32_matmul_precision('high')
DEVICE = 'cuda'
BATCH = 8192


class Model(nn.Module):
    def __init__(self, ff=4):
        super().__init__()
        d = 20
        self.ff_width = ff
        self.token = nn.Embedding(11, d)
        self.pos_left = nn.Parameter(torch.empty(25, 2))
        self.pos_right = nn.Parameter(torch.empty(2, d))
        self.q = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(2)])
        self.o = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(2)])
        self.k = nn.Linear(d, 5, bias=False)
        self.v = nn.Linear(d, 5, bias=False)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, ff, bias=False)
        self.ff2 = nn.Linear(ff, d, bias=False)
        self.final_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 10, bias=False)
        self.apply(self._init)
        nn.init.normal_(self.pos_left, std=.15)
        nn.init.normal_(self.pos_right, std=.15)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.16)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, tokens):
        n = tokens.shape[1]
        x = self.token(tokens) + self.pos_left[:n] @ self.pos_right
        mask = torch.ones(n, n, device=tokens.device, dtype=torch.bool).triu(1)
        for i in range(2):
            z = self.norm1(x)
            b = z.shape[0]
            q = self.q[i](z).view(b, n, 4, 5).transpose(1, 2)
            k, v = self.k(z), self.v(z)
            attn = torch.einsum('bhtd,bsd->bhts', q, k).mul_(5 ** -.5)
            attn.masked_fill_(mask, -torch.inf)
            context = torch.einsum('bhts,bsd->bhtd', attn.softmax(-1), v)
            x = x + self.o[i](context.transpose(1, 2).reshape(b, n, 20))
            x = x + self.ff2(F.gelu(self.ff1(self.norm2(x))))
        return self.head(self.final_norm(x))


def structured(n):
    a = torch.randint(10_000_000, 100_000_000, (n,), device=DEVICE)
    b = torch.randint(10_000_000, 100_000_000, (n,), device=DEVICE)
    kind = torch.randint(0, 8, (n,), device=DEVICE)
    # Exact and near complements exercise full carry propagation.
    m = kind == 1
    aa = torch.randint(10_000_000, 90_000_001, (n,), device=DEVICE)
    delta = torch.randint(-20, 21, (n,), device=DEVICE)
    bb = (100_000_000 - aa + delta).clamp(10_000_000, 99_999_999)
    a = torch.where(m, aa, a); b = torch.where(m, bb, b)
    # Unequal trailing runs of nines create carry starts at every column.
    m = kind == 2
    pows = torch.tensor([10,100,1000,10000,100000,1000000,10000000], device=DEVICE)
    p = pows[torch.randint(0, 7, (n,), device=DEVICE)]
    aa = (a // p) * p + p - 1
    aa = aa.clamp_max(99_999_999)
    bsmall = torch.randint(10_000_000, 100_000_000, (n,), device=DEVICE)
    a = torch.where(m, aa, a); b = torch.where(m, bsmall, b)
    # Values just around decimal boundaries.
    m = kind == 3
    p = pows[torch.randint(1, 7, (n,), device=DEVICE)]
    aa = ((a // p) * p + torch.randint(-12, 13, (n,), device=DEVICE)).clamp(10_000_000,99_999_999)
    bb = ((b // p) * p + torch.randint(-12, 13, (n,), device=DEVICE)).clamp(10_000_000,99_999_999)
    a = torch.where(m, aa, a); b = torch.where(m, bb, b)
    # Repeated digit operands.
    m = kind == 4
    rep = torch.tensor([11_111_111,22_222_222,33_333_333,44_444_444,55_555_555,66_666_666,77_777_777,88_888_888,99_999_999], device=DEVICE)
    a = torch.where(m, rep[torch.randint(0,9,(n,),device=DEVICE)], a)
    b = torch.where(m, rep[torch.randint(0,9,(n,),device=DEVICE)], b)
    # Extreme neighborhoods.
    m = kind == 5
    a = torch.where(m, 10_000_000 + torch.randint(0,10000,(n,),device=DEVICE), a)
    b = torch.where(m, 99_999_999 - torch.randint(0,10000,(n,),device=DEVICE), b)
    # Sparse high prefix plus short suffix.
    m = kind == 6
    lead = torch.randint(1,10,(n,),device=DEVICE) * 10_000_000
    a = torch.where(m, lead + torch.randint(0,1000,(n,),device=DEVICE), a)
    b = torch.where(m, torch.randint(1,10,(n,),device=DEVICE)*10_000_000 + torch.randint(0,1000,(n,),device=DEVICE), b)
    # Asymmetric runs: one near boundary and one arbitrary repeated suffix.
    m = kind == 7
    p = pows[torch.randint(0,7,(n,),device=DEVICE)]
    aa = ((a // p) * p + p - 1).clamp_max(99_999_999)
    bb = ((b // p) * p + torch.randint(0,10,(n,),device=DEVICE)).clamp_max(99_999_999)
    a = torch.where(m, aa, a); b = torch.where(m, bb, b)
    return a, b


def batch(n=BATCH, structured_fraction=.35):
    a = torch.randint(10_000_000, 100_000_000, (n,), device=DEVICE)
    b = torch.randint(10_000_000, 100_000_000, (n,), device=DEVICE)
    count = int(n * structured_fraction)
    if count:
        a[:count], b[:count] = structured(count)
    place8 = torch.tensor([1,10,100,1000,10000,100000,1000000,10000000], device=DEVICE)
    place9 = torch.tensor([1,10,100,1000,10000,100000,1000000,10000000,100000000], device=DEVICE)
    ad = (a[:,None] // place8) % 10
    bd = (b[:,None] // place8) % 10
    target = ((a+b)[:,None] // place9) % 10
    tokens = torch.empty(n,25,dtype=torch.long,device=DEVICE)
    tokens[:,:16:2] = ad; tokens[:,1:16:2] = bd
    tokens[:,16] = 10
    tokens[:,17:] = target[:,:8]
    return tokens, target


@torch.no_grad()
def evaluate(model, n=100000, structured_fraction=.5, chunk=10000):
    model.eval(); correct = total = 0; min_margin = 100.
    for _ in range((n + chunk - 1)//chunk):
        size = min(chunk, n-total)
        tokens, target = batch(size, structured_fraction)
        prefix = tokens[:,:17]
        predictions = []
        for j in range(9):
            logits = model(prefix)[:,-1]
            top = logits.topk(2,dim=-1).values
            min_margin = min(min_margin, float((top[:,0]-top[:,1]).min()))
            digit = logits.argmax(-1)
            predictions.append(digit)
            prefix = torch.cat((prefix,digit[:,None]),1)
        pred = torch.stack(predictions,1)
        correct += int((pred == target).all(1).sum())
        total += size
    model.train()
    return correct, total, min_margin


def train_stage(model, steps, lr, sf, label):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(.9,.98), weight_decay=.01)
    start = time.time()
    model.train()
    for step in range(1, steps+1):
        tokens, target = batch(structured_fraction=sf)
        logits = model(tokens)[:,16:25]
        loss = F.cross_entropy(logits.reshape(-1,10), target.reshape(-1))
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
        optimizer.step()
        if step % 2000 == 0 or step == steps:
            print(label, step, 'loss', round(float(loss),6), 'sec', round(time.time()-start,1), flush=True)
            torch.save(model.state_dict(), f'/workspace/{label}.pt')
    return model


def prune(model, width):
    assert width == model.ff_width - 1
    score = model.ff1.weight.norm(dim=1) * model.ff2.weight.norm(dim=0)
    keep = torch.argsort(score, descending=True)[:width].sort().values
    new = Model(width).to(DEVICE)
    source = model.state_dict(); dest = new.state_dict()
    for name in dest:
        if name == 'ff1.weight': dest[name].copy_(source[name][keep])
        elif name == 'ff2.weight': dest[name].copy_(source[name][:,keep])
        else: dest[name].copy_(source[name])
    return new


def export(model):
    values = [p.detach().float().cpu().flatten().tolist() for p in model.parameters()]
    path = '/workspace/submission.py'
    text = open(path).read()
    begin = text.index('_VALUES = ')
    end = text.index('\n\n\ndef build_model', begin)
    literal = '_VALUES = ' + repr(values)
    output = text[:begin] + literal + text[end:]
    temp = '/workspace/submission.new'
    open(temp,'w').write(output)
    os.replace(temp,path)
    print('exported',sum(p.numel() for p in model.parameters()),'parameters',len(output),'bytes',flush=True)


def main():
    torch.manual_seed(29001); random.seed(29001)
    model = Model(4).to(DEVICE)
    train_stage(model, 36000, 2e-3, .35, 'teacher')
    train_stage(model, 6000, 5e-5, .45, 'teacher_stable')
    print('teacher eval',evaluate(model,200000,.5),flush=True)
    model = prune(model,3)
    train_stage(model, 16000, 2e-5, .45, 'width3')
    print('width3 eval',evaluate(model,200000,.5),flush=True)
    model = prune(model,2)
    train_stage(model, 24000, 1e-5, .45, 'width2a')
    train_stage(model, 16000, 4e-6, .55, 'width2b')
    print('width2 eval',evaluate(model,500000,.5),flush=True)
    torch.save(model.state_dict(),'/workspace/final.pt')
    export(model)


if __name__ == '__main__':
    main()
