import argparse
import math
import os
import random
import time
import torch
from torch import nn

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
BASE = 100_000_000_000_000
POW = torch.tensor([10 ** i for i in range(15)], dtype=torch.long)

D = 10
F = 20

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D)
        self.proj = nn.Linear(D, D)
        self.n2 = nn.LayerNorm(D)
        self.fc1 = nn.Linear(D, F)
        self.fc2 = nn.Linear(F, D)
    def forward(self, x):
        z = self.n1(x)
        q, k, v = self.qkv(z).chunk(3, -1)
        q = q.view(x.shape[0], x.shape[1], 2, 5).transpose(1, 2)
        k = k.view(x.shape[0], x.shape[1], 2, 5).transpose(1, 2)
        v = v.view(x.shape[0], x.shape[1], 2, 5).transpose(1, 2)
        z = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(z.transpose(1, 2).reshape_as(x))
        return x + self.fc2(torch.nn.functional.gelu(self.fc1(self.n2(x))))

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a_emb = nn.Embedding(11, D); self.b_emb = nn.Embedding(11, D); self.o_emb = nn.Embedding(11, D)
        self.pos = nn.Parameter(torch.empty(30, D))
        self.blocks = nn.ModuleList([Block(), Block()])
        self.norm = nn.LayerNorm(D); self.head = nn.Linear(D, 10)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.normal_(self.pos, std=.02)
    def forward(self, ad, bd, prev):
        src = self.a_emb(ad) + self.b_emb(bd)
        bos = torch.full((ad.shape[0], 1), 10, dtype=torch.long, device=ad.device)
        x = torch.cat((src, self.o_emb(torch.cat((bos, prev), 1))), 1)
        x = x + self.pos[:x.shape[1]]
        for block in self.blocks: x = block(x)
        return self.head(self.norm(x[:, 14:]))

def digits(x, n=15):
    p = POW[:n].to(x.device)
    return (x[:, None] // p[None, :]) % 10

def carry_examples(n, device):
    # Random complete pairs, then impose a random exact carry chain on half.
    da = torch.randint(0, 10, (n, 14), device=device)
    db = torch.randint(0, 10, (n, 14), device=device)
    start = torch.randint(0, 14, (n,), device=device)
    length = torch.randint(1, 15, (n,), device=device)
    idx = torch.arange(14, device=device)[None, :]
    inside = (idx >= start[:, None]) & (idx < (start + length)[:, None].clamp(max=14))
    first = idx == start[:, None]
    # Start carry with sums 10..18, then propagate through columns summing to 9.
    db = torch.where(inside, 9 - da, db)
    start_b = 10 - da + torch.randint(0, 9, (n, 14), device=device)
    start_b = start_b.clamp(0, 9)
    db = torch.where(first, start_b, db)
    a = (da * POW[:14].to(device)).sum(1)
    b = (db * POW[:14].to(device)).sum(1)
    return a, b

def patterned(n, device):
    kind = torch.randint(0, 5, (n,), device=device)
    a = torch.randint(0, BASE, (n,), device=device)
    b = torch.randint(0, BASE, (n,), device=device)
    # sparse powers/increments
    p = POW[torch.randint(0, 14, (n,))].to(device)
    b = torch.where(kind == 0, p * torch.randint(1, 10, (n,), device=device), b)
    # repeated decimal digits
    repa = torch.randint(0, 10, (n,), device=device) * 11_111_111_111_111
    repb = torch.randint(0, 10, (n,), device=device) * 11_111_111_111_111
    a = torch.where(kind == 1, repa, a); b = torch.where(kind == 1, repb, b)
    # complements and close non-carry contrasts
    low = torch.randint(1, BASE, (n,), device=device)
    a = torch.where(kind == 2, low, a); b = torch.where(kind == 2, BASE - low, b)
    low2 = torch.randint(1, BASE, (n,), device=device)
    a = torch.where(kind == 3, low2, a); b = torch.where(kind == 3, BASE - 1 - low2, b)
    # small operands/boundaries
    a = torch.where(kind == 4, torch.randint(0, 1_000_000, (n,), device=device), a)
    b = torch.where(kind == 4, torch.randint(0, 1_000_000, (n,), device=device), b)
    return a, b

def batch(n, device, structured=.25):
    a = torch.randint(0, BASE, (n,), device=device)
    b = torch.randint(0, BASE, (n,), device=device)
    m = int(n * structured)
    if m:
        c = m // 2
        a[:c], b[:c] = carry_examples(c, device)
        a[c:m], b[c:m] = patterned(m-c, device)
        order = torch.randperm(n, device=device)
        a, b = a[order], b[order]
    y = digits(a + b, 15)
    return digits(a, 14), digits(b, 14), y

@torch.no_grad()
def evaluate(model, n, mode="random", bs=8192):
    model.eval(); errors = 0; digit_errors = 0
    for off in range(0, n, bs):
        k = min(bs, n-off)
        if mode == "random":
            a = torch.randint(0, BASE, (k,), device="cuda"); b = torch.randint(0, BASE, (k,), device="cuda")
        elif mode == "carry": a, b = carry_examples(k, "cuda")
        else: a, b = patterned(k, "cuda")
        y = digits(a+b, 15); logits = model(digits(a,14), digits(b,14), y[:,:-1])
        pred = logits.argmax(-1); bad = pred.ne(y)
        errors += bad.any(1).sum().item(); digit_errors += bad.sum().item()
    model.train(); return errors, digit_errors

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--steps",type=int,default=22000); ap.add_argument("--batch",type=int,default=4096); ap.add_argument("--resume"); args=ap.parse_args()
    torch.manual_seed(24001); random.seed(24001); torch.set_float32_matmul_precision("high")
    model=Model().cuda()
    if args.resume: model.load_state_dict(torch.load(args.resume, weights_only=True))
    print("parameters", sum(p.numel() for p in model.parameters()), flush=True)
    opt=torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=.005, fused=True)
    scaler=None; start=time.time(); best=10**9
    for step in range(1,args.steps+1):
        if step <= 7000: lr=3e-3; frac=.20
        elif step <= 13000: lr=1e-3; frac=.25
        elif step <= 18000: lr=3e-4; frac=.35
        else: lr=1e-4; frac=.40
        for g in opt.param_groups:g['lr']=lr
        ad,bd,y=batch(args.batch,"cuda",frac)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            logits=model(ad,bd,y[:,:-1]); loss=torch.nn.functional.cross_entropy(logits.reshape(-1,10),y.reshape(-1))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step%500==0 or step==1:
            print(step, f"loss={loss.item():.6g}", f"lr={lr:g}", f"sec={time.time()-start:.1f}", flush=True)
        if step%2000==0 and step>=6000:
            er=evaluate(model,65536,"random"); ec=evaluate(model,32768,"carry"); ep=evaluate(model,32768,"pattern")
            score=er[0]+ec[0]+ep[0]
            print("VALID",step,"random",er,"carry",ec,"pattern",ep,flush=True)
            torch.save(model.state_dict(),"/workspace/latest.pt")
            if score <= best:
                best=score; torch.save(model.state_dict(),"/workspace/best.pt")
    torch.save(model.state_dict(),"/workspace/final.pt")

if __name__=="__main__": main()
