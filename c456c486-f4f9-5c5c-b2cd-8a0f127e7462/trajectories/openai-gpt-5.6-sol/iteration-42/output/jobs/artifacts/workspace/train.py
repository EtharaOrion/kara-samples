import math
import pathlib
import random
import torch
from torch import nn
import torch.nn.functional as F

D, H, P, LIMIT = 9, 3, 2, 100_000_000_000_000
DEVICE = "cuda"

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D)
        self.proj = nn.Linear(D, D)

    def forward(self, x):
        z = self.norm(x)
        q, k, v = self.qkv(z).chunk(3, -1)
        n = x.shape[1]
        q = q.view(-1, n, H, D // H).transpose(1, 2)
        k = k.view(-1, n, H, D // H).transpose(1, 2)
        v = v.view(-1, n, H, D // H).transpose(1, 2)
        z = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        z = z.transpose(1, 2).reshape(-1, n, D)
        return x + self.proj(z)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a_emb = nn.Embedding(11, D)
        self.b_emb = nn.Embedding(11, D)
        self.y_emb = nn.Embedding(10, D)
        self.pos = nn.Parameter(torch.randn(29, P) * .02)
        self.attn1 = Attention()
        self.ffnorm = nn.LayerNorm(D)
        self.ff1 = nn.Linear(D, 1)
        self.ff2 = nn.Linear(1, D)
        self.attn2 = Attention()
        self.final_norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, 10)

    def forward(self, ad, bd, prefix):
        batch = ad.shape[0]
        src = self.a_emb(ad) + self.b_emb(bd)
        x = torch.cat((src, self.y_emb(prefix)), 1)
        x = x + F.pad(self.pos[:x.shape[1]], (0, D-P)).unsqueeze(0)
        x = self.attn1(x)
        x = x + self.ff2(F.gelu(self.ff1(self.ffnorm(x))))
        x = self.attn2(x)
        return self.head(self.final_norm(x))

def digits(x, n=15):
    p = torch.tensor([10**i for i in range(n)], device=x.device, dtype=torch.long)
    return (x[:, None] // p % 10).long()

def make_uniform(bs):
    a = torch.randint(LIMIT, (bs,), device=DEVICE)
    b = torch.randint(LIMIT, (bs,), device=DEVICE)
    return a, b

def make_structured(bs):
    # Regenerated full-pair families; all arithmetic is training-data generation only.
    a, b = make_uniform(bs)
    kind = torch.randint(7, (bs,), device=DEVICE)
    power = torch.randint(1, 14, (bs,), device=DEVICE)
    p10 = torch.tensor([10**i for i in range(15)], device=DEVICE)[power]
    low = torch.randint(1, 10, (bs,), device=DEVICE)

    # Shifted all-9 carry chains: random background above, low trigger at random start.
    start = torch.randint(0, 13, (bs,), device=DEVICE)
    maxlen = 14 - start
    length = (torch.rand(bs, device=DEVICE) * maxlen).long() + 1
    ps = torch.tensor([10**i for i in range(15)], device=DEVICE)
    s10, e10 = ps[start], ps[start + length]
    run9 = e10 - s10
    upper_mod = LIMIT - e10
    upper = torch.where(upper_mod > 0, (a % torch.clamp(upper_mod, min=1) // e10) * e10, torch.zeros_like(a))
    ca = upper + run9
    cb = s10
    m = kind == 0; a = torch.where(m, ca, a); b = torch.where(m, cb, b)

    # Matched near-carry runs (8... plus one), sparse boundaries, complements.
    run8 = (e10 - s10) * 8 // 9
    m = kind == 1; a = torch.where(m, run8, a); b = torch.where(m, s10, b)
    m = kind == 2; a = torch.where(m, low * p10, a); b = torch.where(m, low * p10, b)
    m = kind == 3; a = torch.where(m, 5 * p10, a); b = torch.where(m, 5 * p10, b)
    m = kind == 4; a = torch.where(m, 9 * p10, a); b = torch.where(m, 9 * p10, b)
    comp = p10 - torch.clamp(a % p10, min=1)
    m = kind == 5; b = torch.where(m, comp, b)
    # Repeated digits, including operand swaps through random assignment.
    rep = torch.randint(10, (bs,), device=DEVICE) * 11_111_111_111_111
    m = kind == 6; a = torch.where(m, rep, a)
    swap = torch.rand(bs, device=DEVICE) < .5
    return torch.where(swap, b, a), torch.where(swap, a, b)

def batch(bs, structured_fraction):
    a, b = make_uniform(bs)
    if structured_fraction:
        sa, sb = make_structured(bs)
        mask = torch.rand(bs, device=DEVICE) < structured_fraction
        a, b = torch.where(mask, sa, a), torch.where(mask, sb, b)
    ad, bd = digits(a), digits(b)
    ad[:, 14] = 10
    bd[:, 14] = 10
    y = digits(a + b)
    return ad, bd, y

@torch.no_grad()
def exact(model, total, structured=False, bs=16384):
    model.eval(); errors = 0
    for _ in range((total + bs - 1)//bs):
        n = min(bs, total)
        a, b = make_structured(n) if structured else make_uniform(n)
        ad, bd = digits(a), digits(b); ad[:,14] = 10; bd[:,14] = 10
        target = digits(a+b)
        out = torch.empty((n, 0), dtype=torch.long, device=DEVICE)
        for _ in range(15):
            logits = model(ad, bd, out)
            out = torch.cat((out, logits[:, 14 + out.shape[1]].argmax(-1, keepdim=True)), 1)
        errors += (out != target).any(1).sum().item(); total -= n
    model.train(); return errors

def export(model):
    state = {k: v.detach().cpu().tolist() for k,v in model.state_dict().items()}
    template = pathlib.Path('/workspace/submission_template.py').read_text()
    pathlib.Path('/workspace/submission.py').write_text(template.replace('__STATE__', repr(state)))


def main():
    torch.manual_seed(42); random.seed(42)
    model = Model().to(DEVICE)
    print('parameters', sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(.9,.98), weight_decay=.005)
    phases = [
        (18000, 3e-3, 0.0),
        (18000, 1e-3, .35),
        (18000, 3e-4, .50),
        (18000, 1e-4, .50),
        (24000, 3e-5, .50),
        (24000, 1e-5, .40),
    ]
    step = 0
    for steps, lr, frac in phases:
        for g in opt.param_groups: g['lr'] = lr
        for _ in range(steps):
            ad, bd, y = batch(4096, frac)
            # Teacher-forced prior digits; positions 14..28 predict y0..y14.
            logits = model(ad, bd, y[:, :-1])[:, 14:]
            loss = F.cross_entropy(logits.reshape(-1, 10), y.reshape(-1))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            step += 1
            if step % 3000 == 0:
                print(step, float(loss), 'random_err', exact(model, 16384), 'struct_err', exact(model, 16384, True), flush=True)
                torch.save(model.state_dict(), '/workspace/latest.pt')
                export(model)
    print('final random', exact(model, 1048576), flush=True)
    print('final structured', exact(model, 1048576, True), flush=True)
    torch.save(model.state_dict(), '/workspace/model.pt'); export(model)

if __name__ == '__main__': main()
