import argparse
import importlib.util
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path('/workspace')
DEVICE = 'cuda'
D, H, DH, R = 20, 4, 5, 2


class Model(nn.Module):
    def __init__(self, ff=4):
        super().__init__()
        self.ff = ff
        self.token = nn.Embedding(11, D)
        self.pos_left = nn.Parameter(torch.randn(25, R) * .02)
        self.pos_right = nn.Parameter(torch.randn(R, D) * .02)
        self.norm_attn = nn.LayerNorm(D)
        self.query = nn.ModuleList([nn.Linear(D, D, False) for _ in range(2)])
        self.key = nn.Linear(D, DH, False)
        self.value = nn.Linear(D, DH, False)
        self.proj = nn.ModuleList([nn.Linear(D, D, False) for _ in range(2)])
        self.norm_ff = nn.LayerNorm(D)
        self.ff1 = nn.Linear(D, ff, False)
        self.ff2 = nn.Linear(ff, D, False)
        self.norm_out = nn.LayerNorm(D)
        self.classifier = nn.Linear(D, 10, False)

    def forward(self, t):
        n = t.shape[1]
        x = self.token(t) + self.pos_left[:n] @ self.pos_right
        mask = torch.ones(n, n, dtype=torch.bool, device=t.device).triu(1)
        for i in range(2):
            z = self.norm_attn(x)
            q = self.query[i](z).view(-1, n, H, DH).transpose(1, 2)
            k, v = self.key(z).unsqueeze(1), self.value(z).unsqueeze(1)
            s = (q @ k.transpose(-2, -1)) * (DH ** -.5)
            c = (s.masked_fill(mask, -torch.inf).softmax(-1) @ v).transpose(1, 2).reshape(-1, n, D)
            x = x + self.proj[i](c)
            x = x + self.ff2(F.gelu(self.ff1(self.norm_ff(x))))
        return self.classifier(self.norm_out(x))


def structured(a, b):
    batch = a.numel()
    choose = torch.rand(batch, device=DEVICE)
    # Near complements produce long carry chains at every location.
    m = choose < .20
    delta = torch.randint(-100, 101, (batch,), device=DEVICE)
    b = torch.where(m, 100_000_000 - a + delta, b)
    # Decimal boundaries, independently selected powers and small offsets.
    m = (choose >= .20) & (choose < .34)
    powers = torch.tensor([10,100,1000,10000,100000,1000000,10000000], device=DEVICE)
    p = powers[torch.randint(0, 7, (batch,), device=DEVICE)]
    x = (a // p) * p + torch.randint(-10, 11, (batch,), device=DEVICE)
    a = torch.where(m, x, a)
    # Runs of terminal nines paired with nontrivial operands.
    m = (choose >= .34) & (choose < .44)
    x = (a // p) * p + p - 1
    a = torch.where(m, x, a)
    # Runs of terminal zeroes, often asymmetric with terminal nines.
    m = choose >= .44
    a = torch.where(m, (a // p) * p, a)
    x = (b // p) * p + p - 1
    b = torch.where(m & (torch.rand(batch, device=DEVICE) < .5), x, b)
    return a.clamp(10_000_000, 99_999_999), b.clamp(10_000_000, 99_999_999)


def batch_data(batch, structured_fraction=.45):
    a = torch.randint(10_000_000, 100_000_000, (batch,), device=DEVICE)
    b = torch.randint(10_000_000, 100_000_000, (batch,), device=DEVICE)
    use = torch.rand(batch, device=DEVICE) < structured_fraction
    sa, sb = structured(a.clone(), b.clone())
    a, b = torch.where(use, sa, a), torch.where(use, sb, b)
    total = a + b
    places = (10 ** torch.arange(8, device=DEVICE)).long()
    ad = (a[:, None] // places) % 10
    bd = (b[:, None] // places) % 10
    out_places = (10 ** torch.arange(9, device=DEVICE)).long()
    target = (total[:, None] // out_places) % 10
    inp = torch.empty(batch, 25, dtype=torch.long, device=DEVICE)
    inp[:, :16:2], inp[:, 1:16:2] = ad, bd
    inp[:, 16] = 10
    inp[:, 17:] = target[:, :8]
    return inp, target


@torch.no_grad()
def evaluate(model, count=100000, structured_fraction=0.0, batch=10000):
    model.eval(); good = 0; seen = 0; min_margin = 1e9
    for _ in range((count + batch - 1) // batch):
        n = min(batch, count-seen)
        inp, target = batch_data(n, structured_fraction)
        seq = inp[:, :17]
        pred = []
        for __ in range(9):
            logits = model(seq)[:, -1]
            top = logits.topk(2, -1).values
            min_margin = min(min_margin, float((top[:, 0]-top[:, 1]).min()))
            digit = logits.argmax(-1)
            pred.append(digit)
            if len(pred) < 9: seq = torch.cat([seq, digit[:, None]], 1)
        good += int((torch.stack(pred, 1) == target).all(1).sum())
        seen += n
    model.train()
    return good, seen, min_margin


def prune(model, new_ff):
    result = Model(new_ff).to(DEVICE)
    old = model.state_dict(); new = result.state_dict()
    for k in new:
        if k not in ('ff1.weight','ff2.weight'): new[k].copy_(old[k])
    score = model.ff1.weight.norm(dim=1) * model.ff2.weight.norm(dim=0)
    keep = score.topk(new_ff).indices.sort().values
    new['ff1.weight'].copy_(old['ff1.weight'][keep])
    new['ff2.weight'].copy_(old['ff2.weight'][:, keep])
    result.load_state_dict(new)
    return result


def export(model):
    values = torch.cat([p.detach().cpu().float().reshape(-1) for p in model.parameters()]).tolist()
    path = ROOT/'submission.py'
    text = path.read_text()
    start = text.index('_FLAT = ')
    end = text.index('\n', start)
    literal = '_FLAT = [' + ','.join(f'{x:.9g}' for x in values) + ']'
    text = text[:start] + literal + text[end:]
    text = text.replace('_F = 4', f'_F = {model.ff}')
    path.write_text(text)
    print('exported', len(values), 'parameters')


def train_stage(model, steps, lr, batch, structured_fraction, name):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(.9,.98), weight_decay=.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr*.1)
    for step in range(1, steps+1):
        inp, target = batch_data(batch, structured_fraction)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(inp)[:, 16:25]
            loss = F.cross_entropy(logits.reshape(-1,10), target.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 1000 == 0:
            print(name, step, float(loss), flush=True)
        if step % 6000 == 0:
            torch.save(model.state_dict(), ROOT/f'{name}_{step}.pt')
    torch.save(model.state_dict(), ROOT/f'{name}_final.pt')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--batch',type=int,default=8192); args=parser.parse_args()
    torch.manual_seed(936); torch.set_float32_matmul_precision('high')
    model=Model(4).to(DEVICE)
    train_stage(model, 20000, 2e-3, args.batch, .42, 'w4')
    print('w4 eval', evaluate(model, 100000, 0), evaluate(model,100000,.8)); export(model)
    model=prune(model,3)
    train_stage(model,10000,2e-5,args.batch,.50,'w3')
    print('w3 eval',evaluate(model,200000,0),evaluate(model,200000,.8)); export(model)
    model=prune(model,2)
    train_stage(model,18000,1e-5,args.batch,.52,'w2a')
    train_stage(model,10000,3e-6,args.batch,.58,'w2b')
    print('w2 eval',evaluate(model,500000,0),evaluate(model,500000,.8)); export(model)
    torch.save(model.state_dict(),ROOT/'final_raw.pt')

if __name__ == '__main__': main()
