import math
import os
import random
import torch
from torch import nn
import torch.nn.functional as F

DEVICE = 'cuda'
BATCH = 8192
LO = 10_000_000
HI = 99_999_999


class Model(nn.Module):
    def __init__(self, ff=4):
        super().__init__()
        d = 20
        self.ff_width = ff
        self.token = nn.Embedding(11, d)
        self.pos_a = nn.Parameter(torch.empty(25, 2))
        self.pos_b = nn.Parameter(torch.empty(2, d))
        self.attn_norm = nn.LayerNorm(d)
        self.key = nn.Linear(d, 5, bias=False)
        self.value = nn.Linear(d, 5, bias=False)
        self.query = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(2)])
        self.project = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(2)])
        self.ff_norm = nn.LayerNorm(d)
        self.ff_in = nn.Linear(d, ff, bias=False)
        self.ff_out = nn.Linear(ff, d, bias=False)
        self.final_norm = nn.LayerNorm(d)
        self.output = nn.Linear(d, 10, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.token.weight, std=.2)
        nn.init.normal_(self.pos_a, std=.2)
        nn.init.normal_(self.pos_b, std=.2)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.ones_(self.attn_norm.weight)
        nn.init.zeros_(self.attn_norm.bias)
        nn.init.ones_(self.ff_norm.weight)
        nn.init.zeros_(self.ff_norm.bias)
        nn.init.ones_(self.final_norm.weight)
        nn.init.zeros_(self.final_norm.bias)

    def forward(self, tokens):
        n, length = tokens.shape
        x = self.token(tokens) + (self.pos_a @ self.pos_b)[:length]
        mask = torch.ones(length, length, device=x.device, dtype=torch.bool).triu(1)
        for layer in range(2):
            z = self.attn_norm(x)
            q = self.query[layer](z).view(n, length, 4, 5).transpose(1, 2)
            k = self.key(z).unsqueeze(1)
            v = self.value(z).unsqueeze(1)
            score = (q @ k.transpose(-2, -1)) * (5 ** -.5)
            attn = score.masked_fill(mask, float('-inf')).softmax(-1)
            context = (attn @ v).transpose(1, 2).reshape(n, length, 20)
            x = x + self.project[layer](context)
            x = x + self.ff_out(F.gelu(self.ff_in(self.ff_norm(x))))
        return self.output(self.final_norm(x))


POW10 = torch.tensor([10 ** i for i in range(9)], device=DEVICE, dtype=torch.long)
POW8 = POW10[:8]


def structured_pairs(n):
    # Diverse exact arithmetic families, with random leading digits and carry start/stop positions.
    kind = torch.randint(0, 8, (n,), device=DEVICE)
    a = torch.randint(LO, HI + 1, (n,), device=DEVICE)
    b = torch.randint(LO, HI + 1, (n,), device=DEVICE)

    # Exact/near complements around 10^8 and random decimal boundaries.
    ix = kind == 0
    if ix.any():
        aa = torch.randint(LO, HI + 1, (int(ix.sum()),), device=DEVICE)
        delta = torch.randint(-12, 13, aa.shape, device=DEVICE)
        bb = 100_000_000 - aa + delta
        good = (bb >= LO) & (bb <= HI)
        bb = torch.where(good, bb, torch.randint(LO, HI + 1, bb.shape, device=DEVICE))
        a[ix], b[ix] = aa, bb

    # Force unequal 0/9 suffixes, inducing every length of carry run.
    for knd, fill_a, fill_b in [(1, 9, 0), (2, 0, 9), (3, 9, 9)]:
        ix = kind == knd
        m = int(ix.sum())
        if m:
            run = torch.randint(1, 8, (m,), device=DEVICE)
            scale = POW10[run]
            pa = torch.randint(1_000_000, 100_000_000, (m,), device=DEVICE)
            pb = torch.randint(1_000_000, 100_000_000, (m,), device=DEVICE)
            sa = scale - 1 if fill_a == 9 else torch.zeros_like(scale)
            sb = scale - 1 if fill_b == 9 else torch.zeros_like(scale)
            aa = (pa // scale) * scale + sa
            bb = (pb // scale) * scale + sb
            aa = aa.clamp(LO, HI); bb = bb.clamp(LO, HI)
            a[ix], b[ix] = aa, bb

    # Rounded values with independently selected decimal scales and small offsets.
    ix = kind == 4
    m = int(ix.sum())
    if m:
        ka = torch.randint(1, 8, (m,), device=DEVICE); kb = torch.randint(1, 8, (m,), device=DEVICE)
        sa, sb = POW10[ka], POW10[kb]
        aa = (torch.randint(LO, HI + 1, (m,), device=DEVICE) // sa) * sa
        bb = (torch.randint(LO, HI + 1, (m,), device=DEVICE) // sb) * sb
        aa += torch.randint(0, 10, (m,), device=DEVICE); bb += torch.randint(0, 10, (m,), device=DEVICE)
        a[ix], b[ix] = aa.clamp(LO, HI), bb.clamp(LO, HI)

    # Repeated digits and sparse decimal forms.
    ix = kind == 5
    m = int(ix.sum())
    if m:
        da = torch.randint(1, 10, (m,), device=DEVICE); db = torch.randint(1, 10, (m,), device=DEVICE)
        a[ix], b[ix] = da * 11_111_111, db * 11_111_111
    ix = kind == 6
    m = int(ix.sum())
    if m:
        lead_a = torch.randint(1, 10, (m,), device=DEVICE)
        lead_b = torch.randint(1, 10, (m,), device=DEVICE)
        p1 = torch.randint(0, 7, (m,), device=DEVICE)
        p2 = torch.randint(0, 7, (m,), device=DEVICE)
        aa = lead_a * 10_000_000 + torch.randint(0, 10, (m,), device=DEVICE) * POW10[p1]
        bb = lead_b * 10_000_000 + torch.randint(0, 10, (m,), device=DEVICE) * POW10[p2]
        a[ix], b[ix] = aa.clamp(LO, HI), bb.clamp(LO, HI)

    # Near extrema and broad asymmetric long-nine patterns.
    ix = kind == 7
    m = int(ix.sum())
    if m:
        side = torch.randint(0, 2, (m,), device=DEVICE).bool()
        aa = torch.where(side, LO + torch.randint(0, 10000, (m,), device=DEVICE), HI - torch.randint(0, 10000, (m,), device=DEVICE))
        bb = torch.randint(LO, HI + 1, (m,), device=DEVICE)
        a[ix], b[ix] = aa, bb
    return a, b


def sample(batch=BATCH, structured=.18):
    a = torch.randint(LO, HI + 1, (batch,), device=DEVICE)
    b = torch.randint(LO, HI + 1, (batch,), device=DEVICE)
    n = int(batch * structured)
    if n:
        a[-n:], b[-n:] = structured_pairs(n)
    da = (a[:, None] // POW8) % 10
    db = (b[:, None] // POW8) % 10
    operands = torch.stack((da, db), 2).reshape(batch, 16)
    result = ((a + b)[:, None] // POW10) % 10
    sequence = torch.cat((operands, torch.full((batch, 1), 10, device=DEVICE), result[:, :-1]), 1)
    return sequence, result, a, b


def autoregressive(model, a, b):
    da = (a[:, None] // POW8) % 10
    db = (b[:, None] // POW8) % 10
    seq = torch.cat((torch.stack((da, db), 2).reshape(a.shape[0], 16), torch.full((a.shape[0], 1), 10, device=DEVICE)), 1)
    out = []
    for _ in range(9):
        digit = model(seq)[:, -1].argmax(1)
        out.append(digit)
        seq = torch.cat((seq, digit[:, None]), 1)
    return (torch.stack(out, 1) * POW10).sum(1)


@torch.no_grad()
def evaluate(model, n=100000, structured=.0, chunk=10000):
    model.eval(); correct = 0; total = 0; margin = 1000.
    while total < n:
        bs = min(chunk, n-total)
        _, _, a, b = sample(bs, structured)
        pred = autoregressive(model, a, b)
        correct += int((pred == a+b).sum())
        total += bs
    model.train()
    return correct, total


def train_stage(model, steps, lr, structured, name):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(.9, .98), weight_decay=.01)
    start = 0
    for step in range(1, steps + 1):
        seq, target, _, _ = sample(structured=structured)
        logits = model(seq)[:, 16:25]
        loss = F.cross_entropy(logits.reshape(-1, 10), target.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step <= 1000:
            warm = min(1., step / 500)
            for g in opt.param_groups: g['lr'] = lr * warm
        if step % 1000 == 0:
            acc = evaluate(model, 20000, 0., 10000)[0]
            sacc = evaluate(model, 20000, 1., 10000)[0]
            print(f'{name} {step}/{steps} loss={loss.item():.5f} random={acc}/20000 structured={sacc}/20000', flush=True)
            torch.save({'model': model.state_dict(), 'ff': model.ff_width, 'stage': name, 'step': step}, f'/workspace/{name}.pt')
    return model


def prune(model, width):
    old = model.ff_width
    importance = model.ff_in.weight.norm(dim=1) * model.ff_out.weight.norm(dim=0)
    keep = importance.topk(width).indices.sort().values
    new = Model(width).to(DEVICE)
    state = model.state_dict()
    new_state = new.state_dict()
    for key in new_state:
        if key == 'ff_in.weight': new_state[key] = state[key][keep]
        elif key == 'ff_out.weight': new_state[key] = state[key][:, keep]
        else: new_state[key] = state[key]
    new.load_state_dict(new_state)
    print(f'pruned {old}->{width}, kept {keep.tolist()}', flush=True)
    return new


def main():
    torch.manual_seed(2025); random.seed(2025)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = Model(4).to(DEVICE)
    model = train_stage(model, 36000, 2e-3, .18, 'teacher4')
    print('teacher final', evaluate(model, 100000, 0.), evaluate(model, 100000, 1.))
    model = prune(model, 3)
    model = train_stage(model, 18000, 2e-5, .30, 'pruned3')
    print('width3 final', evaluate(model, 100000, 0.), evaluate(model, 100000, 1.))
    model = prune(model, 2)
    model = train_stage(model, 30000, 1.2e-5, .30, 'pruned2')
    model = train_stage(model, 12000, 4e-6, .40, 'polish2')
    model = train_stage(model, 6000, 2e-6, .55, 'edge2')
    print('FINAL', evaluate(model, 500000, 0.), evaluate(model, 500000, 1.))
    torch.save({'model': model.state_dict(), 'ff': 2}, '/workspace/final.pt')


if __name__ == '__main__':
    main()
