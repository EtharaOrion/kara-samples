import math
import os
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

MAX_N = 100_000_000_000_000
DEVICE = "cuda"
BATCH = int(os.environ.get("BATCH", "4096"))
STEPS = int(os.environ.get("STEPS", "28000"))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(10)
        self.qkv = nn.Linear(10, 30)
        self.proj = nn.Linear(10, 10)
        self.ln2 = nn.LayerNorm(10)
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        z = self.ln1(x)
        q, k, v = self.qkv(z).chunk(3, dim=-1)
        n = x.shape[1]
        q = q.view(-1, n, 2, 5).transpose(1, 2)
        k = k.view(-1, n, 2, 5).transpose(1, 2)
        v = v.view(-1, n, 2, 5).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(-1, n, 10))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a_emb = nn.Embedding(11, 10)
        self.b_emb = nn.Embedding(11, 10)
        self.out_emb = nn.Embedding(11, 10)
        self.pos = nn.Parameter(torch.empty(30, 10))
        self.blocks = nn.ModuleList([Block(), Block()])
        self.norm = nn.LayerNorm(10)
        self.head = nn.Linear(10, 10)
        nn.init.normal_(self.pos, std=.02)

    def forward(self, ad, bd, prev):
        x = torch.cat((self.a_emb(ad) + self.b_emb(bd), self.out_emb(prev)), 1)
        x = x + self.pos[:x.shape[1]]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))[:, 14:]


def to_digits(n):
    places = torch.tensor([10 ** i for i in range(15)], device=n.device)
    return (n[:, None] // places % 10).long()


def digits_to_int(d):
    places = torch.tensor([10 ** i for i in range(15)], device=d.device)
    return (d * places).sum(1)


def make_batch(batch, step):
    # Every label is generated from a complete operand pair; structured operands
    # vary on every batch rather than enumerating a transition table.
    a = torch.randint(MAX_N, (batch,), device=DEVICE)
    b = torch.randint(MAX_N, (batch,), device=DEVICE)
    ad, bd = to_digits(a), to_digits(b)
    structured = batch * (35 if step < 22000 else 50) // 100
    if structured:
        rows = torch.arange(structured, device=DEVICE)
        family = rows % 5
        start = torch.randint(0, 14, (structured,), device=DEVICE)
        length = torch.randint(1, 15, (structured,), device=DEVICE)
        end = torch.minimum(start + length, torch.full_like(start, 14))
        cols = torch.arange(14, device=DEVICE)[None, :]
        chain = (cols >= start[:, None]) & (cols <= end[:, None])
        continuation = chain & (cols != start[:, None])

        # Carry chains: initiating column sums above nine; following columns sum nine.
        cr = family == 0
        if cr.any():
            rr = rows[cr]
            mask = chain[cr]
            x = torch.randint(0, 10, (len(rr), 14), device=DEVICE)
            ad[rr, :14] = torch.where(mask, x, ad[rr, :14])
            bd[rr, :14] = torch.where(mask, 9 - x, bd[rr, :14])
            sc = start[cr]
            av = torch.randint(1, 10, (len(rr),), device=DEVICE)
            ad[rr, sc] = av
            bd[rr, sc] = torch.randint(0, 10, (len(rr),), device=DEVICE)
            bd[rr, sc] = torch.maximum(bd[rr, sc], 10 - av)

        # Matched non-carry runs contrast sharply with propagating nines.
        nc = family == 1
        if nc.any():
            rr = rows[nc]
            mask = chain[nc]
            x = torch.randint(0, 9, (len(rr), 14), device=DEVICE)
            ad[rr, :14] = torch.where(mask, x, ad[rr, :14])
            bd[rr, :14] = torch.where(mask, 8 - x, bd[rr, :14])

        # Sparse boundaries, heavily including the historically difficult 5+5.
        sp = family == 2
        if sp.any():
            rr = rows[sp]
            ad[rr] = 0
            bd[rr] = 0
            p = start[sp]
            av = torch.randint(1, 10, (len(rr),), device=DEVICE)
            force = torch.arange(len(rr), device=DEVICE) % 2 == 0
            av = torch.where(force, torch.full_like(av, 5), av)
            bv = torch.where(force, torch.full_like(av, 5), 10 - av)
            ad[rr, 0] = av
            bd[rr, 0] = bv
            bd[rr, p] = torch.where(p == 0, bd[rr, p], torch.ones_like(p))

        # Repeated and blockwise digit patterns.
        rp = family == 3
        if rp.any():
            rr = rows[rp]
            x = torch.randint(0, 10, (len(rr), 1), device=DEVICE)
            y = torch.randint(0, 10, (len(rr), 1), device=DEVICE)
            ad[rr, :14] = x
            bd[rr, :14] = y
            cut = start[rp]
            mask = cols >= cut[:, None]
            ad[rr, :14] = torch.where(mask, torch.randint(0, 10, (len(rr), 1), device=DEVICE), ad[rr, :14])

        # Complementary random columns, including overflow at the top.
        cp = family == 4
        if cp.any():
            rr = rows[cp]
            x = torch.randint(0, 10, (len(rr), 14), device=DEVICE)
            ad[rr, :14] = x
            bd[rr, :14] = 9 - x
            p = start[cp]
            av = ad[rr, p]
            bd[rr, p] = torch.minimum(torch.full_like(av, 9), 10 - av)

        ad[:structured, 14] = 0
        bd[:structured, 14] = 0
        a[:structured] = digits_to_int(ad[:structured])
        b[:structured] = digits_to_int(bd[:structured])

    target = to_digits(a + b)
    return ad, bd, target


def lr_for(step):
    if step < 800:
        return 3e-3 * (step + 1) / 800
    if step < 12000: return 3e-3
    if step < 20000: return 1e-3
    if step < 25000: return 3e-4
    return 1e-4


@torch.no_grad()
def greedy_errors(model, total=65536, structured=False):
    model.eval()
    errors = 0
    for offset in range(0, total, 4096):
        n = min(4096, total-offset)
        ad, bd, target = make_batch(n, 27000 if structured else 0)
        if not structured:
            a = torch.randint(MAX_N, (n,), device=DEVICE)
            b = torch.randint(MAX_N, (n,), device=DEVICE)
            ad, bd, target = to_digits(a), to_digits(b), to_digits(a+b)
        prev = torch.empty((n, 0), dtype=torch.long, device=DEVICE)
        pred = []
        for _ in range(15):
            d = model(ad, bd, prev)[:, -1].argmax(1)
            pred.append(d)
            prev = torch.cat((prev, d[:, None]), 1)
        pred = torch.stack(pred, 1)
        errors += (pred != target).any(1).sum().item()
    model.train()
    return errors


def export(model):
    path = Path('/workspace/submission.py')
    text = path.read_text()
    marker = '# Placeholder values are replaced by train.py after fitting.\n_WEIGHTS = None'
    values = []
    for p in model.parameters():
        flat = p.detach().float().cpu().reshape(-1).tolist()
        values.append('[' + ','.join(format(x, '.9g') for x in flat) + ']')
    replacement = '# Trained on complete randomly generated operand pairs.\n_WEIGHTS = [\n' + ',\n'.join(values) + '\n]'
    if marker not in text:
        begin = text.index('# Trained on complete randomly generated operand pairs.')
        end = text.index('\n\n\ndef build_model', begin)
        text = text[:begin] + replacement + text[end:]
    else:
        text = text.replace(marker, replacement)
    path.write_text(text)


def main():
    torch.manual_seed(20250828)
    random.seed(20250828)
    torch.set_float32_matmul_precision('high')
    model = Model().to(DEVICE)
    print('parameters', sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=.002, fused=True)
    best = None
    start_time = time.time()
    for step in range(STEPS):
        lr = lr_for(step)
        for g in opt.param_groups: g['lr'] = lr
        ad, bd, target = make_batch(BATCH, step)
        logits = model(ad, bd, target[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, 10), target.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 1000 == 0:
            print(step+1, f'loss={loss.item():.6g}', f'lr={lr:g}', f't={time.time()-start_time:.0f}', flush=True)
            torch.save(model.state_dict(), '/workspace/latest.pt')
        if step + 1 >= 20000 and (step + 1) % 2000 == 0:
            eu = greedy_errors(model, 65536, False)
            es = greedy_errors(model, 65536, True)
            print('validation', eu, es, flush=True)
            score = eu + es
            if best is None or score <= best:
                best = score
                torch.save(model.state_dict(), '/workspace/best.pt')
    if Path('/workspace/best.pt').exists():
        model.load_state_dict(torch.load('/workspace/best.pt', weights_only=True))
    export(model)
    torch.save(model.state_dict(), '/workspace/final.pt')
    print('final validation', greedy_errors(model, 262144, False), greedy_errors(model, 262144, True), flush=True)


if __name__ == '__main__':
    main()
