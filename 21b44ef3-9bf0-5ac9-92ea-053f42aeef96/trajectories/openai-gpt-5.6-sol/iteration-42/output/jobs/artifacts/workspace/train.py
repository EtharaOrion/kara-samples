import argparse
import copy
import importlib.util
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path('/workspace')
DEVICE = 'cuda'
BATCH = 8192
LOW = 10_000_000
HIGH = 99_999_999


class TrainModel(nn.Module):
    def __init__(self, ff=4):
        super().__init__()
        d = 20
        self.ff_width = ff
        self.token = nn.Embedding(11, d)
        self.pos_left = nn.Parameter(torch.empty(25, 2))
        self.pos_right = nn.Parameter(torch.empty(2, d))
        self.norm_attn = nn.LayerNorm(d)
        self.key = nn.Linear(d, 5, bias=False)
        self.value = nn.Linear(d, 5, bias=False)
        self.query = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(2)])
        self.attn_out = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(2)])
        self.norm_ff = nn.LayerNorm(d)
        self.ff_in = nn.Linear(d, ff, bias=False)
        self.ff_out = nn.Linear(ff, d, bias=False)
        self.norm_final = nn.LayerNorm(d)
        self.classifier = nn.Linear(d, 10, bias=False)
        self.register_buffer('causal', torch.triu(torch.ones(25, 25, dtype=torch.bool), 1), persistent=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.pos_left, std=0.2)
        nn.init.normal_(self.pos_right, std=0.2)

    def forward(self, tokens):
        n = tokens.shape[1]
        x = self.token(tokens) + self.pos_left[:n] @ self.pos_right
        mask = self.causal[:n, :n]
        for layer in range(2):
            z = self.norm_attn(x)
            q = self.query[layer](z).view(-1, n, 4, 5).transpose(1, 2)
            k = self.key(z).unsqueeze(1)
            v = self.value(z).unsqueeze(1)
            attn = (q @ k.transpose(-2, -1) / math.sqrt(5)).masked_fill(mask, -torch.inf).softmax(-1)
            x = x + self.attn_out[layer]((attn @ v).transpose(1, 2).reshape(-1, n, 20))
            x = x + self.ff_out(torch.nn.functional.gelu(self.ff_in(self.norm_ff(x))))
        return self.classifier(self.norm_final(x))


def digits(x, count=8):
    out = []
    for _ in range(count):
        out.append(x.remainder(10))
        x = torch.div(x, 10, rounding_mode='floor')
    return torch.stack(out, 1)


def structured(n, device=DEVICE):
    a = torch.randint(LOW, HIGH + 1, (n,), device=device)
    b = torch.randint(LOW, HIGH + 1, (n,), device=device)
    kind = torch.randint(0, 8, (n,), device=device)

    # Complements near 10^8 and other decimal boundaries.
    m = kind == 0
    target = 100_000_000 + torch.randint(-100, 101, (n,), device=device)
    bb = target - a
    valid = m & (bb >= LOW) & (bb <= HIGH)
    b = torch.where(valid, bb, b)

    # Asymmetric suffixes of nines/zeros provoke carries starting at every column.
    for k in range(1, 8):
        m = (kind == 1) & (torch.randint(1, 8, (n,), device=device) == k)
        p = 10 ** k
        aa = torch.div(a, p, rounding_mode='floor') * p + (p - 1)
        bsmall = torch.randint(1, p + 1, (n,), device=device)
        a = torch.where(m, aa, a)
        b = torch.where(m, torch.maximum(bsmall, torch.full_like(bsmall, LOW)), b)

    # Round one side to a random decimal scale and perturb the other nearby.
    scale_idx = torch.randint(1, 8, (n,), device=device)
    powers = torch.tensor([10,100,1000,10000,100000,1000000,10000000], device=device)
    p = powers[scale_idx - 1]
    rounded = torch.div(a, p, rounding_mode='floor') * p
    m = kind == 2
    a = torch.where(m, torch.maximum(rounded, torch.full_like(a, LOW)), a)

    # Repeated digit operands.
    reps = torch.tensor([11_111_111,22_222_222,33_333_333,44_444_444,55_555_555,66_666_666,77_777_777,88_888_888,99_999_999], device=device)
    m = kind == 3
    a = torch.where(m, reps[torch.randint(0, 9, (n,), device=device)], a)

    # Sparse zero/nine digit patterns, always retaining a nonzero leading digit.
    m = (kind == 4) | (kind == 5)
    lead = torch.randint(1, 10, (n, 1), device=device)
    choices = torch.randint(0, 2, (n, 7), device=device) * 9
    sparse = torch.cat((choices, lead), 1)
    place = torch.tensor([1,10,100,1000,10000,100000,1000000,10000000], device=device)
    sparse_num = (sparse * place).sum(1)
    a = torch.where(m, sparse_num, a)

    # Near extrema.
    m = kind == 6
    edge_a = torch.where(torch.rand(n, device=device) < .5, LOW + torch.randint(0, 10000, (n,), device=device), HIGH - torch.randint(0, 10000, (n,), device=device))
    a = torch.where(m, edge_a, a)

    # Unequal carry runs: ...000 + ...999 or ...999 + ...001 at random offsets.
    m = kind == 7
    k = torch.randint(1, 8, (n,), device=device)
    p = powers[k - 1]
    prefix = torch.div(a, p, rounding_mode='floor') * p
    aa = prefix + (p - 1)
    bb = torch.randint(1, 10, (n,), device=device) * torch.maximum(torch.div(p, 10, rounding_mode='floor'), torch.ones_like(p))
    valid = (aa >= LOW) & (aa <= HIGH) & (bb >= LOW) & (bb <= HIGH)
    a = torch.where(m & valid, aa, a)
    b = torch.where(m & valid, bb, b)
    return a, b


def batch(n=BATCH, structured_fraction=.18):
    a = torch.randint(LOW, HIGH + 1, (n,), device=DEVICE)
    b = torch.randint(LOW, HIGH + 1, (n,), device=DEVICE)
    count = int(n * structured_fraction)
    if count:
        sa, sb = structured(count)
        a[:count], b[:count] = sa, sb
    ad, bd = digits(a), digits(b)
    target = digits(a + b, 9)
    operands = torch.stack((ad, bd), 2).reshape(n, 16)
    sequence = torch.cat((operands, torch.full((n, 1), 10, device=DEVICE), target[:, :8]), 1)
    return sequence, target


@torch.no_grad()
def accuracy(model, n=100000, structured_data=False, chunk=10000):
    model.eval()
    correct = total = 0
    minimum_margin = 1e9
    for start in range(0, n, chunk):
        size = min(chunk, n-start)
        a, b = structured(size) if structured_data else (
            torch.randint(LOW, HIGH+1, (size,), device=DEVICE),
            torch.randint(LOW, HIGH+1, (size,), device=DEVICE))
        ad, bd = digits(a), digits(b)
        seq = torch.cat((torch.stack((ad, bd), 2).reshape(size, 16), torch.full((size,1),10,device=DEVICE)), 1)
        predictions = []
        margins = []
        for _ in range(9):
            logits = model(seq)[:, -1]
            top = logits.topk(2, 1).values
            margins.append(top[:, 0] - top[:, 1])
            digit = logits.argmax(1)
            predictions.append(digit)
            seq = torch.cat((seq, digit[:, None]), 1)
        pred = torch.stack(predictions, 1)
        true = digits(a+b, 9)
        correct += (pred == true).all(1).sum().item()
        total += size
        minimum_margin = min(minimum_margin, torch.stack(margins,1).min().item())
    model.train()
    return correct, total, minimum_margin


def train_phase(model, steps, lr, structured_fraction, tag):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(.9,.98), weight_decay=.01, fused=True)
    scheduler = None
    if lr >= 1e-3:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1., (s+1)/500))
    started = time.time()
    for step in range(1, steps+1):
        seq, target = batch(structured_fraction=structured_fraction)
        optimizer.zero_grad(set_to_none=True)
        logits = model(seq)[:, 16:25]
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1,10), target.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler: scheduler.step()
        if step % 1000 == 0 or step == steps:
            elapsed = time.time()-started
            print(f'{tag} {step}/{steps} loss={loss.item():.5f} lr={optimizer.param_groups[0]["lr"]:.2g} sec={elapsed:.1f}', flush=True)
        if step % 6000 == 0 or step == steps:
            c,t,m = accuracy(model, 20000, False)
            cs,ts,ms = accuracy(model, 20000, True)
            print(f'  AR uniform={c/t:.6f} structured={cs/ts:.6f} margin={min(m,ms):.3f}', flush=True)
            torch.save(model.state_dict(), ROOT / f'{tag}_{step}.pt')
    return model


def prune(model, width):
    assert width < model.ff_width
    score = model.ff_in.weight.norm(dim=1) * model.ff_out.weight.norm(dim=0)
    keep = score.topk(width).indices.sort().values
    result = TrainModel(width).to(DEVICE)
    old = model.state_dict()
    new = result.state_dict()
    for name in new:
        if name == 'ff_in.weight': new[name].copy_(old[name][keep])
        elif name == 'ff_out.weight': new[name].copy_(old[name][:, keep])
        else: new[name].copy_(old[name])
    return result


def export_plain(model):
    template = (ROOT/'submission.py').read_text()
    marker = '_TRAINED_STATE = None'
    values = []
    for p in model.parameters():
        flat = p.detach().float().cpu().reshape(-1).tolist()
        values.append('[' + ','.join(format(x, '.9g') for x in flat) + ']')
    replacement = '_TRAINED_STATE = [\n' + ',\n'.join(values) + '\n]'
    assert marker in template
    (ROOT/'submission.py').write_text(template.replace(marker, replacement))
    print('exported plain trained model', sum(p.numel() for p in model.parameters()), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str)
    parser.add_argument('--only-export', action='store_true')
    args = parser.parse_args()
    torch.manual_seed(2025); random.seed(2025)
    torch.backends.cuda.matmul.allow_tf32 = True
    if args.resume:
        state = torch.load(args.resume, map_location=DEVICE, weights_only=True)
        ff = state['ff_in.weight'].shape[0]
        model = TrainModel(ff).to(DEVICE); model.load_state_dict(state)
    else:
        model = TrainModel(4).to(DEVICE)
    if args.only_export:
        export_plain(model); return
    if model.ff_width == 4:
        model = train_phase(model, 36000, 2e-3, .18, 'teacher')
        c,t,m = accuracy(model, 100000, False); cs,ts,ms = accuracy(model,100000,True)
        print('TEACHER',c,t,cs,ts,m,ms,flush=True)
        if min(c/t,cs/ts) < .9999: raise RuntimeError('teacher too weak to prune')
        model = prune(model, 3)
        model = train_phase(model, 18000, 2e-5, .30, 'width3')
    if model.ff_width == 3:
        c,t,m = accuracy(model, 100000, False); cs,ts,ms = accuracy(model,100000,True)
        print('WIDTH3',c,t,cs,ts,m,ms,flush=True)
        if min(c/t,cs/ts) < .995: raise RuntimeError('width3 too weak to prune')
        model = prune(model, 2)
        model = train_phase(model, 30000, 1.2e-5, .35, 'width2')
    model = train_phase(model, 12000, 4e-6, .40, 'polish')
    torch.save(model.state_dict(), ROOT/'final_width2.pt')
    c,t,m = accuracy(model, 500000, False); cs,ts,ms = accuracy(model,500000,True)
    print('FINAL',c,t,cs,ts,m,ms,flush=True)
    export_plain(model)


if __name__ == '__main__':
    main()
