import argparse
import json
import math
import os
import random
import time

import torch
from torch import nn
import torch.nn.functional as F

LIMIT = 100_000_000_000_000
DIGITS = 15
WIDTH = 16
HEADS = 2


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(WIDTH)
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH)
        self.proj = nn.Linear(WIDTH, WIDTH)
        self.ln2 = nn.LayerNorm(WIDTH)
        self.fc1 = nn.Linear(WIDTH, 32)
        self.fc2 = nn.Linear(32, WIDTH)

    def forward(self, x, mask):
        z = self.ln1(x)
        q, k, v = self.qkv(z).chunk(3, -1)
        n = x.shape[1]
        q = q.view(-1, n, HEADS, WIDTH // HEADS).transpose(1, 2)
        k = k.view(-1, n, HEADS, WIDTH // HEADS).transpose(1, 2)
        v = v.view(-1, n, HEADS, WIDTH // HEADS).transpose(1, 2)
        att = F.softmax((q @ k.transpose(-2, -1)) / math.sqrt(WIDTH // HEADS) + mask, -1)
        x = x + self.proj((att @ v).transpose(1, 2).reshape(-1, n, WIDTH))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class Adder(nn.Module):
    def __init__(self, sharing):
        super().__init__()
        self.digit = nn.Embedding(10, WIDTH)
        self.role = nn.Embedding(3, WIDTH)
        self.blocks = nn.ModuleList([Block(), Block()])
        if sharing == 'fc1':
            self.blocks[1].fc1.weight = self.blocks[0].fc1.weight
        elif sharing == 'fc1bias':
            self.blocks[1].fc1.bias = self.blocks[0].fc1.bias
        elif sharing == 'fc2':
            self.blocks[1].fc2.weight = self.blocks[0].fc2.weight
        elif sharing == 'fc1full':
            self.blocks[1].fc1 = self.blocks[0].fc1
        elif sharing == 'none':
            pass
        else:
            raise ValueError(sharing)
        self.final = nn.LayerNorm(WIDTH)
        self.head = nn.Linear(WIDTH, 10)

    def forward(self, tok):
        n = tok.shape[1]
        pos = torch.arange(n, device=tok.device)
        x = self.digit(tok) + self.role(pos.remainder(3))[None]
        d = pos[:, None] - pos[None, :]
        mask = torch.zeros(n, n, device=tok.device, dtype=x.dtype)
        mask.masked_fill_((d < 0) | (d > 5), float('-inf'))
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.final(x))


def digits(x):
    out = []
    for _ in range(DIGITS):
        out.append(x.remainder(10))
        x = torch.div(x, 10, rounding_mode='floor')
    return torch.stack(out, 1)


def batch_data(batch, device, structured=0.0):
    a = torch.randint(LIMIT, (batch,), device=device)
    b = torch.randint(LIMIT, (batch,), device=device)
    if structured:
        sel = torch.rand(batch, device=device)
        # Repeated-digit operands stress persistent carry/no-carry regimes.
        rep = sel < structured * 0.5
        ds = torch.randint(10, (batch,), device=device)
        repeated = ds * 11_111_111_111_111
        a = torch.where(rep, repeated, a)
        # Long suffixes of nines force carry chains of varied lengths.
        chain = (sel >= structured * 0.5) & (sel < structured)
        lens = torch.randint(2, 15, (batch,), device=device)
        powers = torch.tensor([10 ** i for i in range(15)], device=device, dtype=torch.long)
        nines = powers[lens] - 1
        prefix = torch.div(a, powers[lens], rounding_mode='floor') * powers[lens]
        a = torch.where(chain, prefix + nines, a)
    ad, bd, yd = digits(a), digits(b), digits(a + b)
    seq = torch.empty(batch, DIGITS * 3, dtype=torch.long, device=device)
    seq[:, 0::3], seq[:, 1::3], seq[:, 2::3] = ad, bd, yd
    return seq, yd


@torch.inference_mode()
def greedy(model, count, batch=2048, seed=987654321):
    gen = torch.Generator(device='cpu').manual_seed(seed)
    aa = torch.randint(LIMIT, (count,), generator=gen)
    bb = torch.randint(LIMIT, (count,), generator=gen)
    correct = 0
    dev = next(model.parameters()).device
    for start in range(0, count, batch):
        a = aa[start:start+batch].to(dev)
        b = bb[start:start+batch].to(dev)
        ad, bd = digits(a), digits(b)
        target = digits(a + b)
        seq = torch.empty(a.numel(), DIGITS * 3, dtype=torch.long, device=dev)
        seq.zero_()
        seq[:, 0::3], seq[:, 1::3] = ad, bd
        pred = torch.empty_like(target)
        for i in range(DIGITS):
            y = model(seq[:, :3*i+2])[:, -1].argmax(-1)
            pred[:, i] = y
            seq[:, 3*i+2] = y
        correct += (pred == target).all(1).sum().item()
    return correct / count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sharing', default='fc1', choices=['fc1','fc1bias','fc2','fc1full','none'])
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', default='/workspace/model.pt')
    args = ap.parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed)
    torch.set_float32_matmul_precision('high')
    dev = torch.device('cuda')
    model = Adder(args.sharing).to(dev)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=.005)
    milestones = {3000: 1.5e-3, 4500: 7e-4, 6000: 3e-4, 7500: 1e-4}
    t = time.time()
    for step in range(1, args.steps + 1):
        if step in milestones:
            for g in opt.param_groups: g['lr'] = milestones[step]
        seq, y = batch_data(args.batch, dev, structured=0.0)
        logits = model(seq)
        loss = F.cross_entropy(logits[:, 1::3].reshape(-1, 10), y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 250 == 0 or step == 1:
            digit_acc = (logits[:, 1::3].argmax(-1) == y).float().mean().item()
            print(json.dumps({'step':step,'loss':round(loss.item(),6),'teacher_digit':round(digit_acc,6),'lr':opt.param_groups[0]['lr'],'sec':round(time.time()-t,1)}), flush=True)
    acc = greedy(model, 10000)
    print(json.dumps({'sharing':args.sharing,'parameters':params,'greedy_10000':acc}), flush=True)
    torch.save({'state':model.state_dict(),'sharing':args.sharing,'parameters':params,'accuracy':acc}, args.out)

if __name__ == '__main__': main()
