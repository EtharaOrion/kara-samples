import argparse
import importlib.util
import math
import os
import random
import sys

import torch
from torch import nn

DEVICE = 'cuda'
POW10 = torch.tensor([10 ** i for i in range(9)], device=DEVICE, dtype=torch.long)


class TrainModel(nn.Module):
    def __init__(self, ff_width=4):
        super().__init__()
        d = 20
        self.token = nn.Embedding(11, d)
        self.pos_left = nn.Parameter(torch.randn(25, 2) * .02)
        self.pos_right = nn.Parameter(torch.randn(2, d) * .02)
        self.norm_attn = nn.LayerNorm(d)
        self.norm_ff = nn.LayerNorm(d)
        self.final_norm = nn.LayerNorm(d)
        self.key = nn.Linear(d, 5, bias=False)
        self.value = nn.Linear(d, 5, bias=False)
        self.query = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(2)])
        self.attn_out = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(2)])
        self.ff_in = nn.Linear(d, ff_width, bias=False)
        self.ff_out = nn.Linear(ff_width, d, bias=False)
        self.output = nn.Linear(d, 10, bias=False)

    def forward(self, tokens):
        n = tokens.shape[1]
        x = self.token(tokens) + self.pos_left[:n] @ self.pos_right
        mask = torch.ones(n, n, dtype=torch.bool, device=tokens.device).triu(1)
        for layer in range(2):
            z = self.norm_attn(x)
            q = self.query[layer](z).view(z.shape[0], n, 4, 5).transpose(1, 2)
            k, v = self.key(z).unsqueeze(1), self.value(z).unsqueeze(1)
            attention = (q @ k.transpose(-2, -1) * (5 ** -.5)).masked_fill(mask, -torch.inf).softmax(-1) @ v
            x = x + self.attn_out[layer](attention.transpose(1, 2).reshape(z.shape[0], n, 20))
            x = x + self.ff_out(torch.nn.functional.gelu(self.ff_in(self.norm_ff(x))))
        return self.output(self.final_norm(x))


def structured(count):
    a = torch.randint(10_000_000, 100_000_000, (count,), device=DEVICE)
    b = torch.randint(10_000_000, 100_000_000, (count,), device=DEVICE)
    mode = torch.randint(0, 7, (count,), device=DEVICE)
    # Exact and nearby complements around 100M/110M/150M/190M.
    ix = mode == 0
    target = torch.tensor([100_000_000, 110_000_000, 150_000_000, 190_000_000], device=DEVICE)[torch.randint(0, 4, (count,), device=DEVICE)]
    delta = torch.randint(-20, 21, (count,), device=DEVICE)
    b[ix] = (target[ix] - a[ix] + delta[ix]).clamp(10_000_000, 99_999_999)
    # Asymmetric suffixes of 9s and 0s, spanning every carry length.
    ix = mode == 1
    k = torch.randint(1, 8, (count,), device=DEVICE)
    p = POW10[k]
    a9 = (a // p) * p + p - 1
    bsmall = (b // p) * p + torch.randint(0, 20, (count,), device=DEVICE).minimum(p - 1)
    a[ix], b[ix] = a9[ix], bsmall[ix]
    ix = mode == 2
    k2 = torch.randint(1, 8, (count,), device=DEVICE)
    p2 = POW10[k2]
    b9 = (b // p2) * p2 + p2 - 1
    asmall = (a // p2) * p2 + torch.randint(0, 20, (count,), device=DEVICE).minimum(p2 - 1)
    a[ix], b[ix] = asmall[ix], b9[ix]
    # Decimal boundaries with independent perturbations.
    ix = mode == 3
    k3 = torch.randint(1, 8, (count,), device=DEVICE)
    p3 = POW10[k3]
    da = torch.randint(-10, 11, (count,), device=DEVICE)
    db = torch.randint(-10, 11, (count,), device=DEVICE)
    a[ix] = (((a // p3) * p3 + da).clamp(10_000_000, 99_999_999))[ix]
    b[ix] = (((b // p3) * p3 + db).clamp(10_000_000, 99_999_999))[ix]
    # Repeated digits.
    ix = mode == 4
    rep = torch.tensor([11_111_111,22_222_222,33_333_333,44_444_444,55_555_555,66_666_666,77_777_777,88_888_888,99_999_999], device=DEVICE)
    a[ix] = rep[torch.randint(0, 9, (count,), device=DEVICE)][ix]
    # Extremes and sparse round values.
    ix = mode == 5
    edge = torch.tensor([10_000_000,10_000_001,10_000_009,10_000_010,10_000_099,10_001_000,10_100_000,50_000_000,89_999_999,90_000_000,99_000_000,99_900_000,99_990_000,99_999_000,99_999_900,99_999_990,99_999_998,99_999_999], device=DEVICE)
    a[ix] = edge[torch.randint(0, len(edge), (count,), device=DEVICE)][ix]
    b[ix] = edge[torch.randint(0, len(edge), (count,), device=DEVICE)][ix]
    return a, b


def batch(size, structured_fraction=.4):
    a = torch.randint(10_000_000, 100_000_000, (size,), device=DEVICE)
    b = torch.randint(10_000_000, 100_000_000, (size,), device=DEVICE)
    n = int(size * structured_fraction)
    if n:
        sa, sb = structured(n)
        a[:n], b[:n] = sa, sb
    places8 = POW10[:8]
    ad = (a[:, None] // places8) % 10
    bd = (b[:, None] // places8) % 10
    result = ((a + b)[:, None] // POW10) % 10
    operands = torch.stack((ad, bd), -1).reshape(size, 16)
    tokens = torch.cat((operands, torch.full((size, 1), 10, device=DEVICE), result[:, :8]), 1)
    return tokens, result


@torch.no_grad()
def evaluate(model, count=100000, structured_fraction=0.5, chunk=10000):
    model.eval()
    correct = 0
    minimum_margin = 100.
    for start in range(0, count, chunk):
        n = min(chunk, count - start)
        tokens, target = batch(n, structured_fraction)
        seq = tokens[:, :17]
        for digit in range(9):
            logits = model(seq)[:, -1]
            values, indices = logits.topk(2, dim=-1)
            pred = indices[:, 0]
            minimum_margin = min(minimum_margin, (values[:, 0] - values[:, 1]).min().item())
            seq = torch.cat((seq, pred[:, None]), 1)
        correct += (seq[:, -9:] == target).all(1).sum().item()
    model.train()
    return correct / count, minimum_margin


def train_steps(model, optimizer, steps, batch_size, structured_fraction, start_step=0):
    model.train()
    for step in range(1, steps + 1):
        tokens, target = batch(batch_size, structured_fraction)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(tokens)[:, 16:25]
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 10), target.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 1000 == 0:
            teacher = (logits.argmax(-1) == target).all(1).float().mean().item()
            print(f'step {start_step+step} loss {loss.item():.5f} teacher_exact {teacher:.5f}', flush=True)
        if step % 4000 == 0:
            torch.save(model.state_dict(), '/workspace/checkpoint.pt')


def prune(model, width):
    old_width = model.ff_in.weight.shape[0]
    while old_width > width:
        score = model.ff_in.weight.norm(dim=1) * model.ff_out.weight.norm(dim=0)
        keep = [i for i in range(old_width) if i != score.argmin().item()]
        smaller = TrainModel(old_width - 1).to(DEVICE)
        state = model.state_dict()
        state['ff_in.weight'] = state['ff_in.weight'][keep]
        state['ff_out.weight'] = state['ff_out.weight'][:, keep]
        smaller.load_state_dict(state)
        model, old_width = smaller, old_width - 1
    return model


def export(model):
    source_path = '/workspace/submission.py'
    source = open(source_path).read()
    start = source.index('_WEIGHTS = ')
    end = source.index('\n\n\ndef build_model', start)
    state = {name: parameter.detach().float().cpu().reshape(-1).tolist() for name, parameter in model.named_parameters()}
    replacement = '_WEIGHTS = ' + repr(state)
    fresh = source[:start] + replacement + source[end:]
    new_path = '/workspace/submission.new'
    with open(new_path, 'w') as handle:
        handle.write(fresh)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(new_path, source_path)
    print('exported', sum(p.numel() for p in model.parameters()), 'parameters', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume')
    parser.add_argument('--steps', type=int, default=36000)
    parser.add_argument('--batch-size', type=int, default=8192)
    args = parser.parse_args()
    torch.manual_seed(2801)
    model = TrainModel(4).to(DEVICE)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=DEVICE, weights_only=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(.9, .98), weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=5e-5)
    for segment in range(0, args.steps, 2000):
        train_steps(model, optimizer, min(2000, args.steps-segment), args.batch_size, .35, segment)
        for _ in range(min(2000, args.steps-segment)): scheduler.step()
    print('teacher validation', evaluate(model, 50000), flush=True)
    for width, steps, lr in [(3, 16000, 2e-5), (2, 30000, 1e-5)]:
        model = prune(model, width)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(.9, .98), weight_decay=0)
        train_steps(model, optimizer, steps, args.batch_size, .5)
        print('width', width, 'validation', evaluate(model, 100000), flush=True)
        torch.save(model.state_dict(), f'/workspace/width{width}.pt')
        if width == 2:
            export(model)


if __name__ == '__main__':
    main()
