import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, width, heads, ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, ff), nn.GELU(), nn.Linear(ff, width))

    def forward(self, x, mask):
        y = self.norm1(x)
        x = x + self.attn(y, y, y, attn_mask=mask, need_weights=False)[0]
        return x + self.ff(self.norm2(x))


class Model(nn.Module):
    def __init__(self, width, heads, ff):
        super().__init__()
        self.token = nn.Embedding(11, width)
        self.position = nn.Embedding(25, width)
        self.blocks = nn.ModuleList([Block(width, heads, ff) for _ in range(2)])
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, 10, bias=False)

    def forward(self, tokens):
        length = tokens.shape[1]
        x = self.token(tokens) + self.position.weight[:length]
        mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=tokens.device), 1)
        for block in self.blocks:
            x = block(x, mask)
        return self.output(self.norm(x))


POWERS = None


def examples(batch, device, edge_fraction=0.0):
    global POWERS
    if POWERS is None or POWERS.device != device:
        POWERS = (10 ** torch.arange(8, device=device, dtype=torch.long))
    a_digits = torch.randint(0, 10, (batch, 8), device=device)
    b_digits = torch.randint(0, 10, (batch, 8), device=device)
    a_digits[:, 7] = torch.randint(1, 10, (batch,), device=device)
    b_digits[:, 7] = torch.randint(1, 10, (batch,), device=device)
    a = (a_digits * POWERS).sum(1)
    b = (b_digits * POWERS).sum(1)
    if edge_fraction:
        n = int(batch * edge_fraction)
        # Complementary operands exercise a carry through all eight columns.
        ae = torch.randint(10_000_000, 90_000_001, (n,), device=device)
        be = 100_000_000 - ae
        a[:n], b[:n] = ae, be
        a_digits[:n] = (a[:n, None] // POWERS) % 10
        b_digits[:n] = (b[:n, None] // POWERS) % 10
    total = a + b
    result = (total[:, None] // (10 ** torch.arange(9, device=device))) % 10
    operands = torch.stack((a_digits, b_digits), dim=2).reshape(batch, 16)
    context = torch.cat((operands, torch.full((batch, 1), 10, device=device)), 1)
    inputs = torch.cat((context, result[:, :-1]), 1)
    return inputs, result, a, b


@torch.inference_mode()
def evaluate(model, batches=4, batch=4096, edge_fraction=0.0):
    model.eval()
    exact = correct = count = 0
    for _ in range(batches):
        inputs, result, _, _ = examples(batch, inputs_device(model), edge_fraction)
        sequence = inputs[:, :17]
        guesses = []
        for _ in range(9):
            digit = model(sequence)[:, -1].argmax(1)
            guesses.append(digit)
            sequence = torch.cat((sequence, digit[:, None]), 1)
        prediction = torch.stack(guesses, 1)
        exact += (prediction == result).all(1).sum().item()
        correct += (prediction == result).sum().item()
        count += batch
    model.train()
    return exact / count, correct / (count * 9)


def inputs_device(model):
    return next(model.parameters()).device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=24)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--ff', type=int, default=24)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--batch', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda')
    model = Model(args.width, args.heads, args.ff).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    started = time.time()
    best = 0.0
    for step in range(1, args.steps + 1):
        progress = step / args.steps
        edge = 0.0 if progress < .75 else (0.10 if progress < .9 else 0.20)
        inputs, targets, _, _ = examples(args.batch, device, edge)
        logits = model(inputs)[:, 16:25]
        loss = criterion(logits.reshape(-1, 10), targets.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if progress > .75:
            for group in optimizer.param_groups:
                group['lr'] = 2e-4 if progress < .9 else 5e-5
        optimizer.step()
        if step % 1000 == 0 or step == args.steps:
            exact, digit = evaluate(model, batches=2)
            edge_exact, _ = evaluate(model, batches=1, edge_fraction=1.0)
            best = max(best, exact)
            print(json.dumps({'step': step, 'loss': round(loss.item(), 6),
                              'exact': exact, 'digit': digit, 'edge': edge_exact,
                              'seconds': round(time.time()-started, 1)}), flush=True)
    exact, digit = evaluate(model, batches=16)
    edge_exact, _ = evaluate(model, batches=4, edge_fraction=1.0)
    payload = {'state': {k: v.detach().cpu() for k, v in model.state_dict().items()},
               'config': vars(args), 'exact': exact, 'digit': digit,
               'edge_exact': edge_exact,
               'parameters': sum(p.numel() for p in model.parameters())}
    torch.save(payload, args.out)
    print(json.dumps({k: v for k, v in payload.items() if k != 'state'}), flush=True)


if __name__ == '__main__':
    main()
