import argparse
import random
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


class AddTransformer(nn.Module):
    def __init__(self, ff=10):
        super().__init__()
        d = 20
        self.token = nn.Embedding(11, d)
        self.position = nn.Embedding(25, d)
        self.passes = nn.Embedding(2, d)
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.final_norm = nn.LayerNorm(d)
        self.output = nn.Linear(d, 10, bias=False)

    def forward(self, tokens):
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device)
        x = self.token(tokens) + self.position(positions)
        mask = torch.triu(torch.ones(length, length, device=tokens.device, dtype=torch.bool), 1)
        for p in range(2):
            x = x + self.passes.weight[p]
            z = self.norm1(x)
            x = x + self.attn(z, z, z, attn_mask=mask, need_weights=False)[0]
            x = x + self.ff(self.norm2(x))
        return self.output(self.final_norm(x))


def digits(n, count):
    places = (10 ** torch.arange(count, device=n.device, dtype=torch.long))[None]
    return (n[:, None] // places) % 10


def examples(a, b):
    ad, bd = digits(a, 8), digits(b, 8)
    operands = torch.stack((ad, bd), dim=2).reshape(-1, 16)
    result = digits(a + b, 9)
    x = torch.cat((operands, torch.full_like(a[:, None], 10), result[:, :8]), dim=1)
    return x, result

SPECIAL = torch.tensor([
    10000000, 10000001, 10000009, 10000010, 10000011, 10000099, 10000100,
    10000999, 10001000, 10009999, 10010000, 10099999, 10101010, 10999999,
    11000000, 11111111, 12000000, 19999999, 20000000, 22222222, 30000000,
    33333333, 40000000, 44444444, 49999999, 50000000, 50000001, 55555555,
    60000000, 66666666, 70000000, 77777777, 80000000, 88888888, 89999999,
    90000000, 90909090, 98989898, 99000000, 99900000, 99990000, 99999000,
    99999900, 99999990, 99999998, 99999999
], dtype=torch.long)


def sample_batch(size, device, structured=0.25):
    a = torch.randint(10000000, 100000000, (size,), device=device)
    b = torch.randint(10000000, 100000000, (size,), device=device)
    n = int(size * structured)
    if n:
        quarter = n // 4
        # Long carry boundaries, with small perturbations around round target sums.
        k = quarter
        aa = torch.randint(10000000, 90000001, (k,), device=device)
        totals = torch.tensor([99999999, 100000000, 100000001, 110000000, 150000000], device=device)
        total = totals[torch.randint(0, len(totals), (k,), device=device)]
        bb = total - aa + torch.randint(-10, 11, (k,), device=device)
        bb = bb.clamp(10000000, 99999999)
        a[:k], b[:k] = aa, bb

        # Numbers ending in long runs of zeroes or nines.
        k2 = quarter
        start = quarter
        powers = torch.tensor([10, 100, 1000, 10000, 100000, 1000000, 10000000], device=device)
        p = powers[torch.randint(0, 7, (k2,), device=device)]
        base = torch.randint(10000000, 100000000, (k2,), device=device)
        a[start:start+k2] = (base // p * p).clamp(10000000, 99999999)
        b[start:start+k2] = (torch.randint(10000000, 100000000, (k2,), device=device) // p * p + p - 1).clamp(10000000, 99999999)

        # Repeated and hand-selected difficult patterns.
        k3 = quarter
        start += k2
        special = SPECIAL.to(device)
        a[start:start+k3] = special[torch.randint(0, len(special), (k3,), device=device)]
        b[start:start+k3] = special[torch.randint(0, len(special), (k3,), device=device)]

        # Near lower/upper boundaries and random sparse decimal patterns.
        start += k3
        k4 = n - start
        side = torch.randint(0, 2, (k4,), device=device)
        off = torch.randint(0, 100000, (k4,), device=device)
        a[start:n] = torch.where(side == 0, 10000000 + off, 99999999 - off)
        b[start:n] = special[torch.randint(0, len(special), (k4,), device=device)]
    return examples(a, b)


@torch.inference_mode()
def predict(model, a, b):
    x, y = examples(a, b)
    x = x[:, :17]
    for _ in range(9):
        logits = model(x)
        digit = logits[:, -1].argmax(-1, keepdim=True)
        x = torch.cat((x, digit), 1)
    pred = x[:, 17:26]
    return pred, y


@torch.inference_mode()
def validate(model, count=20000, structured=False):
    model.eval()
    device = next(model.parameters()).device
    hits = total = 0
    chunk = 4096
    while total < count:
        n = min(chunk, count-total)
        if structured:
            x, y = sample_batch(n, device, 1.0)
            a_digits, b_digits = x[:, :16:2], x[:, 1:16:2]
            places = (10 ** torch.arange(8, device=device))[None]
            a, b = (a_digits*places).sum(1), (b_digits*places).sum(1)
        else:
            a = torch.randint(10000000, 100000000, (n,), device=device)
            b = torch.randint(10000000, 100000000, (n,), device=device)
        p, y = predict(model, a, b)
        hits += (p == y).all(1).sum().item()
        total += n
    model.train()
    return hits / total


def tensor_literal(t):
    return repr(t.detach().cpu().tolist())


def export(model, path):
    state_lines = []
    for name, value in model.state_dict().items():
        state_lines.append(f"    {name!r}: torch.tensor({tensor_literal(value)}),")
    template = '''import torch
from torch import nn


class AdditionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        d = 20
        self.token = nn.Embedding(11, d)
        self.position = nn.Embedding(25, d)
        self.passes = nn.Embedding(2, d)
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 10), nn.GELU(), nn.Linear(10, d))
        self.final_norm = nn.LayerNorm(d)
        self.output = nn.Linear(d, 10, bias=False)

    def forward(self, tokens):
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device)
        x = self.token(tokens) + self.position(positions)
        mask = torch.triu(torch.ones(length, length, device=tokens.device, dtype=torch.bool), 1)
        for p in range(2):
            x = x + self.passes.weight[p]
            z = self.norm1(x)
            x = x + self.attn(z, z, z, attn_mask=mask, need_weights=False)[0]
            x = x + self.ff(self.norm2(x))
        return self.output(self.final_norm(x))


_STATE = {
__STATE_LITERAL__
}


def build_model():
    model = AdditionTransformer()
    model.load_state_dict(_STATE)
    model.eval()
    return model, {"architecture": "two-pass shared causal transformer", "digit_order": "least-significant-first"}


def add(model, a: int, b: int) -> int:
    ad = [ord(c) - 48 for c in reversed(str(a))]
    bd = [ord(c) - 48 for c in reversed(str(b))]
    tokens = []
    for x, y in zip(ad, bd):
        tokens.extend((x, y))
    tokens.append(10)
    sequence = torch.tensor([tokens], dtype=torch.long, device=next(model.parameters()).device)
    with torch.inference_mode():
        for _ in range(9):
            digit = int(model(sequence)[0, -1].argmax())
            sequence = torch.cat((sequence, torch.tensor([[digit]], device=sequence.device)), dim=1)
    result = 0
    for digit in reversed(sequence[0, -9:].tolist()):
        result = result * 10 + digit
    return result
'''
    Path(path).write_text(template.replace("__STATE_LITERAL__", "\n".join(state_lines)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=24000)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--batch', type=int, default=4096)
    parser.add_argument('--resume')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda')
    model = AddTransformer().to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=2e-5)
    model.train()
    best = 0.0
    for step in range(1, args.steps + 1):
        structured = 0.25 if step < int(args.steps*0.7) else 0.55
        x, target = sample_batch(args.batch, device, structured)
        logits = model(x)
        loss = F.cross_entropy(logits[:, 16:25].reshape(-1, 10), target.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % 1000 == 0 or step == args.steps:
            random_acc = validate(model, 10000, False)
            edge_acc = validate(model, 10000, True)
            print(f'{step:5d} loss={loss.item():.5f} random={random_acc:.5f} edge={edge_acc:.5f} lr={scheduler.get_last_lr()[0]:.2g}', flush=True)
            torch.save(model.state_dict(), '/workspace/latest.pt')
            export(model, '/workspace/submission.py')
            if min(random_acc, edge_acc) > best:
                best = min(random_acc, edge_acc)
                torch.save(model.state_dict(), '/workspace/best.pt')
                Path('/workspace/best_score.txt').write_text(f'{step} {random_acc} {edge_acc}\n')
    print('parameters', sum(p.numel() for p in model.parameters()))


if __name__ == '__main__':
    main()
