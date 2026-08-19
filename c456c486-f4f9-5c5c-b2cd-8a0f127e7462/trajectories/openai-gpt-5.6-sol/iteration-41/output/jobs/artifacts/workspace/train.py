import argparse
import importlib.util
import math
import random
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

D = 9
HEADS = 3
MAX_VALUE = 100_000_000_000_000
POW10 = torch.tensor([10 ** i for i in range(15)], dtype=torch.long)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D)
        self.proj = nn.Linear(D, D)

    def forward(self, x):
        z = self.norm(x)
        q, k, v = self.qkv(z).chunk(3, dim=-1)
        b, n, _ = q.shape
        q = q.view(b, n, HEADS, 3).transpose(1, 2)
        k = k.view(b, n, HEADS, 3).transpose(1, 2)
        v = v.view(b, n, HEADS, 3).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return x + self.proj(y.transpose(1, 2).reshape(b, n, D))


class AddTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.a_embed = nn.Embedding(11, D)
        self.b_embed = nn.Embedding(11, D)
        self.out_embed = nn.Embedding(10, D)
        self.position = nn.Parameter(torch.empty(29, 3))
        self.attn1 = Attention()
        self.ff_norm = nn.LayerNorm(D)
        self.ff1 = nn.Linear(D, 1)
        self.ff2 = nn.Linear(1, D)
        self.attn2 = Attention()
        self.final_norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, 10)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, a_digits, b_digits, previous):
        source = self.a_embed(a_digits) + self.b_embed(b_digits)
        if previous.shape[1]:
            x = torch.cat((source, self.out_embed(previous)), dim=1)
        else:
            x = source
        p = F.pad(self.position[:x.shape[1]], (0, D - 3))
        x = x + p
        x = self.attn1(x)
        x = x + self.ff2(F.gelu(self.ff1(self.ff_norm(x))))
        x = self.attn2(x)
        return self.head(self.final_norm(x))


def digits(values):
    p = POW10.to(values.device)
    d = values[:, None].div(p, rounding_mode='floor').remainder(10)
    d[:, 14] = 10
    return d


def output_digits(values):
    p = POW10.to(values.device)
    return values[:, None].div(p, rounding_mode='floor').remainder(10)


def random_batch(n, device, structured_fraction=0.0, boundary_fraction=0.0):
    a = torch.randint(MAX_VALUE, (n,), device=device)
    b = torch.randint(MAX_VALUE, (n,), device=device)
    ns = int(n * structured_fraction)
    if ns:
        idx = torch.arange(ns, device=device)
        kind = idx.remainder(6)
        pos = torch.randint(0, 14, (ns,), device=device)
        length = torch.randint(1, 15, (ns,), device=device)
        length = torch.minimum(length, 14 - pos)
        lo = torch.tensor([10 ** i for i in range(15)], device=device)[pos]
        hi = torch.tensor([10 ** i for i in range(15)], device=device)[pos + length]
        span = hi - lo

        # Exact shifted 9-runs plus a sparse increment, inducing carries of varied lengths.
        mask = kind == 0
        aa = hi - lo
        bb = lo
        a[:ns] = torch.where(mask, aa, a[:ns])
        b[:ns] = torch.where(mask, bb, b[:ns])

        # Near-carry matched contrasts: a run ending in 8 receives one unit.
        mask = kind == 1
        aa = hi - lo - lo
        bb = lo
        a[:ns] = torch.where(mask, aa.clamp_min(0), a[:ns])
        b[:ns] = torch.where(mask, bb, b[:ns])

        # Sparse equal digits, emphasizing 5+5 and 9+9 at every column.
        mask = kind == 2
        val = torch.where(idx.remainder(2) == 0, 5 * lo, 9 * lo)
        a[:ns] = torch.where(mask, val, a[:ns])
        b[:ns] = torch.where(mask, val, b[:ns])

        # Repeated full-length digits.
        mask = kind == 3
        rep_a = torch.randint(10, (ns,), device=device)
        rep_b = torch.randint(10, (ns,), device=device)
        ones = 11_111_111_111_111
        a[:ns] = torch.where(mask, rep_a * ones, a[:ns])
        b[:ns] = torch.where(mask, rep_b * ones, b[:ns])

        # Complementary low blocks, with random higher context.
        mask = kind == 4
        low = torch.randint(1, MAX_VALUE, (ns,), device=device)
        power_pos = torch.randint(1, 15, (ns,), device=device)
        power = torch.tensor([10 ** i for i in range(15)], device=device)[power_pos]
        low = low.remainder(power).clamp_min(1)
        aa = low
        bb = power - low
        a[:ns] = torch.where(mask, aa, a[:ns])
        b[:ns] = torch.where(mask, bb, b[:ns])

        # One sparse operand against a random full operand.
        mask = kind == 5
        sparse_digit = torch.randint(1, 10, (ns,), device=device) * lo
        a[:ns] = torch.where(mask, sparse_digit, a[:ns])

    nb = int(n * boundary_fraction)
    if nb:
        pos = torch.randint(0, 14, (nb,), device=device)
        p = torch.tensor([10 ** i for i in range(15)], device=device)[pos]
        choice = torch.randint(4, (nb,), device=device)
        x = torch.where(choice < 2, 5 * p, 9 * p)
        y = torch.where(choice.remainder(2) == 0, x, p)
        a[-nb:] = x
        b[-nb:] = y
    return a, b


def loss_for(model, a, b):
    ad = digits(a)
    bd = digits(b)
    target = output_digits(a + b)
    previous = target[:, :-1]
    logits = model(ad, bd, previous)[:, 14:]
    return F.cross_entropy(logits.reshape(-1, 10), target.reshape(-1))


@torch.no_grad()
def decode(model, a, b):
    ad = digits(a)
    bd = digits(b)
    previous = torch.empty((a.shape[0], 0), dtype=torch.long, device=a.device)
    for _ in range(15):
        token = model(ad, bd, previous)[:, -1].argmax(-1, keepdim=True)
        previous = torch.cat((previous, token), dim=1)
    return previous


@torch.no_grad()
def evaluate(model, count, batch, structured=0.0, boundary=0.0):
    model.eval()
    errors = 0
    device = next(model.parameters()).device
    for start in range(0, count, batch):
        n = min(batch, count - start)
        a, b = random_batch(n, device, structured, boundary)
        errors += (decode(model, a, b) != output_digits(a + b)).any(1).sum().item()
    model.train()
    return errors


def export_submission(model, path):
    state = {k: v.detach().float().cpu().reshape(-1).tolist() for k, v in model.state_dict().items()}
    shapes = {k: list(v.shape) for k, v in model.state_dict().items()}
    template = '''import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(9)
        self.qkv = nn.Linear(9, 27)
        self.proj = nn.Linear(9, 9)

    def forward(self, x):
        z = self.norm(x)
        q, k, v = self.qkv(z).chunk(3, dim=-1)
        batch, length, _ = q.shape
        q = q.view(batch, length, 3, 3).transpose(1, 2)
        k = k.view(batch, length, 3, 3).transpose(1, 2)
        v = v.view(batch, length, 3, 3).transpose(1, 2)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return x + self.proj(attended.transpose(1, 2).reshape(batch, length, 9))


class AddTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.a_embed = nn.Embedding(11, 9)
        self.b_embed = nn.Embedding(11, 9)
        self.out_embed = nn.Embedding(10, 9)
        self.position = nn.Parameter(torch.empty(29, 3))
        self.attn1 = Attention()
        self.ff_norm = nn.LayerNorm(9)
        self.ff1 = nn.Linear(9, 1)
        self.ff2 = nn.Linear(1, 9)
        self.attn2 = Attention()
        self.final_norm = nn.LayerNorm(9)
        self.head = nn.Linear(9, 10)

    def forward(self, a_digits, b_digits, previous):
        source = self.a_embed(a_digits) + self.b_embed(b_digits)
        if previous.shape[1]:
            x = torch.cat((source, self.out_embed(previous)), dim=1)
        else:
            x = source
        x = x + F.pad(self.position[:x.shape[1]], (0, 6))
        x = self.attn1(x)
        x = x + self.ff2(F.gelu(self.ff1(self.ff_norm(x))))
        x = self.attn2(x)
        return self.head(self.final_norm(x))


_WEIGHTS = __STATE__
_SHAPES = __SHAPES__


def build_model():
    model = AddTransformer()
    values = {name: torch.tensor(data, dtype=torch.float32).reshape(_SHAPES[name]) for name, data in _WEIGHTS.items()}
    model.load_state_dict(values)
    model.eval()
    return model, {"architecture": "aligned causal autoregressive transformer", "parameters": 1295}


def _operand_digits(value):
    result = []
    for _ in range(14):
        result.append(value % 10)
        value //= 10
    result.append(10)
    return result


@torch.no_grad()
def add(model, a: int, b: int) -> int:
    device = next(model.parameters()).device
    ad = torch.tensor([_operand_digits(a)], dtype=torch.long, device=device)
    bd = torch.tensor([_operand_digits(b)], dtype=torch.long, device=device)
    previous = torch.empty((1, 0), dtype=torch.long, device=device)
    for _ in range(15):
        token = model(ad, bd, previous)[:, -1].argmax(dim=-1, keepdim=True)
        previous = torch.cat((previous, token), dim=1)
    value = 0
    place = 1
    for token in previous[0]:
        value += int(token) * place
        place *= 10
    return value
'''
    text = template.replace('__STATE__', repr(state)).replace('__SHAPES__', repr(shapes))
    Path(path).write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=72000)
    parser.add_argument('--batch', type=int, default=4096)
    parser.add_argument('--resume', type=str)
    args = parser.parse_args()
    torch.manual_seed(41)
    random.seed(41)
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda')
    model = AddTransformer().to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True))
    print('parameters', sum(p.numel() for p in model.parameters()), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.002, fused=True)
    milestones = [(8000, 3e-3, 0.0, 0.0), (22000, 1e-3, 0.25, 0.0),
                  (42000, 3e-4, 0.40, 0.05), (58000, 1e-4, 0.45, 0.10),
                  (args.steps, 3e-5, 0.45, 0.12)]
    start = 0
    for end, lr, structured, boundary in milestones:
        if end <= start:
            continue
        for group in optimizer.param_groups:
            group['lr'] = lr
        for step in range(start, end):
            a, b = random_batch(args.batch, device, structured, boundary)
            loss = loss_for(model, a, b)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if (step + 1) % 1000 == 0:
                print(step + 1, float(loss), lr, structured, boundary, flush=True)
            if (step + 1) % 10000 == 0:
                raw = model
                torch.save(raw.state_dict(), f'/workspace/checkpoint_{step + 1}.pt')
                export_submission(raw, '/workspace/submission.py')
        start = end
    raw = model
    torch.save(raw.state_dict(), '/workspace/model.pt')
    export_submission(raw, '/workspace/submission.py')
    for label, s, bd in [('uniform', 0.0, 0.0), ('mixed', 0.45, 0.1), ('structured', 1.0, 0.1)]:
        print(label, evaluate(raw, 131072, 4096, s, bd), flush=True)


if __name__ == '__main__':
    main()


def refine(steps=24000, batch=4096):
    torch.manual_seed(4101)
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda')
    model = AddTransformer().to(device)
    model.load_state_dict(torch.load('/workspace/model.pt', map_location=device, weights_only=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.001, fused=True)
    model.train()
    for step in range(steps):
        a, b = random_batch(batch, device, 0.45, 0.12)
        loss = loss_for(model, a, b)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % 1000 == 0:
            print('refine', step + 1, float(loss), flush=True)
    torch.save(model.state_dict(), '/workspace/model_refined.pt')
    export_submission(model, '/workspace/submission.py')
    for label, s, bd in [('uniform', 0.0, 0.0), ('mixed', 0.45, 0.1), ('structured', 1.0, 0.1)]:
        print(label, evaluate(model, 262144, 4096, s, bd), flush=True)

