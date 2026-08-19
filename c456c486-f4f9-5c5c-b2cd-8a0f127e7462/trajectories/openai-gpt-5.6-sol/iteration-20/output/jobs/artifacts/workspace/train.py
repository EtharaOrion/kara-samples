import math
import random
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

DIGITS = 15
LIMIT = 100_000_000_000_000
D_MODEL = 10
HEADS = 2
D_FF = 12
LAYERS = 6


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.LayerNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.proj = nn.Linear(D_MODEL, D_MODEL)
        self.n2 = nn.LayerNorm(D_MODEL)
        self.fc1 = nn.Linear(D_MODEL, D_FF)
        self.fc2 = nn.Linear(D_FF, D_MODEL)

    def forward(self, x, mask):
        z = self.n1(x)
        q, k, v = self.qkv(z).chunk(3, -1)
        shape = (x.shape[0], DIGITS, HEADS, D_MODEL // HEADS)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).reshape_as(x)
        x = x + self.proj(y)
        return x + self.fc2(F.gelu(self.fc1(self.n2(x))))


class Adder(nn.Module):
    def __init__(self):
        super().__init__()
        self.a_embed = nn.Embedding(10, D_MODEL)
        self.b_embed = nn.Embedding(10, D_MODEL)
        self.pos = nn.Parameter(torch.empty(DIGITS, D_MODEL))
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.final = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, 10)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, a, b):
        x = self.a_embed(a) + self.b_embed(b) + self.pos
        mask = torch.ones(DIGITS, DIGITS, dtype=torch.bool, device=x.device).tril()
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.final(x))


def to_digits(x):
    places = torch.tensor([10**i for i in range(DIGITS)], device=x.device)
    return (x[:, None] // places % 10).long()


def make_batch(n, device, structured=0.25):
    a = torch.randint(0, LIMIT, (n,), device=device)
    b = torch.randint(0, LIMIT, (n,), device=device)
    m = int(n * structured)
    if not m:
        return to_digits(a), to_digits(b), to_digits(a + b)

    # Regenerated full-pair augmentations; labels always come from complete additions.
    q = m // 6
    if q:
        # Sparse increments create carry chains beginning at varied columns.
        starts = torch.randint(0, 14, (q,), device=device)
        lengths = torch.randint(1, 15, (q,), device=device)
        lengths = torch.minimum(lengths, 14 - starts)
        p0 = torch.pow(torch.tensor(10, device=device, dtype=torch.int64), starts)
        plen = torch.pow(torch.tensor(10, device=device, dtype=torch.int64), lengths)
        prefix = torch.randint(0, 10_000_000, (q,), device=device) * p0 * plen
        low = torch.randint(0, 10_000_000, (q,), device=device) % p0
        a[:q] = (prefix + (plen - 1) * p0 + low) % LIMIT
        b[:q] = p0

        # Near-complements and maximum-boundary overflows.
        r = slice(q, 2*q)
        x = torch.randint(0, LIMIT, (q,), device=device)
        a[r] = x
        b[r] = (LIMIT - 1 - x + torch.randint(0, 10, (q,), device=device)).clamp(0, LIMIT-1)

        # Repeated digits.
        r = slice(2*q, 3*q)
        da = torch.randint(0, 10, (q,), device=device)
        db = torch.randint(0, 10, (q,), device=device)
        rep = 11_111_111_111_111
        a[r] = da * rep
        b[r] = db * rep

        # Small and sparse operands at arbitrary decimal places.
        r = slice(3*q, 4*q)
        p = torch.pow(torch.tensor(10, device=device, dtype=torch.int64), torch.randint(0, 14, (q,), device=device))
        a[r] = torch.randint(0, LIMIT, (q,), device=device)
        b[r] = torch.randint(0, 10, (q,), device=device) * p

        # Long runs ending at the top position.
        r = slice(4*q, 5*q)
        k = torch.randint(1, 15, (q,), device=device)
        p = torch.pow(torch.tensor(10, device=device, dtype=torch.int64), k)
        a[r] = (p - 1).clamp_max(LIMIT - 1)
        b[r] = 1

        # Operand-swap duplicates enforce commutative representations.
        r = slice(5*q, 6*q)
        a[r] = b[:q]
        b[r] = a[:q]
    return to_digits(a), to_digits(b), to_digits(a + b)


@torch.no_grad()
def evaluate(model, batches, batch_size, structured):
    model.eval()
    errors = 0
    total = 0
    for _ in range(batches):
        a, b, y = make_batch(batch_size, a_device, structured)
        pred = model(a, b).argmax(-1)
        errors += (pred != y).any(1).sum().item()
        total += batch_size
    model.train()
    return errors, total


def export(model, path):
    state = {k: v.detach().float().cpu().tolist() for k, v in model.state_dict().items()}
    source = '''import torch
from torch import nn
import torch.nn.functional as F

DIGITS = 15
D_MODEL = 10
HEADS = 2
D_FF = 12

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.LayerNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.proj = nn.Linear(D_MODEL, D_MODEL)
        self.n2 = nn.LayerNorm(D_MODEL)
        self.fc1 = nn.Linear(D_MODEL, D_FF)
        self.fc2 = nn.Linear(D_FF, D_MODEL)

    def forward(self, x, mask):
        z = self.n1(x)
        q, k, v = self.qkv(z).chunk(3, -1)
        shape = (x.shape[0], DIGITS, HEADS, D_MODEL // HEADS)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.proj(y.transpose(1, 2).reshape_as(x))
        return x + self.fc2(F.gelu(self.fc1(self.n2(x))))

class Adder(nn.Module):
    def __init__(self):
        super().__init__()
        self.a_embed = nn.Embedding(10, D_MODEL)
        self.b_embed = nn.Embedding(10, D_MODEL)
        self.pos = nn.Parameter(torch.empty(DIGITS, D_MODEL))
        self.blocks = nn.ModuleList([Block() for _ in range(6)])
        self.final = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, 10)

    def forward(self, a, b):
        x = self.a_embed(a) + self.b_embed(b) + self.pos
        mask = torch.ones(DIGITS, DIGITS, dtype=torch.bool, device=x.device).tril()
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.final(x))

_STATE = STATE_LITERAL

def build_model():
    model = Adder()
    model.load_state_dict({k: torch.tensor(v) for k, v in _STATE.items()})
    model.eval()
    return model, {"architecture": "six distinct causal transformer blocks", "digits": 14}

def _tokens(value):
    result = []
    for _ in range(DIGITS):
        value, digit = divmod(value, 10)
        result.append(digit)
    return result

def add(model, a: int, b: int) -> int:
    device = next(model.parameters()).device
    ta = torch.tensor([_tokens(a)], dtype=torch.long, device=device)
    tb = torch.tensor([_tokens(b)], dtype=torch.long, device=device)
    with torch.no_grad():
        digits = model(ta, tb)[0].argmax(-1).tolist()
    value = 0
    for digit in reversed(digits):
        value = value * 10 + digit
    return value
'''.replace('STATE_LITERAL', repr(state))
    path.write_text(source)


if __name__ == '__main__':
    torch.manual_seed(20250814)
    torch.set_float32_matmul_precision('high')
    a_device = torch.device('cuda')
    model = Adder().to(a_device)
    print('parameters', sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.003)
    best = 10**9
    steps = 24000
    batch_size = 8192
    for step in range(1, steps + 1):
        if step == 10000:
            for g in opt.param_groups: g['lr'] = 1e-3
        if step == 17000:
            for g in opt.param_groups: g['lr'] = 3e-4
        if step == 22000:
            for g in opt.param_groups: g['lr'] = 1e-4
        frac = 0.20 if step < 10000 else 0.35
        x, z, y = make_batch(batch_size, a_device, frac)
        loss = F.cross_entropy(model(x, z).reshape(-1, 10), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0 or step == 1:
            e, n = evaluate(model, 8, 4096, 0.0)
            es, ns = evaluate(model, 4, 4096, 0.75)
            print(step, float(loss), 'random', e, '/', n, 'structured', es, '/', ns, flush=True)
            if e + es <= best:
                best = e + es
                torch.save(model.state_dict(), '/workspace/best.pt')
                export(model, Path('/workspace/submission.py'))
    model.load_state_dict(torch.load('/workspace/best.pt', weights_only=True))
    export(model, Path('/workspace/submission.py'))
