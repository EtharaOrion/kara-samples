import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

N = 15
DEVICE = "cuda"


class Block(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.ln2 = nn.LayerNorm(width)
        self.ff1 = nn.Linear(width, hidden)
        self.ff2 = nn.Linear(hidden, width)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, -1)
        q = q.view(b, n, 2, d // 2).transpose(1, 2)
        k = k.view(b, n, 2, d // 2).transpose(1, 2)
        v = v.view(b, n, 2, d // 2).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(b, n, d))
        return x + self.ff2(F.gelu(self.ff1(self.ln2(x))))


class Adder(nn.Module):
    def __init__(self, width=10, hidden=10):
        super().__init__()
        self.a_embed = nn.Embedding(10, width)
        self.b_embed = nn.Embedding(10, width)
        self.pos = nn.Parameter(torch.empty(N, width))
        self.blocks = nn.ModuleList([Block(width, hidden) for _ in range(6)])
        self.final = nn.LayerNorm(width)
        self.head = nn.Linear(width, 10)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, a, b):
        x = self.a_embed(a) + self.b_embed(b) + self.pos
        for block in self.blocks:
            x = block(x)
        return self.head(self.final(x))


def result_digits(a, b):
    carry = torch.zeros(a.shape[0], device=a.device, dtype=torch.long)
    out = torch.empty_like(a)
    for p in range(N):
        total = a[:, p] + b[:, p] + carry
        out[:, p] = total.remainder(10)
        carry = total.div(10, rounding_mode="floor")
    return out


def uniform_batch(batch):
    a = torch.randint(0, 10, (batch, N), device=DEVICE)
    b = torch.randint(0, 10, (batch, N), device=DEVICE)
    # A 14-digit operand has a fixed zero fifteenth column.
    a[:, -1] = 0
    b[:, -1] = 0
    return a, b, result_digits(a, b)


def structured_batch(batch):
    a = torch.randint(0, 10, (batch, N), device=DEVICE)
    b = torch.randint(0, 10, (batch, N), device=DEVICE)
    a[:, -1] = 0
    b[:, -1] = 0
    rows = torch.arange(batch, device=DEVICE)
    kind = torch.randint(0, 8, (batch,), device=DEVICE)

    # Repeated operands and repeated blocks.
    m = kind == 0
    a[m, :14] = a[m, :1]
    b[m, :14] = b[m, :1]
    m = kind == 1
    for p in range(14):
        a[m, p] = a[m, (p // 3) * 3]
        b[m, p] = b[m, (p // 3) * 3]

    # Sparse operands, including increments at arbitrary decimal positions.
    m = kind == 2
    b[m] = 0
    count = int(m.sum())
    if count:
        p = torch.randint(0, 14, (count,), device=DEVICE)
        b[m, p] = torch.randint(1, 10, (count,), device=DEVICE)

    # Complementary columns create both isolated and chained carries.
    m = kind == 3
    b[m, :14] = 9 - a[m, :14]
    starts = torch.randint(0, 14, (batch,), device=DEVICE)
    a[rows[m], starts[m]] = torch.clamp(a[rows[m], starts[m]] + 1, max=9)

    # Vectorized random runs. Positions are selected by start <= p < end.
    start = torch.randint(0, 14, (batch,), device=DEVICE)
    max_length = 14 - start
    length = (torch.rand(batch, device=DEVICE) * max_length).long() + 1
    end = start + length
    places = torch.arange(14, device=DEVICE)[None]
    run = (places >= start[:, None]) & (places < end[:, None])

    # Explicit carry chain: initiating >=10, following columns exactly 9.
    m = (kind == 4) | (kind == 5)
    chain = run & m[:, None]
    av = torch.randint(0, 10, (batch, 14), device=DEVICE)
    a[:, :14] = torch.where(chain, av, a[:, :14])
    b[:, :14] = torch.where(chain, 9 - av, b[:, :14])
    initiator = m & (start < 14)
    ir = rows[initiator]
    ip = start[initiator]
    ia = torch.randint(1, 10, (ir.numel(),), device=DEVICE)
    a[ir, ip] = ia
    b[ir, ip] = 10 - ia + torch.randint(0, 10, (ir.numel(),), device=DEVICE).remainder(ia)
    stop = m & (end < 14)
    sr = rows[stop]
    sp = end[stop]
    sa = torch.randint(0, 9, (sr.numel(),), device=DEVICE)
    a[sr, sp] = sa
    b[sr, sp] = torch.randint(0, 9, (sr.numel(),), device=DEVICE).remainder(9 - sa)

    # Near-carry contrast: long exact-9 runs with no incoming carry.
    m = kind == 6
    contrast = run & m[:, None]
    av = torch.randint(0, 10, (batch, 14), device=DEVICE)
    a[:, :14] = torch.where(contrast, av, a[:, :14])
    b[:, :14] = torch.where(contrast, 9 - av, b[:, :14])
    before = m & (start > 0)
    br = rows[before]
    bp = start[before] - 1
    a[br, bp] = 0
    b[br, bp] = torch.randint(0, 10, (br.numel(),), device=DEVICE)

    # Runs just below/at/above the boundary distinguish 8, 9 and 10.
    m = kind == 7
    boundary = run & m[:, None]
    target = torch.randint(8, 11, (batch, 1), device=DEVICE)
    low = (target - 9).clamp_min(0)
    av = low + (torch.rand(batch, 14, device=DEVICE) * (torch.minimum(target, torch.tensor(9, device=DEVICE)) - low + 1)).long()
    a[:, :14] = torch.where(boundary, av, a[:, :14])
    b[:, :14] = torch.where(boundary, target - av, b[:, :14])

    return a, b, result_digits(a, b)


def mixed_batch(batch, structured_fraction):
    a, b, y = uniform_batch(batch)
    n = int(batch * structured_fraction)
    if n:
        sa, sb, sy = structured_batch(n)
        a[:n], b[:n], y[:n] = sa, sb, sy
    order = torch.randperm(batch, device=DEVICE)
    return a[order], b[order], y[order]


@torch.inference_mode()
def evaluate(model, batches, batch=8192, structured=False):
    model.eval()
    errors = examples = digit_errors = 0
    for _ in range(batches):
        a, b, y = structured_batch(batch) if structured else uniform_batch(batch)
        pred = model(a, b).argmax(-1)
        bad_digits = pred.ne(y)
        errors += bad_digits.any(1).sum().item()
        digit_errors += bad_digits.sum().item()
        examples += batch
    model.train()
    return errors, digit_errors, examples


def export_submission(model, path):
    state = {k: v.detach().float().cpu().tolist() for k, v in model.state_dict().items()}
    width = model.pos.shape[1]
    hidden = model.blocks[0].ff1.out_features
    source = '''import torch\nfrom torch import nn\nimport torch.nn.functional as F\n\nN = 15\nWIDTH = %d\nHIDDEN = %d\nSTATE = %s\n\n\nclass Block(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.ln1 = nn.LayerNorm(WIDTH)\n        self.qkv = nn.Linear(WIDTH, 3 * WIDTH)\n        self.proj = nn.Linear(WIDTH, WIDTH)\n        self.ln2 = nn.LayerNorm(WIDTH)\n        self.ff1 = nn.Linear(WIDTH, HIDDEN)\n        self.ff2 = nn.Linear(HIDDEN, WIDTH)\n\n    def forward(self, x):\n        batch, length, width = x.shape\n        q, k, v = self.qkv(self.ln1(x)).chunk(3, -1)\n        q = q.view(batch, length, 2, width // 2).transpose(1, 2)\n        k = k.view(batch, length, 2, width // 2).transpose(1, 2)\n        v = v.view(batch, length, 2, width // 2).transpose(1, 2)\n        scores = torch.matmul(q, k.transpose(-2, -1)) / (width // 2) ** 0.5\n        mask = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)\n        scores = scores.masked_fill(mask, float("-inf"))\n        attended = torch.softmax(scores, -1).matmul(v)\n        x = x + self.proj(attended.transpose(1, 2).reshape(batch, length, width))\n        return x + self.ff2(F.gelu(self.ff1(self.ln2(x))))\n\n\nclass AdditionTransformer(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.a_embed = nn.Embedding(10, WIDTH)\n        self.b_embed = nn.Embedding(10, WIDTH)\n        self.pos = nn.Parameter(torch.empty(N, WIDTH))\n        self.blocks = nn.ModuleList([Block() for _ in range(6)])\n        self.final = nn.LayerNorm(WIDTH)\n        self.head = nn.Linear(WIDTH, 10)\n\n    def forward(self, a_digits, b_digits):\n        x = self.a_embed(a_digits) + self.b_embed(b_digits) + self.pos\n        for block in self.blocks:\n            x = block(x)\n        return self.head(self.final(x))\n\n\ndef build_model():\n    model = AdditionTransformer()\n    tensors = {name: torch.tensor(value) for name, value in STATE.items()}\n    model.load_state_dict(tensors)\n    model.eval()\n    return model, {"architecture": "six-block parallel causal transformer", "digits": 14}\n\n\ndef _digits(value):\n    text = str(value).zfill(15)\n    return [int(character) for character in reversed(text)]\n\n\ndef add(model, a: int, b: int) -> int:\n    a_digits = torch.tensor([_digits(a)], dtype=torch.long)\n    b_digits = torch.tensor([_digits(b)], dtype=torch.long)\n    with torch.inference_mode():\n        predicted = model(a_digits, b_digits).argmax(-1)[0].tolist()\n    return int("".join(str(digit) for digit in reversed(predicted)))\n''' % (width, hidden, repr(state))
    Path(path).write_text(source)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--resume")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    model = Adder(10, args.hidden).to(DEVICE)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, weights_only=True))
    params = sum(p.numel() for p in model.parameters())
    print("parameters", params, flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.005)
    best = 10**9
    checkpoint = Path(f"/workspace/model_h{args.hidden}.pt")
    for step in range(1, args.steps + 1):
        if step <= args.steps * 0.45:
            lr, fraction = 3e-3, 0.25
        elif step <= args.steps * 0.72:
            lr, fraction = 1e-3, 0.35
        elif step <= args.steps * 0.90:
            lr, fraction = 3e-4, 0.50
        else:
            lr, fraction = 1e-4, 0.50
        for group in optimizer.param_groups:
            group["lr"] = lr
        a, b, y = mixed_batch(args.batch, fraction)
        logits = model(a, b)
        loss = F.cross_entropy(logits.reshape(-1, 10), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 1000 == 0 or step == args.steps:
            ue, ud, un = evaluate(model, 8)
            se, sd, sn = evaluate(model, 4, structured=True)
            score = ue + 2 * se
            print(json.dumps({"step": step, "loss": loss.item(), "uniform": [ue, un], "structured": [se, sn], "digit_errors": [ud, sd]}), flush=True)
            if score <= best:
                best = score
                torch.save(model.state_dict(), checkpoint)
                export_submission(model, "/workspace/submission.py")
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    export_submission(model, "/workspace/submission.py")
    print("saved", checkpoint, "best", best, flush=True)


if __name__ == "__main__":
    main()
