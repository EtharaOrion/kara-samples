import argparse
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


class TinyAdder(nn.Module):
    def __init__(self, ff=20):
        super().__init__()
        d = 20
        self.token = nn.Embedding(11, d)
        self.position = nn.Parameter(torch.zeros(25, d))
        self.pass_embedding = nn.Parameter(torch.zeros(2, d))
        self.norm1 = nn.LayerNorm(d)
        self.attention = nn.MultiheadAttention(d, 4, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, ff)
        self.ff2 = nn.Linear(ff, d)
        self.final_norm = nn.LayerNorm(d)
        self.output = nn.Linear(d, 10, bias=False)
        self.register_buffer("causal", torch.triu(torch.ones(25, 25, dtype=torch.bool), 1))

    def forward(self, tokens):
        n = tokens.shape[1]
        x = self.token(tokens) + self.position[:n]
        mask = self.causal[:n, :n]
        for p in range(2):
            y = self.norm1(x + self.pass_embedding[p])
            x = x + self.attention(y, y, y, attn_mask=mask, need_weights=False)[0]
            x = x + self.ff2(F.gelu(self.ff1(self.norm2(x))))
        return self.output(self.final_norm(x))


def digits_of(x):
    places = torch.tensor([1,10,100,1000,10000,100000,1000000,10000000], device=x.device)
    return (x[:, None] // places % 10).long()


def batch_data(n, device, structured=0.25):
    a = torch.randint(10_000_000, 100_000_000, (n,), device=device)
    b = torch.randint(10_000_000, 100_000_000, (n,), device=device)
    m = int(n * structured)
    if m:
        q = m // 4
        # Exact 100,000,000 complements exercise long carries.
        aa = torch.randint(10_000_000, 90_000_001, (q,), device=device)
        a[:q], b[:q] = aa, 100_000_000 - aa
        # Near-maximum operands and repeated nines.
        a[q:2*q] = 99_999_999 - torch.randint(0, 100_000, (q,), device=device)
        b[q:2*q] = 99_999_999 - torch.randint(0, 100_000, (q,), device=device)
        # Round numbers with independently chosen trailing-zero counts.
        r = q
        raw_a = torch.randint(10_000_000, 100_000_000, (r,), device=device)
        raw_b = torch.randint(10_000_000, 100_000_000, (r,), device=device)
        powers = torch.tensor([10,100,1000,10000,100000,1000000,10000000], device=device)
        pa = powers[torch.randint(0, 7, (r,), device=device)]
        pb = powers[torch.randint(0, 7, (r,), device=device)]
        a[2*q:3*q] = (raw_a // pa) * pa
        b[2*q:3*q] = (raw_b // pb) * pb
        # Repeated-digit and sparse-looking patterns.
        r = m - 3*q
        da = torch.randint(1, 10, (r,), device=device)
        db = torch.randint(1, 10, (r,), device=device)
        a[3*q:m] = da * 11_111_111
        b[3*q:m] = db * 11_111_111
    ad, bd = digits_of(a), digits_of(b)
    s = a + b
    places9 = torch.tensor([1,10,100,1000,10000,100000,1000000,10000000,100000000], device=device)
    target = (s[:, None] // places9 % 10).long()
    tokens = torch.empty(n, 25, dtype=torch.long, device=device)
    tokens[:, 0:16:2], tokens[:, 1:16:2] = ad, bd
    tokens[:, 16] = 10
    tokens[:, 17:] = target[:, :8]
    return tokens, target


@torch.no_grad()
def evaluate(model, n=10000, structured=0.0, batch=5000):
    model.eval()
    good = total = 0
    for _ in range((n + batch - 1)//batch):
        k = min(batch, n-total)
        x, y = batch_data(k, next(model.parameters()).device, structured)
        pred = model(x)[:, 16:25].argmax(-1)
        good += pred.eq(y).all(1).sum().item()
        total += k
    model.train()
    return good / total


def export(model, ff, path):
    state = {k: v.detach().cpu() for k,v in model.state_dict().items() if k != "causal"}
    assignments = []
    for k,v in state.items():
        assignments.append(f"        {k!r}: torch.tensor({v.tolist()!r}, dtype=torch.{str(v.dtype).split('.')[-1]}),")
    source = '''import math\nimport torch\nfrom torch import nn\nimport torch.nn.functional as F\n\n\nclass TinyAdder(nn.Module):\n    def __init__(self):\n        super().__init__()\n        d = 20\n        self.token = nn.Embedding(11, d)\n        self.position = nn.Parameter(torch.zeros(25, d))\n        self.pass_embedding = nn.Parameter(torch.zeros(2, d))\n        self.norm1 = nn.LayerNorm(d)\n        self.attention = nn.MultiheadAttention(d, 4, batch_first=True)\n        self.norm2 = nn.LayerNorm(d)\n        self.ff1 = nn.Linear(d, FF)\n        self.ff2 = nn.Linear(FF, d)\n        self.final_norm = nn.LayerNorm(d)\n        self.output = nn.Linear(d, 10, bias=False)\n        self.register_buffer("causal", torch.triu(torch.ones(25, 25, dtype=torch.bool), 1))\n\n    def forward(self, tokens):\n        n = tokens.shape[1]\n        x = self.token(tokens) + self.position[:n]\n        mask = self.causal[:n, :n]\n        for p in range(2):\n            y = self.norm1(x + self.pass_embedding[p])\n            x = x + self.attention(y, y, y, attn_mask=mask, need_weights=False)[0]\n            x = x + self.ff2(F.gelu(self.ff1(self.norm2(x))))\n        return self.output(self.final_norm(x))\n\n\ndef build_model():\n    model = TinyAdder()\n    state = {\nSTATE\n    }\n    model.load_state_dict(state, strict=False)\n    model.eval()\n    return model, {"architecture": "shared two-pass causal transformer", "parameters": sum(p.numel() for p in model.parameters())}\n\n\n@torch.inference_mode()\ndef add(model, a: int, b: int) -> int:\n    left = [ord(c) - 48 for c in f"{a:08d}"[::-1]]\n    right = [ord(c) - 48 for c in f"{b:08d}"[::-1]]\n    prefix = []\n    for x, y in zip(left, right):\n        prefix.extend((x, y))\n    prefix.append(10)\n    tokens = torch.tensor([prefix], dtype=torch.long, device=next(model.parameters()).device)\n    result = []\n    for _ in range(9):\n        digit = int(model(tokens)[0, -1].argmax())\n        result.append(digit)\n        tokens = torch.cat((tokens, torch.tensor([[digit]], device=tokens.device)), dim=1)\n    return int("".join(str(x) for x in reversed(result)))\n'''.replace('FF', str(ff)).replace('STATE', '\n'.join(assignments))
    path.write_text(source)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ff', type=int, default=20)
    ap.add_argument('--steps', type=int, default=22000)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--seed', type=int, default=5)
    ap.add_argument('--out', default='/workspace/submission.py')
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda')
    raw = TinyAdder(args.ff).to(device)
    model = raw
    opt = torch.optim.AdamW(raw.parameters(), lr=2e-3, weight_decay=0.01)
    best = 0.0
    for step in range(1, args.steps + 1):
        if step == 14001:
            for g in opt.param_groups: g['lr'] = 2e-4
        if step == 20001:
            for g in opt.param_groups: g['lr'] = 2e-5
        structured = 0.25 if step <= 14000 else (0.5 if step <= 20000 else 0.75)
        x,y = batch_data(args.batch, device, structured)
        logits = model(x)[:,16:25]
        loss = F.cross_entropy(logits.reshape(-1,10), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)
        opt.step()
        if step % 500 == 0 or step == args.steps:
            random_acc = evaluate(raw, 10000, 0.0)
            edge_acc = evaluate(raw, 10000, 0.8)
            print(f'{step} loss={loss.item():.5f} random={random_acc:.4f} edge={edge_acc:.4f}', flush=True)
            score = min(random_acc, edge_acc)
            if score >= best:
                best = score
                export(raw, args.ff, Path(args.out))
                torch.save(raw.state_dict(), f'/workspace/model_ff{args.ff}.pt')
    print('parameters', sum(p.numel() for p in raw.parameters()), 'best', best)

if __name__ == '__main__':
    main()
