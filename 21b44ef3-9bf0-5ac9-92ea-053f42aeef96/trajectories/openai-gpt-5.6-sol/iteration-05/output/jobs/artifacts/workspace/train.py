import argparse
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F


class AdditionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        d = 20
        self.token = nn.Embedding(11, d)
        self.position = nn.Embedding(25, d)
        self.pass_embedding = nn.Parameter(torch.empty(2, d))
        self.norm1 = nn.LayerNorm(d)
        self.attention = nn.MultiheadAttention(d, 4, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.feed_forward = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.final_norm = nn.LayerNorm(d)
        self.output = nn.Linear(d, 10, bias=False)
        self.register_buffer('causal_mask', torch.triu(torch.full((25, 25), float('-inf')), 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.token.weight, std=.08)
        nn.init.normal_(self.position.weight, std=.08)
        nn.init.normal_(self.pass_embedding, std=.08)

    def forward(self, tokens):
        n = tokens.shape[1]
        p = torch.arange(n, device=tokens.device)
        h = self.token(tokens) + self.position(p)
        mask = self.causal_mask[:n, :n]
        for r in range(2):
            h = h + self.pass_embedding[r]
            z = self.norm1(h)
            h = h + self.attention(z, z, z, attn_mask=mask, need_weights=False)[0]
            h = h + self.feed_forward(self.norm2(h))
        return self.output(self.final_norm(h))

POW10 = None

def random_batch(batch, device, structured=0.0):
    global POW10
    if POW10 is None: POW10 = (10 ** torch.arange(8, device=device, dtype=torch.long))
    a = torch.randint(10_000_000, 100_000_000, (batch,), device=device)
    b = torch.randint(10_000_000, 100_000_000, (batch,), device=device)
    if structured:
        n = int(batch * structured)
        kinds = torch.randint(0, 4, (n,), device=device)
        # Exact complementary pairs exercise the longest carry chains.
        x = torch.randint(10_000_000, 90_000_001, (n,), device=device)
        aa, bb = x, 100_000_000 - x
        # Multiples with long zero suffixes and repeated digits.
        places = 10 ** torch.randint(1, 7, (n,), device=device)
        rounded = (torch.randint(10_000_000, 100_000_000, (n,), device=device) // places) * places
        rounded = rounded.clamp(10_000_000, 99_999_999)
        aa = torch.where(kinds == 1, rounded, aa)
        repeated = torch.randint(1, 10, (n,), device=device) * 11_111_111
        aa = torch.where(kinds == 2, repeated, aa)
        bb = torch.where(kinds == 2, torch.randint(1, 10, (n,), device=device) * 11_111_111, bb)
        # Near limits.
        near_hi = 99_999_999 - torch.randint(0, 100_000, (n,), device=device)
        aa = torch.where(kinds == 3, near_hi, aa)
        bb = torch.where(kinds == 3, 10_000_000 + torch.randint(0, 100_000, (n,), device=device), bb)
        a[:n], b[:n] = aa, bb
    ad = (a[:, None] // POW10) % 10
    bd = (b[:, None] // POW10) % 10
    answer = a + b
    yd = (answer[:, None] // (10 ** torch.arange(9, device=device))) % 10
    inputs = torch.empty((batch, 25), dtype=torch.long, device=device)
    inputs[:, 0:16:2] = ad
    inputs[:, 1:16:2] = bd
    inputs[:, 16] = 10
    inputs[:, 17:] = yd[:, :-1]
    return inputs, yd

@torch.no_grad()
def evaluate(model, count=10000):
    model.eval(); correct = 0; digits = 0
    for _ in range((count + 1023)//1024):
        x, y = random_batch(min(1024, count-correct*0), 'cuda', .35)
        # Autoregressive exact-match evaluation.
        prefix = x[:, :17]
        out = []
        for j in range(9):
            pred = model(prefix)[:, -1].argmax(-1)
            out.append(pred)
            prefix = torch.cat((prefix, pred[:, None]), 1)
        pred = torch.stack(out, 1)
        correct += (pred == y).all(1).sum().item()
        digits += y.shape[0]
        if digits >= count: break
    model.train()
    return correct / digits


def export(model, path):
    state = {k: v.detach().cpu() for k, v in model.state_dict().items() if k != 'causal_mask'}
    header = '''import torch
from torch import nn


class AdditionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        width = 20
        self.token = nn.Embedding(11, width)
        self.position = nn.Embedding(25, width)
        self.pass_embedding = nn.Parameter(torch.empty(2, width))
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.final_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, 10, bias=False)
        self.register_buffer("causal_mask", torch.triu(torch.full((25, 25), float("-inf")), 1))

    def forward(self, tokens):
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device)
        hidden = self.token(tokens) + self.position(positions)
        mask = self.causal_mask[:length, :length]
        for pass_number in range(2):
            hidden = hidden + self.pass_embedding[pass_number]
            normalized = self.norm1(hidden)
            attended = self.attention(normalized, normalized, normalized, attn_mask=mask, need_weights=False)[0]
            hidden = hidden + attended
            hidden = hidden + self.feed_forward(self.norm2(hidden))
        return self.output(self.final_norm(hidden))

'''
    lines = [header, '_TRAINED_STATE = {\n']
    for key, value in state.items():
        data = repr(value.tolist())
        lines.append(f'    {key!r}: torch.tensor({data}),\n')
    lines.append('}\n\n')
    lines.append('''def build_model():
    model = AdditionTransformer()
    model.load_state_dict(_TRAINED_STATE, strict=False)
    return model, {"architecture": "weight-tied recurrent causal transformer", "width": 20, "heads": 4, "refinement_passes": 2, "output_order": "least-significant-first"}


def add(model, a: int, b: int) -> int:
    a_digits = [ord(c) - 48 for c in str(a)[::-1]]
    b_digits = [ord(c) - 48 for c in str(b)[::-1]]
    prefix = []
    for left, right in zip(a_digits, b_digits):
        prefix.extend((left, right))
    prefix.append(10)
    device = next(model.parameters()).device
    for _ in range(9):
        tokens = torch.tensor([prefix], dtype=torch.long, device=device)
        digit = int(model(tokens)[0, -1].argmax().item())
        prefix.append(digit)
    return int("".join(str(digit) for digit in prefix[17:][::-1]))
''')
    path.write_text(''.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--batch', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=5)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision('high')
    model = AdditionTransformer().cuda().train()
    print('parameters', sum(p.numel() for p in model.parameters()), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(.9, .98), weight_decay=.01)
    scaler = None
    for step in range(1, args.steps + 1):
        structured = .15 if step < 12000 else .35
        x, y = random_batch(args.batch, 'cuda', structured)
        lr = 2e-3 if step < 12000 else (3e-4 if step < 17000 else 7e-5)
        for group in optimizer.param_groups: group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)[:, 16:25]
            loss = F.cross_entropy(logits.reshape(-1, 10), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 500 == 0:
            with torch.no_grad():
                digit_acc = (logits.argmax(-1) == y).float().mean().item()
            print(step, f'loss={loss.item():.5f}', f'digit={digit_acc:.5f}', flush=True)
        if step % 4000 == 0:
            score = evaluate(model, 3000)
            print('autoregressive', score, flush=True)
            export(model, Path('/workspace/submission.py'))
    export(model, Path('/workspace/submission.py'))

if __name__ == '__main__': main()
