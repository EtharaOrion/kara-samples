import argparse
import importlib.util
import pathlib
import torch
from torch import nn
import torch.nn.functional as F

DEVICE = 'cuda'
LO, HI = 10_000_000, 99_999_999


class Block(nn.Module):
    def __init__(self, ff=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(20)
        self.attn = nn.MultiheadAttention(20, 4, batch_first=True)
        self.norm2 = nn.LayerNorm(20)
        self.ff = nn.Sequential(nn.Linear(20, ff), nn.GELU(), nn.Linear(ff, 20))

    def forward(self, x, mask):
        z = self.norm1(x)
        x = x + self.attn(z, z, z, attn_mask=mask, need_weights=False)[0]
        return x + self.ff(self.norm2(x))


class Model(nn.Module):
    def __init__(self, rank=8, ff=4):
        super().__init__()
        self.rank = rank
        self.token = nn.Embedding(11, 20)
        self.position_left = nn.Parameter(torch.empty(25, rank))
        self.position_right = nn.Parameter(torch.empty(rank, 20))
        self.passes = nn.Embedding(2, 20)
        self.block = Block(ff)
        self.norm = nn.LayerNorm(20)
        self.output = nn.Linear(20, 10, bias=False)
        nn.init.normal_(self.position_left, std=.2)
        nn.init.normal_(self.position_right, std=.2)

    def forward(self, tokens):
        n = tokens.shape[1]
        x = self.token(tokens) + self.position_left[:n] @ self.position_right
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=tokens.device), 1)
        for step in range(2):
            x = self.block(x + self.passes.weight[step], mask)
        return self.output(self.norm(x))


def digits(x, count):
    out = []
    for _ in range(count):
        out.append(x % 10)
        x = x // 10
    return torch.stack(out, 1)


def batch(n, structured=.45):
    a = torch.randint(LO, HI + 1, (n,), device=DEVICE)
    b = torch.randint(LO, HI + 1, (n,), device=DEVICE)
    k = int(n * structured)
    if k:
        group = torch.arange(k, device=DEVICE) % 5
        # Exact complements exercise eight-column carry disappearance.
        ix = torch.where(group == 0)[0]
        aa = torch.randint(LO, 90_000_001, (len(ix),), device=DEVICE)
        a[ix], b[ix] = aa, 100_000_000 - aa
        # Large operands and long carry chains.
        ix = torch.where(group == 1)[0]
        a[ix] = torch.randint(90_000_000, HI + 1, (len(ix),), device=DEVICE)
        b[ix] = torch.randint(90_000_000, HI + 1, (len(ix),), device=DEVICE)
        # Round decimal boundaries at varying powers.
        ix = torch.where(group == 2)[0]
        pows = 10 ** torch.randint(1, 7, (len(ix),), device=DEVICE)
        a[ix] = ((torch.randint(LO, HI + 1, (len(ix),), device=DEVICE) // pows) * pows).clamp(min=LO)
        b[ix] = ((torch.randint(LO, HI + 1, (len(ix),), device=DEVICE) // pows) * pows).clamp(min=LO)
        # Repeated digits.
        ix = torch.where(group == 3)[0]
        a[ix] = 11_111_111 * torch.randint(1, 10, (len(ix),), device=DEVICE)
        b[ix] = 11_111_111 * torch.randint(1, 10, (len(ix),), device=DEVICE)
        # Nine-heavy patterns mixed with random counterpart digits.
        ix = torch.where(group == 4)[0]
        tail = torch.randint(1, 8, (len(ix),), device=DEVICE)
        scale = 10 ** tail
        a[ix] = (torch.randint(1, 10, (len(ix),), device=DEVICE) * scale - 1).clamp(LO, HI)
        b[ix] = torch.randint(LO, HI + 1, (len(ix),), device=DEVICE)
    ad, bd = digits(a, 8), digits(b, 8)
    operands = torch.stack((ad, bd), 2).reshape(n, 16)
    result = digits(a + b, 9)
    inp = torch.cat((operands, torch.full((n, 1), 10, device=DEVICE), result[:, :-1]), 1)
    return inp, result, a, b


@torch.no_grad()
def validate(model, n=100000, structured=.0, chunk=10000):
    model.eval()
    correct = total = 0
    digit_correct = 0
    for _ in range((n + chunk - 1) // chunk):
        bs = min(chunk, n-total)
        inp, target, _, _ = batch(bs, structured)
        tok = inp[:, :17]
        pred = []
        for j in range(9):
            d = model(tok)[:, -1].argmax(1)
            pred.append(d)
            if j < 8: tok = torch.cat((tok, d[:, None]), 1)
        pred = torch.stack(pred, 1)
        correct += (pred == target).all(1).sum().item()
        digit_correct += (pred == target).sum().item()
        total += bs
    model.train()
    return correct / total, digit_correct / (total * 9)


def export(model, rank):
    template = '''import torch
from torch import nn


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(20)
        self.attn = nn.MultiheadAttention(20, 4, batch_first=True)
        self.norm2 = nn.LayerNorm(20)
        self.ff = nn.Sequential(nn.Linear(20, 4), nn.GELU(), nn.Linear(4, 20))

    def forward(self, x, mask):
        z = self.norm1(x)
        x = x + self.attn(z, z, z, attn_mask=mask, need_weights=False)[0]
        return x + self.ff(self.norm2(x))


class AdditionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token = nn.Embedding(11, 20)
        self.position_left = nn.Parameter(torch.empty(25, __RANK__))
        self.position_right = nn.Parameter(torch.empty(__RANK__, 20))
        self.passes = nn.Embedding(2, 20)
        self.block = Block()
        self.norm = nn.LayerNorm(20)
        self.output = nn.Linear(20, 10, bias=False)

    def forward(self, tokens):
        n = tokens.shape[1]
        x = self.token(tokens) + self.position_left[:n] @ self.position_right
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=tokens.device), 1)
        for step in range(2):
            x = self.block(x + self.passes.weight[step], mask)
        return self.output(self.norm(x))


def build_model():
    model = AdditionTransformer()
    model.load_state_dict(_TRAINED_STATE)
    model.eval()
    return model, {'architecture': 'low-rank-position recurrent causal transformer', 'digit_order': 'least-significant-first'}


def add(model, a: int, b: int) -> int:
    ad = [ord(c) - 48 for c in str(a)[::-1]]
    bd = [ord(c) - 48 for c in str(b)[::-1]]
    prefix = [v for pair in zip(ad, bd) for v in pair] + [10]
    tokens = torch.tensor([prefix], dtype=torch.long, device=next(model.parameters()).device)
    digits = []
    with torch.no_grad():
        for _ in range(9):
            digit = int(model(tokens)[0, -1].argmax())
            digits.append(digit)
            tokens = torch.cat((tokens, torch.tensor([[digit]], device=tokens.device)), dim=1)
    return int(''.join(str(d) for d in digits[::-1]))


_TRAINED_STATE = __WEIGHTS__
'''
    state_parts = []
    for key, value in model.state_dict().items():
        data = value.detach().cpu().tolist()
        state_parts.append(repr(key) + ': torch.tensor(' + repr(data) + ')')
    text = template.replace('__RANK__', str(rank)).replace('__WEIGHTS__', '{\n' + ',\n'.join(state_parts) + '\n}')
    pathlib.Path('/workspace/submission.py').write_text(text)
    torch.save(model.state_dict(), f'/workspace/rank{rank}.pt')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rank', type=int, default=8)
    ap.add_argument('--steps', type=int, default=26000)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--seed', type=int, default=81)
    ap.add_argument('--resume')
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision('high')
    model = Model(args.rank).to(DEVICE)
    if args.resume: model.load_state_dict(torch.load(args.resume, weights_only=True))
    print('parameters', sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=.01)
    for step in range(1, args.steps + 1):
        # Increase hard examples late while retaining uniform coverage.
        structured = .45 if step < args.steps * .8 else .6
        inp, target, _, _ = batch(args.batch, structured)
        logits = model(inp)[:, 16:25]
        loss = F.cross_entropy(logits.reshape(-1, 10), target.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step == int(args.steps * .8):
            for g in opt.param_groups: g['lr'] = 1e-4
        if step == int(args.steps * .92):
            for g in opt.param_groups: g['lr'] = 2e-5
        if step % 1000 == 0:
            acc, da = validate(model, 10000, .5)
            print(step, float(loss), acc, da, flush=True)
    r = validate(model, 200000, 0)
    s = validate(model, 200000, .75)
    print('FINAL random', r, 'structured', s, flush=True)
    export(model, args.rank)


if __name__ == '__main__':
    main()
