import argparse
import random
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

DIGITS = 15
SEQ = DIGITS * 3


class Block(nn.Module):
    def __init__(self, d=16, hidden=32, shared_mlp=None):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.mlp = shared_mlp if shared_mlp is not None else nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))

    def forward(self, x, mask):
        z = self.n1(x)
        q, k, v = self.qkv(z).chunk(3, -1)
        q = q.view(*q.shape[:-1], 2, 8).transpose(1, 2)
        k = k.view(*k.shape[:-1], 2, 8).transpose(1, 2)
        v = v.view(*v.shape[:-1], 2, 8).transpose(1, 2)
        att = F.softmax((q @ k.transpose(-2, -1)) * (8 ** -0.5) + mask, -1)
        z = (att @ v).transpose(1, 2).reshape_as(x)
        x = x + self.proj(z)
        return x + self.mlp(self.n2(x))


class Adder(nn.Module):
    def __init__(self, share_mlp=True):
        super().__init__()
        self.digit = nn.Embedding(10, 16)
        self.role = nn.Embedding(3, 16)
        shared = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 16)) if share_mlp else None
        self.blocks = nn.ModuleList([Block(shared_mlp=shared), Block(shared_mlp=shared)])
        self.norm = nn.LayerNorm(16)
        self.head = nn.Linear(16, 10)
        p = torch.arange(SEQ)
        allowed = (p[:, None] >= p[None, :]) & (p[:, None] - p[None, :] < 6)
        self.register_buffer('mask', torch.where(allowed, 0.0, float('-inf')).view(1, 1, SEQ, SEQ), persistent=False)

    def forward(self, tok):
        length = tok.shape[1]
        roles = torch.arange(length, device=tok.device) % 3
        x = self.digit(tok) + self.role(roles)
        mask = self.mask[:, :, :length, :length]
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.norm(x))


def digits(x):
    columns = []
    for _ in range(DIGITS):
        columns.append(x % 10)
        x = x // 10
    return torch.stack(columns, 1)


def batch(n, structured=0.0):
    limit = 100_000_000_000_000
    a = torch.randint(0, limit, (n,))
    b = torch.randint(0, limit, (n,))
    if structured and random.random() < structured:
        mode = random.randrange(3)
        if mode == 0:
            # Long carry chains generated as complete operand pairs.
            lengths = torch.randint(2, 15, (n,))
            powers = torch.tensor([10 ** int(v) for v in lengths], dtype=torch.long)
            a = powers - 1
            b = torch.randint(1, 10, (n,))
        elif mode == 1:
            da = torch.randint(0, 10, (n, 1))
            db = torch.randint(0, 10, (n, 1))
            weights = torch.tensor([10 ** i for i in range(14)]).view(1, 14)
            a = (da * weights).sum(1)
            b = (db * weights).sum(1)
        else:
            b = limit - 1 - a
    ad, bd, yd = digits(a), digits(b), digits(a + b)
    tok = torch.stack((ad, bd, yd), 2).reshape(n, SEQ)
    return tok


@torch.no_grad()
def exact_accuracy(model, n=10000, seed=12345, chunk=1000):
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    good = total = 0
    model.eval()
    for _ in range((n + chunk - 1) // chunk):
        size = min(chunk, n - total)
        full = batch(size)
        seq = torch.empty((size, 0), dtype=torch.long)
        predicted = []
        for col in range(DIGITS):
            seq = torch.cat((seq, full[:, col * 3:col * 3 + 2]), 1)
            out = model(seq)[:, -1].argmax(-1, keepdim=True)
            predicted.append(out)
            seq = torch.cat((seq, out), 1)
        pred = torch.cat(predicted, 1)
        true = full[:, 2::3]
        good += (pred == true).all(1).sum().item()
        total += size
    torch.random.set_rng_state(state)
    model.train()
    return good / total


def export(model, path):
    state = model.state_dict()
    # Shared modules occur under both block names in state_dict; retain one copy and alias during loading.
    vals = {k: v.detach().flatten().tolist() for k, v in state.items() if not ('.blocks.1.mlp.' in '.' + k)}
    shapes = {k: list(v.shape) for k, v in state.items() if k in vals}
    template = Path('/workspace/submission_template.py').read_text()
    payload = repr(vals)
    shape_payload = repr(shapes)
    Path(path).write_text(template.replace('__VALUES__', payload).replace('__SHAPES__', shape_payload))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--unshared', action='store_true')
    ap.add_argument('--steps', type=int, default=5000)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--resume')
    ap.add_argument('--out', default='/workspace/model.pt')
    args = ap.parse_args()
    torch.set_num_threads(min(16, torch.get_num_threads()))
    torch.manual_seed(20250813)
    model = Adder(not args.unshared)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, weights_only=True))
    print('parameters', sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=.005)
    milestones = {int(args.steps*.55): 1.5e-3, int(args.steps*.75): 7e-4, int(args.steps*.88): 3e-4, int(args.steps*.95): 1.5e-4}
    start = time.time()
    for step in range(1, args.steps + 1):
        if step in milestones:
            for group in opt.param_groups: group['lr'] = milestones[step]
        tok = batch(args.batch, structured=.35 if step > args.steps * .75 else .05)
        logits = model(tok[:, :-1])
        positions = torch.arange(1, SEQ - 1, 3)
        loss = F.cross_entropy(logits[:, positions].reshape(-1, 10), tok[:, positions + 1].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 250 == 0:
            acc = exact_accuracy(model, 1000, 800000 + step)
            print(step, f'loss={loss.item():.6f}', f'acc={acc:.4f}', f's={time.time()-start:.1f}', flush=True)
            torch.save(model.state_dict(), args.out)
            export(model, '/workspace/submission.py')
    print('final', exact_accuracy(model, 10000), flush=True)
    torch.save(model.state_dict(), args.out)
    export(model, '/workspace/submission.py')


if __name__ == '__main__':
    main()
