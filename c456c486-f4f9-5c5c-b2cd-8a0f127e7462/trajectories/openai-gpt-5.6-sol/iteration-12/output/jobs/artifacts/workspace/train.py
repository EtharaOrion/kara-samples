import argparse
import math
import random
import time

import torch
from torch import nn
import torch.nn.functional as F


class AdditionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        d = 12
        self.digit = nn.Embedding(10, d)
        self.position = nn.Parameter(torch.empty(15, d))
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.projection = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.up = nn.Linear(d, 32)
        self.down = nn.Linear(32, d)
        self.final_norm = nn.LayerNorm(d)
        self.output = nn.Linear(d, 10)
        self.register_buffer("mask", torch.triu(torch.full((15, 15), float("-inf")), 1))
        nn.init.normal_(self.position, std=0.02)

    def forward(self, left, right):
        x = self.digit(left) + self.digit(right) + self.position
        for _ in range(6):
            y = self.norm1(x)
            q, k, v = self.qkv(y).chunk(3, dim=-1)
            q = q.view(-1, 15, 2, 6).transpose(1, 2)
            k = k.view(-1, 15, 2, 6).transpose(1, 2)
            v = v.view(-1, 15, 2, 6).transpose(1, 2)
            scores = q @ k.transpose(-2, -1) / math.sqrt(6)
            attended = (torch.softmax(scores + self.mask, -1) @ v).transpose(1, 2).reshape(-1, 15, 12)
            x = x + self.projection(attended)
            x = x + self.down(F.gelu(self.up(self.norm2(x))))
        return self.output(self.final_norm(x))


def targets(a, b):
    out = torch.empty_like(a)
    carry = torch.zeros(a.shape[0], device=a.device, dtype=torch.long)
    for p in range(15):
        total = a[:, p] + b[:, p] + carry
        out[:, p] = total.remainder(10)
        carry = total.div(10, rounding_mode="floor")
    return out


def batch_random(n, device):
    a = torch.randint(0, 10, (n, 15), device=device)
    b = torch.randint(0, 10, (n, 15), device=device)
    a[:, 14] = 0
    b[:, 14] = 0
    return a, b, targets(a, b)


def batch_structured(n, device):
    # Every call regenerates positions, lengths, digits, and pattern variants.
    a = torch.randint(0, 10, (n, 15), device=device)
    b = torch.randint(0, 10, (n, 15), device=device)
    a[:, 14] = 0
    b[:, 14] = 0
    rows = torch.arange(n, device=device)
    kind = torch.randint(0, 8, (n,), device=device)

    # Random carry chains, including chains ending at or overflowing the top.
    chain = kind < 3
    starts = torch.randint(0, 14, (n,), device=device)
    max_len = 14 - starts
    lengths = 1 + (torch.rand(n, device=device) * max_len).long()
    overflow = kind == 2
    lengths = torch.where(overflow, max_len, lengths)
    for p in range(14):
        active = chain & (starts <= p) & (p < starts + lengths)
        first = active & (starts == p)
        continuation = active & ~first
        x = torch.randint(1, 10, (n,), device=device)
        a[:, p] = torch.where(first, x, a[:, p])
        b[:, p] = torch.where(first, 10 - x + torch.randint(0, 9, (n,), device=device).minimum(x - 1), b[:, p])
        x2 = torch.randint(0, 10, (n,), device=device)
        a[:, p] = torch.where(continuation, x2, a[:, p])
        b[:, p] = torch.where(continuation, 9 - x2, b[:, p])
        ending = chain & (starts + lengths == p)
        lo = torch.randint(0, 9, (n,), device=device)
        a[:, p] = torch.where(ending, lo, a[:, p])
        b[:, p] = torch.where(ending, torch.randint(0, 9, (n,), device=device).minimum(8 - lo), b[:, p])

    # Long complementary runs with independently randomized boundaries.
    comp = kind == 3
    lo = torch.randint(0, 10, (n,), device=device)
    hi = lo + 1 + (torch.rand(n, device=device) * (14 - lo).clamp(min=1)).long()
    for p in range(14):
        use = comp & (lo <= p) & (p < hi)
        x = torch.randint(0, 10, (n,), device=device)
        a[:, p] = torch.where(use, x, a[:, p])
        b[:, p] = torch.where(use, 9 - x, b[:, p])

    # Repeated operands and repeated complementary operands.
    repeated = kind == 4
    da = torch.randint(0, 10, (n,), device=device)
    db = torch.randint(0, 10, (n,), device=device)
    a[:, :14] = torch.where(repeated[:, None], da[:, None], a[:, :14])
    b[:, :14] = torch.where(repeated[:, None], db[:, None], b[:, :14])

    # Sparse digits at arbitrary positions.
    sparse = kind == 5
    a[sparse, :14] = 0
    b[sparse, :14] = 0
    for _ in range(4):
        pos = torch.randint(0, 14, (n,), device=device)
        vals_a = torch.randint(0, 10, (n,), device=device)
        vals_b = torch.randint(0, 10, (n,), device=device)
        use_rows = rows[sparse]
        a[use_rows, pos[sparse]] = vals_a[sparse]
        b[use_rows, pos[sparse]] = vals_b[sparse]

    # Blockwise constant digits with fresh split points.
    blocks = kind == 6
    split1 = torch.randint(1, 8, (n,), device=device)
    split2 = torch.randint(8, 14, (n,), device=device)
    vals = torch.randint(0, 10, (n, 6), device=device)
    for p in range(14):
        seg = (p >= split1).long() + (p >= split2).long()
        va = vals[rows, seg]
        vb = vals[rows, seg + 3]
        a[:, p] = torch.where(blocks, va, a[:, p])
        b[:, p] = torch.where(blocks, vb, b[:, p])

    # Near maxima, zeros, ones, and operand-swapped variants.
    boundary = kind == 7
    choices = torch.randint(0, 6, (n,), device=device)
    templates = torch.tensor([0, 1, 5, 8, 9, 9], device=device)
    da = templates[choices]
    db = templates[torch.randint(0, 6, (n,), device=device)]
    a[:, :14] = torch.where(boundary[:, None], da[:, None], a[:, :14])
    b[:, :14] = torch.where(boundary[:, None], db[:, None], b[:, :14])

    swap = torch.rand(n, device=device) < 0.5
    old_a = a.clone()
    a = torch.where(swap[:, None], b, a)
    b = torch.where(swap[:, None], old_a, b)
    return a, b, targets(a, b)


@torch.no_grad()
def evaluate(model, batches, n, structured=False):
    model.eval()
    correct = total = digit_correct = 0
    for _ in range(batches):
        fn = batch_structured if structured else batch_random
        a, b, y = fn(n, "cuda")
        pred = model(a, b).argmax(-1)
        correct += (pred == y).all(1).sum().item()
        digit_correct += (pred == y).sum().item()
        total += n
    model.train()
    return correct / total, digit_correct / (total * 15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=16000)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--resume")
    args = parser.parse_args()
    torch.manual_seed(12012)
    random.seed(12012)
    torch.set_float32_matmul_precision("high")
    model = AdditionTransformer().cuda()
    if args.resume:
        model.load_state_dict(torch.load(args.resume, weights_only=True))
    print("parameters", sum(p.numel() for p in model.parameters()), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.003, betas=(0.9, 0.98))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=3e-5)
    compiled = torch.compile(model)
    start = time.time()
    best_score = -1
    for step in range(1, args.steps + 1):
        structured_n = args.batch // 4
        ar, br, yr = batch_random(args.batch - structured_n, "cuda")
        ast, bst, yst = batch_structured(structured_n, "cuda")
        a = torch.cat((ar, ast))
        b = torch.cat((br, bst))
        y = torch.cat((yr, yst))
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(compiled(a, b).flatten(0, 1), y.flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % 500 == 0 or step == args.steps:
            random_acc, digit_acc = evaluate(model, 8, 8192)
            edge_acc, _ = evaluate(model, 4, 4096, True)
            score = random_acc + min(edge_acc, 0.99999)
            if score > best_score:
                best_score = score
                torch.save(model.state_dict(), "/workspace/model.pt")
            print(step, f"loss={loss.item():.6f}", f"random={random_acc:.7f}", f"digit={digit_acc:.9f}", f"edge={edge_acc:.7f}", f"lr={scheduler.get_last_lr()[0]:.2g}", f"sec={time.time()-start:.1f}", flush=True)


if __name__ == "__main__":
    main()
