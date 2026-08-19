import math
import os
import random
import time
import torch
from torch import nn
import torch.nn.functional as F

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
DEVICE = "cuda"
DIGITS = 14


class Adder(nn.Module):
    def __init__(self, width=12, hidden=32, rounds=6):
        super().__init__()
        self.width = width
        self.rounds = rounds
        self.digit = nn.Embedding(10, width)
        self.position = nn.Parameter(torch.empty(15, width))
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.project = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.up = nn.Linear(width, hidden)
        self.down = nn.Linear(hidden, width)
        self.final_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, 10)
        self.register_buffer("causal", torch.tril(torch.ones(15, 15, dtype=torch.bool)), persistent=False)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, left, right):
        x = self.digit(left) + self.digit(right) + self.position
        scale = (self.width // 2) ** -0.5
        for _ in range(self.rounds):
            z = self.norm1(x)
            q, k, v = self.qkv(z).chunk(3, dim=-1)
            q = q.view(-1, 15, 2, self.width // 2).transpose(1, 2)
            k = k.view(-1, 15, 2, self.width // 2).transpose(1, 2)
            v = v.view(-1, 15, 2, self.width // 2).transpose(1, 2)
            scores = (q @ k.transpose(-2, -1)) * scale
            scores = scores.masked_fill(~self.causal, -torch.inf)
            attended = (scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(-1, 15, self.width)
            x = x + self.project(attended)
            x = x + self.down(F.gelu(self.up(self.norm2(x))))
        return self.output(self.final_norm(x))


def outputs(left, right):
    batch = left.shape[0]
    result = torch.empty(batch, 15, dtype=torch.long, device=left.device)
    carry = torch.zeros(batch, dtype=torch.long, device=left.device)
    for p in range(14):
        total = left[:, p] + right[:, p] + carry
        result[:, p] = total.remainder(10)
        carry = total.ge(10).long()
    result[:, 14] = carry
    return result


# Uniformly choose a digit pair compatible with each requested carry transition.
PAIR_TABLE = torch.zeros(2, 2, 55, 2, dtype=torch.long, device=DEVICE)
PAIR_COUNT = torch.zeros(2, 2, dtype=torch.long, device=DEVICE)
for cin in range(2):
    for cout in range(2):
        pairs = [(a, b) for a in range(10) for b in range(10) if int(a + b + cin >= 10) == cout]
        PAIR_COUNT[cin, cout] = len(pairs)
        for j, pair in enumerate(pairs):
            PAIR_TABLE[cin, cout, j] = torch.tensor(pair, device=DEVICE)


def random_batch(batch):
    left = torch.randint(10, (batch, 15), device=DEVICE)
    right = torch.randint(10, (batch, 15), device=DEVICE)
    left[:, 14] = 0
    right[:, 14] = 0
    return left, right, outputs(left, right)


def carry_batch(batch, mode="mixed"):
    # Desired carry-out trajectories include arbitrary transitions and contiguous chains.
    desired = torch.zeros(batch, 14, dtype=torch.long, device=DEVICE)
    split = batch // 2
    desired[:split] = torch.randint(2, (split, 14), device=DEVICE)
    starts = torch.randint(14, (batch - split,), device=DEVICE)
    lengths = torch.randint(1, 15, (batch - split,), device=DEVICE)
    positions = torch.arange(14, device=DEVICE)
    desired[split:] = ((positions >= starts[:, None]) &
                       (positions < (starts + lengths).clamp_max(14)[:, None])).long()
    # Ensure edge trajectories occur every batch.
    if batch >= 16:
        desired[-1].fill_(1)
        desired[-2].zero_()
        desired[-3] = (positions >= 1).long()
        desired[-4] = (positions < 13).long()
    incoming = torch.cat((torch.zeros(batch, 1, dtype=torch.long, device=DEVICE), desired[:, :-1]), 1)
    counts = PAIR_COUNT[incoming, desired]
    choice = (torch.rand(batch, 14, device=DEVICE) * counts).long()
    pairs = PAIR_TABLE[incoming, desired, choice]
    left = torch.zeros(batch, 15, dtype=torch.long, device=DEVICE)
    right = torch.zeros_like(left)
    left[:, :14] = pairs[:, :, 0]
    right[:, :14] = pairs[:, :, 1]
    return left, right, outputs(left, right)


def mixed_batch(batch, structured_fraction):
    n = int(batch * structured_fraction)
    left, right, target = random_batch(batch)
    if n:
        sl, sr, st = carry_batch(n)
        left[:n], right[:n], target[:n] = sl, sr, st
    return left, right, target


@torch.no_grad()
def evaluate(model, batches=10, batch=8192, structured=False):
    model.eval()
    wrong_seq = wrong_digit = total = 0
    for _ in range(batches):
        left, right, target = carry_batch(batch) if structured else random_batch(batch)
        pred = model(left, right).argmax(-1)
        errors = pred.ne(target)
        wrong_seq += errors.any(1).sum().item()
        wrong_digit += errors.sum().item()
        total += batch
    model.train()
    return wrong_seq, wrong_digit, total


def main():
    width = int(os.environ.get("WIDTH", 12))
    hidden = int(os.environ.get("HIDDEN", 32))
    rounds = int(os.environ.get("ROUNDS", 6))
    steps = int(os.environ.get("STEPS", 10000))
    batch = int(os.environ.get("BATCH", 4096))
    seed = int(os.environ.get("SEED", 20250814))
    torch.manual_seed(seed)
    random.seed(seed)
    model = Adder(width, hidden, rounds).to(DEVICE)
    print("parameters", sum(p.numel() for p in model.parameters()), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.98), weight_decay=0.003)
    best = None
    start = time.time()
    for step in range(1, steps + 1):
        # Introduce unusual carry trajectories throughout, then emphasize them gently.
        frac = 0.12 if step < steps * 0.65 else 0.25
        left, right, target = mixed_batch(batch, frac)
        logits = model(left, right)
        loss = F.cross_entropy(logits.flatten(0, 1), target.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        progress = step / steps
        if progress < .45:
            lr = 3e-3
        elif progress < .7:
            lr = 1e-3
        elif progress < .88:
            lr = 3e-4
        elif progress < .96:
            lr = 1e-4
        else:
            lr = 3e-5
        for group in optimizer.param_groups:
            group["lr"] = lr
        if step % 500 == 0 or step == steps:
            rw, rd, rn = evaluate(model, batches=8)
            sw, sd, sn = evaluate(model, batches=4, structured=True)
            metric = rw / rn + 2 * sw / sn
            print(f"step={step} loss={loss.item():.6g} lr={lr:g} random={rw}/{rn} digits={rd} structured={sw}/{sn} digits={sd} sec={time.time()-start:.1f}", flush=True)
            if best is None or metric < best[0]:
                best = (metric, step, rw, sw)
                torch.save({"model": model.state_dict(), "config": (width, hidden, rounds), "step": step, "metric": metric}, "/workspace/best.pt")
    checkpoint = torch.load("/workspace/best.pt", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    rw, rd, rn = evaluate(model, batches=64)
    sw, sd, sn = evaluate(model, batches=16, structured=True)
    print("BEST", checkpoint["step"], "random", rw, rn, "structured", sw, sn, "digit_errors", rd, sd, flush=True)


if __name__ == "__main__":
    main()
