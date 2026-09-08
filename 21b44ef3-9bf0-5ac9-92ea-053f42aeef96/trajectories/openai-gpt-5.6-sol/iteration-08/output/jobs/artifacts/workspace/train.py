import argparse
import os
import random
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

D = 20
HEADS = 4
MAX_LEN = 25


class SharedBlock(nn.Module):
    def __init__(self, ff_width):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(D, HEADS, batch_first=True)
        self.ln2 = nn.LayerNorm(D)
        self.ff1 = nn.Linear(D, ff_width)
        self.ff2 = nn.Linear(ff_width, D)

    def forward(self, x, mask):
        z = self.ln1(x)
        x = x + self.attn(z, z, z, attn_mask=mask, need_weights=False)[0]
        x = x + self.ff2(F.gelu(self.ff1(self.ln2(x))))
        return x


class Adder(nn.Module):
    def __init__(self, ff_width=10):
        super().__init__()
        self.ff_width = ff_width
        self.token = nn.Embedding(11, D)
        self.position = nn.Embedding(MAX_LEN, D)
        self.passes = nn.Embedding(2, D)
        self.block = SharedBlock(ff_width)
        self.final_norm = nn.LayerNorm(D)
        self.output = nn.Linear(D, 10, bias=False)
        self.register_buffer("causal", torch.triu(torch.ones(MAX_LEN, MAX_LEN, dtype=torch.bool), 1), persistent=False)

    def forward(self, tokens):
        n = tokens.shape[1]
        x = self.token(tokens) + self.position.weight[:n]
        for p in range(2):
            x = x + self.passes.weight[p]
            x = self.block(x, self.causal[:n, :n])
        return self.output(self.final_norm(x))


def digits(x, width):
    out = []
    for _ in range(width):
        out.append(x.remainder(10))
        x = torch.div(x, 10, rounding_mode="floor")
    return torch.stack(out, 1)


def random_pairs(batch, device, structured=0.45):
    a = torch.randint(10_000_000, 100_000_000, (batch,), device=device)
    b = torch.randint(10_000_000, 100_000_000, (batch,), device=device)
    n = int(batch * structured)
    if not n:
        return a, b
    kind = torch.randint(0, 6, (n,), device=device)
    aa = a[:n].clone()
    bb = b[:n].clone()

    # Exact complements create long carry chains and the ninth output digit.
    m = kind == 0
    aa[m] = torch.randint(10_000_000, 90_000_001, (int(m.sum()),), device=device)
    bb[m] = 100_000_000 - aa[m]

    # Near-maximum operands stress repeated nines and top-end carries.
    m = kind == 1
    aa[m] = 99_999_999 - torch.randint(0, 1_000_000, (int(m.sum()),), device=device)
    bb[m] = 99_999_999 - torch.randint(0, 1_000_000, (int(m.sum()),), device=device)

    # Multiples of powers of ten cover sparse and round operands.
    m = kind == 2
    count = int(m.sum())
    if count:
        powers = torch.tensor([10, 100, 1000, 10000, 100000, 1000000], device=device)
        scale = powers[torch.randint(0, len(powers), (count,), device=device)]
        aa[m] = torch.clamp(torch.div(aa[m], scale, rounding_mode="floor") * scale, 10_000_000, 99_999_999)
        bb[m] = torch.clamp(torch.div(bb[m], scale, rounding_mode="floor") * scale, 10_000_000, 99_999_999)

    # Repeated digits are rare under uniform sampling but common edge probes.
    m = kind == 3
    count = int(m.sum())
    if count:
        rep = torch.tensor([11_111_111, 22_222_222, 33_333_333, 44_444_444, 55_555_555,
                            66_666_666, 77_777_777, 88_888_888, 99_999_999], device=device)
        aa[m] = rep[torch.randint(0, 9, (count,), device=device)]
        bb[m] = rep[torch.randint(0, 9, (count,), device=device)]

    # Force a carry through k low columns while retaining random high digits.
    m = kind == 4
    count = int(m.sum())
    if count:
        k = torch.randint(2, 8, (count,), device=device)
        scale = torch.pow(torch.full((count,), 10, device=device, dtype=torch.long), k)
        low = aa[m].remainder(scale)
        high = torch.randint(1, 9, (count,), device=device) * 10_000_000
        forced = high + (scale - low).remainder(scale)
        bb[m] = torch.clamp(forced, 10_000_000, 99_999_999)

    # Sparse decimal patterns and exact lower boundary.
    m = kind == 5
    count = int(m.sum())
    if count:
        vals = torch.tensor([10_000_000, 10_000_001, 10_000_009, 10_000_010, 10_000_099,
                             10_000_100, 10_000_999, 10_009_999, 10_099_999, 10_999_999,
                             50_000_000, 90_000_000, 99_000_000, 99_900_000, 99_990_000,
                             99_999_000, 99_999_900, 99_999_990, 99_999_999], device=device)
        aa[m] = vals[torch.randint(0, len(vals), (count,), device=device)]
        bb[m] = vals[torch.randint(0, len(vals), (count,), device=device)]
    a[:n], b[:n] = aa, bb
    return a, b


def batch_data(batch, device, structured=0.45):
    a, b = random_pairs(batch, device, structured)
    da, db, ds = digits(a, 8), digits(b, 8), digits(a + b, 9)
    prefix = torch.stack((da, db), 2).reshape(batch, 16)
    sentinel = torch.full((batch, 1), 10, dtype=torch.long, device=device)
    tokens = torch.cat((prefix, sentinel, ds[:, :-1]), 1)
    return tokens, ds, a, b


@torch.inference_mode()
def exact_accuracy(model, count, batch=8192, structured=0.0, seed=12345):
    torch.manual_seed(seed)
    model.eval()
    good = total = 0
    for start in range(0, count, batch):
        n = min(batch, count - start)
        tokens, target, _, _ = batch_data(n, next(model.parameters()).device, structured)
        seq = tokens[:, :17]
        pred = []
        for _ in range(9):
            d = model(seq)[:, -1].argmax(1)
            pred.append(d)
            if len(pred) < 9:
                seq = torch.cat((seq, d[:, None]), 1)
        prediction = torch.stack(pred, 1)
        good += (prediction == target).all(1).sum().item()
        total += n
    model.train()
    return good, total


def prune_one(model):
    old = model
    w = old.ff_width
    importance = old.block.ff1.weight.norm(dim=1) * old.block.ff2.weight.norm(dim=0)
    keep = torch.topk(importance, w - 1).indices.sort().values
    new = Adder(w - 1).to(next(old.parameters()).device)
    state = old.state_dict()
    new_state = new.state_dict()
    for name in new_state:
        if name == "block.ff1.weight":
            new_state[name] = state[name][keep]
        elif name == "block.ff1.bias":
            new_state[name] = state[name][keep]
        elif name == "block.ff2.weight":
            new_state[name] = state[name][:, keep]
        else:
            new_state[name] = state[name]
    new.load_state_dict(new_state)
    return new


def tensor_literal(t):
    values = t.detach().cpu().tolist()
    return "torch.tensor(" + repr(values) + ")"


def export_submission(model, path):
    model = model.eval().cpu()
    state_lines = [f"    {name!r}: {tensor_literal(value)}," for name, value in model.state_dict().items()]
    source = '''import torch
from torch import nn
import torch.nn.functional as F

D = 20
MAX_LEN = 25
FF_WIDTH = %d


class SharedBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(D, 4, batch_first=True)
        self.ln2 = nn.LayerNorm(D)
        self.ff1 = nn.Linear(D, FF_WIDTH)
        self.ff2 = nn.Linear(FF_WIDTH, D)

    def forward(self, x, mask):
        z = self.ln1(x)
        x = x + self.attn(z, z, z, attn_mask=mask, need_weights=False)[0]
        return x + self.ff2(F.gelu(self.ff1(self.ln2(x))))


class AdditionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token = nn.Embedding(11, D)
        self.position = nn.Embedding(MAX_LEN, D)
        self.passes = nn.Embedding(2, D)
        self.block = SharedBlock()
        self.final_norm = nn.LayerNorm(D)
        self.output = nn.Linear(D, 10, bias=False)
        self.register_buffer("causal", torch.triu(torch.ones(MAX_LEN, MAX_LEN, dtype=torch.bool), 1), persistent=False)

    def forward(self, tokens):
        n = tokens.shape[1]
        x = self.token(tokens) + self.position.weight[:n]
        for p in range(2):
            x = x + self.passes.weight[p]
            x = self.block(x, self.causal[:n, :n])
        return self.output(self.final_norm(x))


_STATE = {
%s
}


def build_model():
    model = AdditionTransformer()
    model.load_state_dict(_STATE)
    model.eval()
    return model, {"architecture": "shared-two-pass-causal-transformer", "digit_order": "least-significant-first"}


@torch.inference_mode()
def add(model, a: int, b: int) -> int:
    left = [int(c) for c in f"{a:08d}"[::-1]]
    right = [int(c) for c in f"{b:08d}"[::-1]]
    operands = [digit for pair in zip(left, right) for digit in pair]
    tokens = torch.tensor([operands + [10]], dtype=torch.long, device=next(model.parameters()).device)
    result = []
    for step in range(9):
        digit = int(model(tokens)[0, -1].argmax().item())
        result.append(str(digit))
        if step != 8:
            tokens = torch.cat((tokens, torch.tensor([[digit]], device=tokens.device)), dim=1)
    return int("".join(reversed(result)))
''' % (model.ff_width, "\n".join(state_lines))
    Path(path).write_text(source)
    model.to("cuda" if torch.cuda.is_available() else "cpu")


def train_stage(model, steps, lr, batch, structured, log_every, output):
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    started = time.time()
    model.train()
    for step in range(1, steps + 1):
        tokens, target, _, _ = batch_data(batch, device, structured)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(tokens)[:, 16:25]
            loss = F.cross_entropy(logits.reshape(-1, 10), target.reshape(-1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if step % log_every == 0 or step == steps:
            teacher = (logits.argmax(2) == target).all(1).float().mean().item()
            elapsed = time.time() - started
            print(f"step {step}/{steps} loss={loss.item():.5f} teacher_exact={teacher:.4f} {elapsed:.1f}s", flush=True)
            torch.save({"ff_width": model.ff_width, "state": model.state_dict()}, output)
            export_submission(model, "/workspace/submission.py")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--steps", type=int, default=24000)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--structured", type=float, default=0.45)
    parser.add_argument("--prune", action="store_true")
    parser.add_argument("--output", default="/workspace/model.pt")
    parser.add_argument("--validate", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(7)
    random.seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        model = Adder(checkpoint["ff_width"]).to(device)
        model.load_state_dict(checkpoint["state"])
    else:
        model = Adder(args.width).to(device)
    if args.prune:
        model = prune_one(model)
        print(f"pruned to FF width {model.ff_width}")
    export_submission(model, "/workspace/submission.py")
    if args.init:
        torch.save({"ff_width": model.ff_width, "state": model.state_dict()}, args.output)
        return
    if args.steps:
        train_stage(model, args.steps, args.lr, args.batch, args.structured, 250, args.output)
    if args.validate:
        print("random", exact_accuracy(model, args.validate, structured=0.0))
        print("structured", exact_accuracy(model, args.validate, structured=1.0, seed=54321))
    torch.save({"ff_width": model.ff_width, "state": model.state_dict()}, args.output)
    export_submission(model, "/workspace/submission.py")


if __name__ == "__main__":
    main()
