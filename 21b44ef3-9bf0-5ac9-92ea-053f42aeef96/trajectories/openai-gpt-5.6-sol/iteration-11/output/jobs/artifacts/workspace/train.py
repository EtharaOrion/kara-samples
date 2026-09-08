import argparse
import math
import os
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

WIDTH = 20
HEADS = 4
FF = 4
SEQ = 25
POW10 = torch.tensor([10**i for i in range(9)], dtype=torch.long)


class Adder(nn.Module):
    def __init__(self, pos_rank=8, pass_mode="full"):
        super().__init__()
        self.pos_rank = pos_rank
        self.pass_mode = pass_mode
        self.token = nn.Embedding(11, WIDTH)
        self.pos_left = nn.Parameter(torch.empty(SEQ, pos_rank))
        self.pos_right = nn.Parameter(torch.empty(pos_rank, WIDTH))
        if pass_mode == "full":
            self.pass_embed = nn.Parameter(torch.empty(2, WIDTH))
        else:
            self.pass_vector = nn.Parameter(torch.empty(WIDTH))
        self.norm1 = nn.LayerNorm(WIDTH)
        self.attn = nn.MultiheadAttention(WIDTH, HEADS, batch_first=True)
        self.norm2 = nn.LayerNorm(WIDTH)
        self.fc1 = nn.Linear(WIDTH, FF)
        self.fc2 = nn.Linear(FF, WIDTH)
        self.final_norm = nn.LayerNorm(WIDTH)
        self.head = nn.Linear(WIDTH, 10, bias=False)
        self.register_buffer("mask", torch.triu(torch.ones(SEQ, SEQ, dtype=torch.bool), diagonal=1), persistent=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.token.weight, std=0.2)
        nn.init.normal_(self.pos_left, std=0.2)
        nn.init.normal_(self.pos_right, std=0.2)
        if self.pass_mode == "full":
            nn.init.normal_(self.pass_embed, std=0.2)
        else:
            nn.init.normal_(self.pass_vector, std=0.2)

    def forward(self, tokens):
        length = tokens.shape[1]
        h = self.token(tokens) + (self.pos_left[:length] @ self.pos_right)
        mask = self.mask[:length, :length]
        for step in range(2):
            if self.pass_mode == "full":
                h = h + self.pass_embed[step]
            elif step == 1:
                h = h + self.pass_vector
            z = self.norm1(h)
            h = h + self.attn(z, z, z, attn_mask=mask, need_weights=False)[0]
            h = h + self.fc2(F.gelu(self.fc1(self.norm2(h))))
        return self.head(self.final_norm(h))


def digits(n, width):
    p = POW10[:width].to(n.device)
    return torch.remainder(torch.div(n[:, None], p, rounding_mode="floor"), 10)


def batch_data(batch, device, structured=0.2):
    a = torch.randint(10_000_000, 100_000_000, (batch,), device=device)
    b = torch.randint(10_000_000, 100_000_000, (batch,), device=device)
    count = int(batch * structured)
    if count:
        mode = torch.randint(0, 5, (count,), device=device)
        x = torch.randint(10_000_000, 90_000_001, (count,), device=device)
        complement_b = 100_000_000 - x
        near_a = 99_999_999 - torch.randint(0, 100_000, (count,), device=device)
        near_b = 99_999_999 - torch.randint(0, 100_000, (count,), device=device)
        scale = torch.pow(torch.full((count,), 10, device=device), torch.randint(1, 7, (count,), device=device))
        round_a = torch.div(a[:count], scale, rounding_mode="floor") * scale
        round_b = torch.div(b[:count], scale, rounding_mode="floor") * scale
        repeat_a = torch.randint(1, 10, (count,), device=device) * 11_111_111
        repeat_b = torch.randint(1, 10, (count,), device=device) * 11_111_111
        carry_scale = torch.pow(torch.full((count,), 10, device=device), torch.randint(2, 8, (count,), device=device))
        carry_a = torch.randint(10_000_000, 100_000_000, (count,), device=device)
        carry_b = torch.div(b[:count], carry_scale, rounding_mode="floor") * carry_scale
        carry_b += torch.remainder(carry_scale - torch.remainder(carry_a, carry_scale), carry_scale)
        carry_b = torch.where(carry_b < 10_000_000, carry_b + 10_000_000, carry_b)
        carry_b = torch.where(carry_b >= 100_000_000, carry_b - 10_000_000, carry_b)
        a[:count] = torch.where(mode == 0, x, torch.where(mode == 1, near_a, torch.where(mode == 2, round_a, torch.where(mode == 3, repeat_a, carry_a))))
        b[:count] = torch.where(mode == 0, complement_b, torch.where(mode == 1, near_b, torch.where(mode == 2, round_b, torch.where(mode == 3, repeat_b, carry_b))))
    da, db = digits(a, 8), digits(b, 8)
    operands = torch.stack((da, db), dim=2).reshape(batch, 16)
    target = digits(a + b, 9)
    tokens = torch.empty(batch, SEQ, dtype=torch.long, device=device)
    tokens[:, :16] = operands
    tokens[:, 16] = 10
    tokens[:, 17:] = target[:, :8]
    return tokens, target, a, b


@torch.no_grad()
def autoregressive_accuracy(model, total=10000, batch=2000, structured=0.0):
    model.eval()
    good = 0
    digit_good = 0
    seen = 0
    device = next(model.parameters()).device
    while seen < total:
        n = min(batch, total - seen)
        tokens, target, _, _ = batch_data(n, device, structured)
        prefix = tokens[:, :17]
        out = []
        for _ in range(9):
            pred = model(prefix)[:, -1].argmax(1)
            out.append(pred)
            prefix = torch.cat((prefix, pred[:, None]), 1)
        pred = torch.stack(out, 1)
        good += (pred == target).all(1).sum().item()
        digit_good += (pred == target).sum().item()
        seen += n
    model.train()
    return good / total, digit_good / (9 * total)


def train_steps(model, optimizer, steps, batch, lr, structured, start=0, checkpoint=None):
    device = next(model.parameters()).device
    for group in optimizer.param_groups:
        group["lr"] = lr
    model.train()
    for step in range(start, steps):
        tokens, target, _, _ = batch_data(batch, device, structured)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(tokens)[:, 16:25]
            loss = F.cross_entropy(logits.reshape(-1, 10), target.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % 500 == 0:
            print(f"step {step+1}/{steps} loss {loss.item():.6f}", flush=True)
        if (step + 1) % 2000 == 0 and checkpoint:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step + 1}, checkpoint)
    if checkpoint:
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": steps}, checkpoint)


def copy_common(source, target):
    src = source.state_dict()
    dst = target.state_dict()
    for key in dst:
        if key in src and dst[key].shape == src[key].shape:
            dst[key].copy_(src[key])
    target.load_state_dict(dst)


def compress_position(source, rank):
    target = Adder(rank, source.pass_mode).to(next(source.parameters()).device)
    copy_common(source, target)
    with torch.no_grad():
        matrix = source.pos_left @ source.pos_right
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        root = s[:rank].sqrt()
        target.pos_left.copy_(u[:, :rank] * root)
        target.pos_right.copy_(root[:, None] * vh[:rank])
    return target


def fixed_pass(source):
    target = Adder(source.pos_rank, "fixed").to(next(source.parameters()).device)
    copy_common(source, target)
    with torch.no_grad():
        target.token.weight.add_(source.pass_embed[0])
        target.pass_vector.copy_(source.pass_embed[1])
    return target


def tensor_literal(tensor):
    value = tensor.detach().cpu().float().tolist()
    return "torch.tensor(" + repr(value) + ")"


def export_submission(model, path):
    state = model.state_dict()
    state.pop("mask", None)
    entries = ",\n".join(f"    {key!r}: {tensor_literal(value)}" for key, value in state.items())
    source = f'''import torch
from torch import nn
import torch.nn.functional as F


class AdditionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token = nn.Embedding(11, 20)
        self.pos_left = nn.Parameter(torch.empty(25, 7))
        self.pos_right = nn.Parameter(torch.empty(7, 20))
        self.pass_vector = nn.Parameter(torch.empty(20))
        self.norm1 = nn.LayerNorm(20)
        self.attn = nn.MultiheadAttention(20, 4, batch_first=True)
        self.norm2 = nn.LayerNorm(20)
        self.fc1 = nn.Linear(20, 4)
        self.fc2 = nn.Linear(4, 20)
        self.final_norm = nn.LayerNorm(20)
        self.head = nn.Linear(20, 10, bias=False)
        self.register_buffer("mask", torch.triu(torch.ones(25, 25, dtype=torch.bool), diagonal=1), persistent=False)

    def forward(self, tokens):
        length = tokens.shape[1]
        h = self.token(tokens) + self.pos_left[:length] @ self.pos_right
        mask = self.mask[:length, :length]
        for step in range(2):
            if step == 1:
                h = h + self.pass_vector
            z = self.norm1(h)
            h = h + self.attn(z, z, z, attn_mask=mask, need_weights=False)[0]
            h = h + self.fc2(F.gelu(self.fc1(self.norm2(h))))
        return self.head(self.final_norm(h))


_TRAINED_STATE = {{
{entries}
}}


def build_model():
    model = AdditionTransformer()
    model.load_state_dict(_TRAINED_STATE)
    model.eval()
    return model, {{"architecture": "recurrent causal digit transformer", "digit_order": "least-significant-first"}}


def add(model, a: int, b: int) -> int:
    left = [ord(c) - 48 for c in reversed(f"{{a:08d}}")] 
    right = [ord(c) - 48 for c in reversed(f"{{b:08d}}")] 
    sequence = []
    for x, y in zip(left, right):
        sequence.extend((x, y))
    sequence.append(10)
    device = next(model.parameters()).device
    tokens = torch.tensor([sequence], dtype=torch.long, device=device)
    generated = []
    with torch.no_grad():
        for _ in range(9):
            digit = int(model(tokens)[0, -1].argmax().item())
            generated.append(digit)
            tokens = torch.cat((tokens, torch.tensor([[digit]], device=device)), dim=1)
    return int("".join(str(d) for d in reversed(generated)))
'''
    Path(path).write_text(source)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-steps", type=int, default=26000)
    parser.add_argument("--stabilize-steps", type=int, default=10000)
    parser.add_argument("--rank7-steps", type=int, default=10000)
    parser.add_argument("--fixed-steps", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    base = Adder(8, "full").to(device)
    optimizer = torch.optim.AdamW(base.parameters(), lr=2e-3, weight_decay=0.01, fused=True)
    train_steps(base, optimizer, args.base_steps, args.batch, 2e-3, 0.20, checkpoint="/workspace/base.pt")
    optimizer = torch.optim.AdamW(base.parameters(), lr=1e-4, weight_decay=0.0, fused=True)
    train_steps(base, optimizer, args.stabilize_steps, args.batch, 1e-4, 0.35, checkpoint="/workspace/base_stable.pt")
    print("base validation", autoregressive_accuracy(base, 100000, structured=0.0), autoregressive_accuracy(base, 100000, structured=0.7), flush=True)

    rank7 = compress_position(base, 7)
    optimizer = torch.optim.AdamW(rank7.parameters(), lr=2e-5, weight_decay=0.0, fused=True)
    train_steps(rank7, optimizer, args.rank7_steps, args.batch, 2e-5, 0.55, checkpoint="/workspace/rank7.pt")
    print("rank7 validation", autoregressive_accuracy(rank7, 100000, structured=0.0), autoregressive_accuracy(rank7, 100000, structured=0.7), flush=True)

    final = fixed_pass(rank7)
    optimizer = torch.optim.AdamW(final.parameters(), lr=1e-5, weight_decay=0.0, fused=True)
    train_steps(final, optimizer, args.fixed_steps, args.batch, 1e-5, 0.55, checkpoint="/workspace/final.pt")
    print("fixed validation", autoregressive_accuracy(final, 200000, structured=0.0), autoregressive_accuracy(final, 200000, structured=0.7), flush=True)
    export_submission(final, "/workspace/submission.py")
    print("parameters", sum(p.numel() for p in final.parameters()), flush=True)


if __name__ == "__main__":
    main()
