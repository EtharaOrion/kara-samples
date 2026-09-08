import argparse
import importlib.util
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

WORKSPACE = "/workspace"
SUBMISSION = os.path.join(WORKSPACE, "submission.py")
CHECKPOINT = os.path.join(WORKSPACE, "checkpoint.pt")
DEVICE = "cuda"
BATCH = 8192
POW10 = torch.tensor([10 ** i for i in range(9)], device=DEVICE, dtype=torch.long)


def load_submission():
    spec = importlib.util.spec_from_file_location("addition_submission", SUBMISSION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digits(values, count):
    return (values[:, None] // POW10[:count]) % 10


def make_batch(batch=BATCH, structured=0.18):
    a = torch.randint(10_000_000, 100_000_000, (batch,), device=DEVICE)
    b = torch.randint(10_000_000, 100_000_000, (batch,), device=DEVICE)
    n = int(batch * structured)
    if n:
        kind = torch.randint(0, 7, (n,), device=DEVICE)
        x = torch.randint(10_000_000, 90_000_001, (n,), device=DEVICE)
        y = 100_000_000 - x
        # Exact and near complements exercise the ninth digit and full carry chains.
        delta = torch.randint(-20, 21, (n,), device=DEVICE)
        aa, bb = x, (y + delta).clamp(10_000_000, 99_999_999)
        # Asymmetric 0/9 suffixes with every carry start and stop position.
        run = torch.randint(1, 8, (n,), device=DEVICE)
        scale = torch.pow(torch.tensor(10, device=DEVICE), run)
        prefix_a = torch.randint(1, 10, (n,), device=DEVICE) * 10_000_000
        prefix_a += torch.randint(0, 10_000_000, (n,), device=DEVICE) // scale * scale
        suffix_a = torch.randint(0, 10_000_000, (n,), device=DEVICE) % scale
        suffix_b = (scale - suffix_a + torch.randint(-2, 3, (n,), device=DEVICE)) % scale
        carry_a = (prefix_a + suffix_a).clamp(10_000_000, 99_999_999)
        carry_b = (torch.randint(1, 10, (n,), device=DEVICE) * 10_000_000 + suffix_b).clamp(10_000_000, 99_999_999)
        use = kind >= 2
        aa = torch.where(use, carry_a, aa)
        bb = torch.where(use, carry_b, bb)
        # Repeated/sparse/extreme decimal patterns.
        reps = torch.tensor([11_111_111, 22_222_222, 33_333_333, 44_444_444,
                             55_555_555, 66_666_666, 77_777_777, 88_888_888,
                             99_999_999], device=DEVICE)
        ri = torch.randint(0, len(reps), (n,), device=DEVICE)
        aa = torch.where(kind == 5, reps[ri], aa)
        bb = torch.where(kind == 5, reps[torch.randint(0, len(reps), (n,), device=DEVICE)], bb)
        edge = torch.randint(0, 2, (n,), device=DEVICE).bool()
        aa = torch.where(kind == 6, torch.where(edge, torch.full_like(aa, 10_000_000), torch.full_like(aa, 99_999_999)), aa)
        bb = torch.where(kind == 6, torch.randint(10_000_000, 100_000_000, (n,), device=DEVICE), bb)
        a[:n], b[:n] = aa, bb
    ad, bd = digits(a, 8), digits(b, 8)
    out = digits(a + b, 9)
    tokens = torch.empty(batch, 25, dtype=torch.long, device=DEVICE)
    tokens[:, 0:16:2] = ad
    tokens[:, 1:16:2] = bd
    tokens[:, 16] = 10
    tokens[:, 17:] = out[:, :8]
    return tokens, out


@torch.no_grad()
def evaluate(model, batches=8, structured=0.0):
    model.eval()
    right = total = 0
    min_margin = 1e9
    for _ in range(batches):
        tokens, target = make_batch(BATCH, structured)
        seq = tokens[:, :17]
        pred = []
        for j in range(9):
            logits = model(seq)[:, -1]
            top = logits.topk(2, 1).values
            min_margin = min(min_margin, float((top[:, 0] - top[:, 1]).min()))
            p = logits.argmax(1)
            pred.append(p)
            if j < 8:
                seq = torch.cat((seq, p[:, None]), 1)
        pred = torch.stack(pred, 1)
        right += int((pred == target).all(1).sum())
        total += BATCH
    model.train()
    return right / total, min_margin


def train_stage(model, steps, lr, structured, name):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.01)
    model.train()
    started = time.time()
    for step in range(1, steps + 1):
        tokens, target = make_batch(BATCH, structured)
        logits = model(tokens)
        loss = F.cross_entropy(logits[:, 16:].reshape(-1, 10), target.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 1000 == 0:
            acc, margin = evaluate(model, 2, 0.0)
            sacc, _ = evaluate(model, 2, 0.55)
            elapsed = time.time() - started
            print(f"{name} {step}/{steps} loss={loss.item():.6f} random={acc:.6f} structured={sacc:.6f} margin={margin:.3f} sec={elapsed:.1f}", flush=True)
            torch.save({"model": model.state_dict(), "ff": model.ff1.out_features,
                        "stage": name, "step": step}, CHECKPOINT)
    torch.save({"model": model.state_dict(), "ff": model.ff1.out_features,
                "stage": name, "step": steps}, CHECKPOINT)


def prune_ff(module, model, width):
    old = model
    new = module.AdditionTransformer(width).to(DEVICE)
    state = old.state_dict()
    importance = old.ff1.weight.norm(dim=1) * old.ff2.weight.norm(dim=0)
    keep = importance.topk(width).indices.sort().values
    state["ff1.weight"] = state["ff1.weight"][keep]
    state["ff2.weight"] = state["ff2.weight"][:, keep]
    new.load_state_dict(state)
    return new


def literal(t):
    values = t.detach().cpu().reshape(-1).tolist()
    return "torch.tensor(" + repr(values) + ",dtype=torch.float32).reshape(" + repr(tuple(t.shape)) + ")"


def export(model):
    source = open(SUBMISSION).read()
    marker = "_STATE = None"
    entries = []
    for key, value in model.state_dict().items():
        entries.append(repr(key) + ":" + literal(value))
    replacement = "_STATE = {" + ",\n".join(entries) + "}"
    if marker not in source:
        start = source.index("_STATE = {")
        end = source.index("\n\n\nclass AdditionTransformer", start)
        source = source[:start] + replacement + source[end:]
    else:
        source = source.replace(marker, replacement, 1)
    path = SUBMISSION + ".new"
    with open(path, "w") as handle:
        handle.write(source)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(path, SUBMISSION)
    print("exported", sum(p.numel() for p in model.parameters()), "parameters", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(2025)
    random.seed(2025)
    torch.backends.cuda.matmul.allow_tf32 = True
    module = load_submission()
    model = module.AdditionTransformer(4).to(DEVICE)
    # Write a trained submission as soon as the teacher is robust, then overwrite after each safe compression.
    train_stage(model, 10000 if args.quick else 36000, 2e-3, 0.18, "teacher")
    train_stage(model, 2000 if args.quick else 6000, 5e-5, 0.30, "stabilize")
    print("teacher final", evaluate(model, 20, 0.0), evaluate(model, 20, 0.55), flush=True)
    export(model)
    if args.quick:
        return
    model = prune_ff(module, model, 3)
    train_stage(model, 18000, 2e-5, 0.35, "width3")
    print("width3 final", evaluate(model, 20, 0.0), evaluate(model, 20, 0.55), flush=True)
    export(model)
    model = prune_ff(module, model, 2)
    train_stage(model, 30000, 1.2e-5, 0.40, "width2")
    train_stage(model, 12000, 4e-6, 0.40, "polish")
    print("width2 final", evaluate(model, 64, 0.0), evaluate(model, 64, 0.55), flush=True)
    export(model)


if __name__ == "__main__":
    main()
