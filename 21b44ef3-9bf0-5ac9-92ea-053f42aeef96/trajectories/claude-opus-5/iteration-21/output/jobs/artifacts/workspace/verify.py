"""Independent audit of the graded file /workspace/submission.py.

Imports the shipped file fresh (nothing from the training side), then measures
everything that gets reported: parameter count, held-out accuracy through add(),
edge cases, every carry pattern, whether add() really depends on the forward
pass, and whether attention is load bearing.
"""

import argparse
import ast
import importlib.util
import random
import sys

import torch


def load(path="/workspace/submission.py"):
    spec = importlib.util.spec_from_file_location("shipped", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["shipped"] = mod
    spec.loader.exec_module(mod)
    return mod


def imports_of(path):
    tree = ast.parse(open(path).read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return sorted(names)


def arithmetic_in_add(path):
    """Any binary or augmented arithmetic appearing inside add()."""
    tree = ast.parse(open(path).read())
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "add":
            for sub in ast.walk(node):
                if isinstance(sub, ast.BinOp):
                    found.append(type(sub.op).__name__)
                elif isinstance(sub, ast.AugAssign):
                    found.append("Aug" + type(sub.op).__name__)
    return found


def rand_operand(rng, width=8):
    return rng.randrange(10 ** (width - 1), 10 ** width)


def carry_heavy(rng):
    """A full-width pair whose places are mostly transparent or generating."""
    da, db = [], []
    for _ in range(8):
        u = rng.random()
        if u < 0.55:                       # transparent
            x = rng.randrange(0, 10)
            y = 9 - x
        elif u < 0.80:                     # generate
            x = rng.randrange(1, 10)
            y = rng.randrange(10 - x, 10)
        else:                              # absorb
            x = rng.randrange(0, 9)
            y = rng.randrange(0, 9 - x)
        da.append(x)
        db.append(y)
    da[7] = max(da[7], 1)
    db[7] = max(db[7], 1)
    return (int("".join(str(d) for d in reversed(da))),
            int("".join(str(d) for d in reversed(db))))


def rate(mod, model, pairs):
    good = sum(1 for a, b in pairs if mod.add(model, a, b) == a + b)
    return good / len(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="/workspace/submission.py")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    print("file:", args.path)
    print("imports:", imports_of(args.path))
    print("arithmetic inside add():", arithmetic_in_add(args.path) or "none")

    mod = load(args.path)
    model, meta = mod.build_model()
    assert isinstance(model, torch.nn.Module)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_par}   buffers: {sum(b.numel() for b in model.buffers())}")
    print("metadata:", meta)

    rng = random.Random(args.seed)
    cases = [(rand_operand(rng), rand_operand(rng)) for _ in range(args.n)]
    got = [mod.add(model, a, b) for a, b in cases]
    bad = [(a, b, g, a + b) for (a, b), g in zip(cases, got) if g != a + b]
    print(f"random 8-digit pairs: {args.n - len(bad)}/{args.n} = "
          f"{1 - len(bad) / args.n:.6f}")
    for row in bad[:5]:
        print("   wrong:", row)

    edges = [(10000000, 10000000), (99999999, 99999999), (99999999, 10000001),
             (10000000, 99999999), (12345678, 87654321), (19999999, 10000001),
             (11111111, 88888889), (55555555, 44444445), (10000001, 19999999),
             (99999998, 10000002), (98765432, 12345678), (50000000, 50000000)]
    ebad = [(a, b) for a, b in edges if mod.add(model, a, b) != a + b]
    print(f"edge cases: {len(edges) - len(ebad)}/{len(edges)}  wrong={ebad}")

    # every carry structure: each place independently absorb / transparent / generate
    rng2 = random.Random(7)
    pats = []
    for pat in range(3 ** 8):
        da, db, q = [], [], pat
        for _ in range(8):
            kind, q = q % 3, q // 3
            if kind == 0:
                x = rng2.randrange(0, 9)
                y = rng2.randrange(0, 9 - x)
            elif kind == 1:
                x = rng2.randrange(0, 10)
                y = 9 - x
            else:
                x = rng2.randrange(1, 10)
                y = rng2.randrange(10 - x, 10)
            da.append(x)
            db.append(y)
        da[7], db[7] = max(da[7], 1), max(db[7], 1)
        pats.append((int("".join(str(d) for d in reversed(da))),
                     int("".join(str(d) for d in reversed(db)))))
    acc = rate(mod, model, pats)
    print(f"all 3^8 carry patterns: {round(acc * len(pats))}/{len(pats)} = {acc:.6f}")

    # does add() actually read the forward pass?
    real_forward = type(model).forward
    type(model).forward = lambda self, pairs: real_forward(self, pairs).roll(3, dims=-1)
    probe = [(rand_operand(rng), rand_operand(rng)) for _ in range(200)]
    corrupted = 1 - rate(mod, model, probe)
    type(model).forward = real_forward
    print(f"corrupted forward -> wrong answers: {corrupted:.4f} (must be 1.0000)")

    # is attention load bearing?  zeroing key_scale removes the content term and
    # leaves a fixed recency pattern; measured on carry-heavy inputs, where the
    # nearest earlier place is usually the wrong place to read the carry from
    rng3 = random.Random(99)
    hard = [carry_heavy(rng3) for _ in range(2000)]
    live = rate(mod, model, hard)
    saved = model.key_scale.clone()
    with torch.no_grad():
        model.key_scale.zero_()
    frozen = rate(mod, model, hard)
    with torch.no_grad():
        model.key_scale.copy_(saved)
    print(f"carry-heavy accuracy: {live:.4f} with attention, "
          f"{frozen:.4f} with the content key removed")

    # how much does the attention map move with the input?
    probe_pairs = torch.tensor([
        [[9, 0], [4, 5], [7, 3], [1, 8], [2, 2], [6, 6], [3, 3], [5, 4]],
        [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8]],
        [[5, 4], [5, 4], [5, 4], [5, 4], [5, 4], [5, 4], [5, 4], [5, 4]],
    ])
    maps = []
    for i in range(probe_pairs.shape[0]):
        rows = attention_map(model, probe_pairs[i:i + 1])
        maps.append(rows)
    spread = max((maps[i] - maps[j]).abs().max().item()
                 for i in range(len(maps)) for j in range(len(maps)))
    print(f"attention map varies across inputs by up to {spread:.4f} "
          f"(0 would mean a fixed pattern)")
    print("done")


def attention_map(model, pairs):
    """Re-derive the strictly-causal attention weights from the model's own tensors."""
    code = model.code()
    x = code[pairs[..., 0]] + code[pairs[..., 1]]
    pad = x.new_zeros(x.shape[0], 1)
    x = torch.cat([pad, x, pad], dim=1)
    places = x.shape[1]
    gate = torch.clamp(model.gate_slope * (x.unsqueeze(-1) - model.knee), 0.0, 1.0)
    key = model.key_scale * (gate[..., 1] - gate[..., 0])
    pos = torch.arange(places)
    gap = pos.unsqueeze(1) - pos.unsqueeze(0)
    strict = torch.where(gap <= 0, torch.finfo(x.dtype).min, 0.0)
    strict[0, 0] = 0.0
    return torch.softmax(key.unsqueeze(1) + model.recency * gap + strict, dim=-1)[0]


if __name__ == "__main__":
    main()
