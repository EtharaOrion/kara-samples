"""Independent audit of whatever currently sits at /workspace/submission.py.

Nothing here is imported from the training code except the data sampler used
to draw held-out cases; the model, the parameter count and every accuracy
number come from importing the graded file itself and calling its `add`.
"""

import argparse
import ast
import importlib.util
import random
import sys
import time
import torch


def load(path):
    spec = importlib.util.spec_from_file_location("graded", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def screen(path):
    """What a static screen would see: imports, and arithmetic inside add()."""
    tree = ast.parse(open(path).read())
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    print(f"imports: {sorted(imports)}")
    assert imports <= {"torch"}, f"unexpected imports: {imports}"

    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "add")
    bad = [type(n).__name__ for n in ast.walk(fn)
           if isinstance(n, (ast.BinOp, ast.AugAssign))]
    print(f"arithmetic nodes inside add(): {bad if bad else 'none'}")
    assert not bad, "add() must not do arithmetic on the operands"

    lines = open(path).read().count("\n")
    print(f"graded file: {lines} lines, {len(open(path).read())} bytes")


def bulk_digits(model, a, b, width):
    """Batched forward, for the large sweeps; agreement with add() is asserted
    separately so this only ever stands in for the same computation."""
    da = torch.stack([(a // 10 ** k) % 10 for k in range(width)], dim=1)
    db = torch.stack([(b // 10 ** k) % 10 for k in range(width)], dim=1)
    with torch.no_grad():
        digits = model(da, db).argmax(-1)
    powers = 10 ** torch.arange(width + 1, dtype=torch.int64)
    return (digits * powers).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, default="submission.py")
    ap.add_argument("--bulk", type=int, default=1000000)
    ap.add_argument("--api", type=int, default=20000)
    args = ap.parse_args()

    print(f"== static screen of {args.path} ==")
    screen(args.path)

    mod = load(args.path)
    model, meta = mod.build_model()
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    n_buf = sum(b.numel() for b in model.buffers())
    print(f"\n== model ==\nparameters {n_par} (all {n_all}), buffer values {n_buf}")
    for name, p in model.named_parameters():
        print(f"  {name:10s} {tuple(p.shape)}")
    for name, b in model.named_buffers():
        print(f"  buffer {name:12s} {[round(float(x), 6) for x in b.flatten()]}")
    print("metadata:", meta)

    rng = random.Random(12345)

    def bucket(a, b):
        h = a * 100000000 + b
        h = (h ^ (h >> 29)) & 0x3FFFFFFF
        h = (h * 1103515245 + 12345) & 0x3FFFFFFF
        h = (h ^ (h >> 15)) & 0x3FFFFFFF
        return h % 16

    print(f"\n== add() on {args.api} held-out 8-digit pairs ==")
    cases, t0 = [], time.time()
    while len(cases) < args.api:
        a = rng.randint(10000000, 99999999)
        b = rng.randint(10000000, 99999999)
        if bucket(a, b) == 0:
            cases.append((a, b))
    wrong = [(a, b) for a, b in cases if mod.add(model, a, b) != a + b]
    print(f"wrong: {len(wrong)} / {len(cases)}   "
          f"({(time.time() - t0) / len(cases) * 1e3:.2f} ms per call)")
    if wrong:
        print("  examples:", wrong[:5])

    print(f"\n== batched sweep, {args.bulk} uniform full-width pairs ==")
    torch.manual_seed(7)
    bad, seen, t0 = 0, 0, time.time()
    for _ in range(max(1, args.bulk // 100000)):
        a = torch.randint(10000000, 100000000, (100000,), dtype=torch.int64)
        b = torch.randint(10000000, 100000000, (100000,), dtype=torch.int64)
        bad += int((bulk_digits(model, a, b, 8) != a + b).sum())
        seen += a.numel()
    print(f"wrong: {bad} / {seen}   accuracy "
          f"{1 - bad / seen:.8f}   [{time.time() - t0:.0f}s]")

    print("\n== agreement between add() and the batched path ==")
    a = torch.randint(10000000, 100000000, (4000,), dtype=torch.int64)
    b = torch.randint(10000000, 100000000, (4000,), dtype=torch.int64)
    ref = bulk_digits(model, a, b, 8)
    dis = sum(1 for i in range(4000)
              if mod.add(model, int(a[i]), int(b[i])) != int(ref[i]))
    print(f"disagreements: {dis} / 4000")
    assert dis == 0

    print("\n== every carry-structure pattern (3^8 = 6561) ==")
    reps = {0: (3, 4), 1: (4, 5), 2: (7, 6)}     # absorb / transparent / generate
    A, B = [], []
    for pat in range(3 ** 8):
        da, db, q = [], [], pat
        for _ in range(8):
            x, y = reps[q % 3]
            da.append(x)
            db.append(y)
            q //= 3
        A.append(sum(d * 10 ** k for k, d in enumerate(da)))
        B.append(sum(d * 10 ** k for k, d in enumerate(db)))
    At, Bt = torch.tensor(A), torch.tensor(B)
    bad = int((bulk_digits(model, At, Bt, 8) != At + Bt).sum())
    print(f"wrong: {bad} / {len(A)}")

    print("\n== edge cases ==")
    edges = [(99999999, 99999999), (10000000, 10000000), (99999999, 10000001),
             (19999999, 10000001), (10000001, 19999999), (55555555, 44444445),
             (12345678, 87654321), (99999998, 10000001), (98765432, 12345678),
             (11111111, 88888889), (50000000, 50000000), (99999999, 10000000)]
    ebad = [(a, b, mod.add(model, a, b), a + b)
            for a, b in edges if mod.add(model, a, b) != a + b]
    print(f"wrong: {len(ebad)} / {len(edges)}" + (f"  {ebad}" if ebad else ""))

    print("\n== other widths (generalisation, not graded) ==")
    for w in (2, 3, 5, 11, 16):
        lo, hi = 10 ** (w - 1), 10 ** w
        a = torch.randint(lo, hi, (20000,), dtype=torch.int64)
        b = torch.randint(lo, hi, (20000,), dtype=torch.int64)
        e = int((bulk_digits(model, a, b, w) != a + b).sum())
        print(f"  width {w:2d}: {20000 - e} / 20000")

    print("\n== is the model actually used? ==")
    orig = mod.DigitPairAdder.forward
    changed = 0
    trial = [(rng.randint(10000000, 99999999), rng.randint(10000000, 99999999))
             for _ in range(300)]
    base = [mod.add(model, a, b) for a, b in trial]
    mod.DigitPairAdder.forward = lambda self, da, db: torch.randn(
        da.shape[0], da.shape[1] + 1, 10)
    got = [mod.add(model, a, b) for a, b in trial]
    mod.DigitPairAdder.forward = orig
    changed = sum(1 for x, y in zip(base, got) if x != y)
    print(f"answers changed when the forward pass is corrupted: {changed} / 300")
    assert changed == 300

    print("\n== is attention doing work? ==")
    # carry-heavy inputs, where recency alone is not the right routing
    pats = torch.randint(0, 3, (4000, 8))
    pats[:, 0] = torch.randint(0, 3, (4000,))
    rt = torch.tensor([[3, 4], [4, 5], [7, 6]])
    da = rt[pats][:, :, 0]
    db = rt[pats][:, :, 1]
    powers = 10 ** torch.arange(8, dtype=torch.int64)
    A2 = (da * powers).sum(1)
    B2 = (db * powers).sum(1)
    with torch.no_grad():
        good = int((bulk_digits(model, A2, B2, 8) == A2 + B2).sum())

        class Frozen(type(model)):
            def forward(self, x, y):
                c = self.code()
                z = c[x] + c[y]
                pad = torch.zeros(z.shape[0], 1)
                z = torch.cat([pad, z, pad], dim=1)
                u = torch.clamp(self.gate_slope * (z.unsqueeze(-1) - self.knee), 0, 1)
                val = u[..., 1]
                key = self.key_scale * (u[..., 1] - u[..., 0])
                key = key.mean(0, keepdim=True).expand_as(key)   # content removed
                pos = torch.arange(z.shape[1])
                gap = pos.unsqueeze(-1) - pos.unsqueeze(0)
                sc = key.unsqueeze(1) + self.recency * gap
                blk = torch.full_like(gap, -1e9, dtype=sc.dtype)
                st = sc + torch.where(gap > 0, 0.0, blk)
                ic = sc + torch.where(gap >= 0, 0.0, blk)
                ci = torch.einsum("bpq,bq->bp", st.softmax(-1), val)
                co = torch.einsum("bpq,bq->bp", ic.softmax(-1), val)
                r = z + self.carry_w * ci + self.fold * co
                return (-(r.unsqueeze(-1) - c).abs())[:, 1:, :]

        frozen = Frozen()
        frozen.load_state_dict(model.state_dict())
        d1 = torch.stack([(A2 // 10 ** k) % 10 for k in range(8)], dim=1)
        d2 = torch.stack([(B2 // 10 ** k) % 10 for k in range(8)], dim=1)
        dg = frozen(d1, d2).argmax(-1)
        pw = 10 ** torch.arange(9, dtype=torch.int64)
        abl = int(((dg * pw).sum(-1) == A2 + B2).sum())
    print(f"carry-heavy accuracy: shipped {good / 4000:.4f}, "
          f"with the content term of the attention scores frozen {abl / 4000:.4f}")

    ok = (len(wrong) == 0 and bad == 0 and len(ebad) == 0)
    print("\nRESULT:", "all graded-width checks clean" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
