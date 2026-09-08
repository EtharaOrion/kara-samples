"""Independent audit of the graded file /workspace/submission.py.

Deliberately shares no code with the training pipeline: it imports the built
file, re-derives ground truth with Python integer arithmetic, and reports what
it measures.  Every number quoted about the submission comes from here.
"""

import argparse
import importlib.util
import random
import sys

import torch


def load(path):
    spec = importlib.util.spec_from_file_location("graded_submission", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["graded_submission"] = mod
    spec.loader.exec_module(mod)
    return mod


def param_report(model):
    rows = [(n, tuple(p.shape), p.numel(), bool(p.requires_grad))
            for n, p in model.named_parameters()]
    total = sum(r[2] for r in rows)
    buffers = [(n, tuple(b.shape), b.tolist()) for n, b in model.named_buffers()]
    return rows, total, buffers


def batched_add(mod, model, pairs):
    """Run add() one pair at a time -- exactly the graded interface."""
    wrong = []
    for a, b in pairs:
        got = mod.add(model, a, b)
        if got != a + b:
            wrong.append((a, b, got, a + b))
    return wrong


def sweep(mod, model, n, rng, lo=10 ** 7, hi=10 ** 8 - 1, chunk=8192):
    """Fast bulk check that drives the model's forward pass directly, one
    forward per batch, then compares against Python integer sums."""
    dev = next(model.parameters()).device
    bad = 0
    seen = 0
    while seen < n:
        m = min(chunk, n - seen)
        a = [rng.randint(lo, hi) for _ in range(m)]
        b = [rng.randint(lo, hi) for _ in range(m)]
        width = 8
        rows = torch.zeros(m, width + 2, 2, dtype=torch.long)
        for k in range(width):
            rows[:, k + 1, 0] = torch.tensor([(x // 10 ** k) % 10 for x in a])
            rows[:, k + 1, 1] = torch.tensor([(x // 10 ** k) % 10 for x in b])
        with torch.no_grad():
            pred = model(rows.to(dev)).argmax(-1)[:, 1:].cpu()
        pw = torch.tensor([10 ** k for k in range(width + 1)], dtype=torch.long)
        got = (pred * pw).sum(1)
        want = torch.tensor([x + y for x, y in zip(a, b)], dtype=torch.long)
        bad += int((got != want).sum())
        seen += m
    return bad, seen


def carry_patterns(mod, model):
    """All 3^8 absorb/transparent/generate class patterns, one representative
    problem each, with the most significant place forced non-zero."""
    rng = random.Random(20260907)
    cases = []
    for code in range(3 ** 8):
        da, db, c = [], [], code
        for k in range(8):
            cls = c % 3
            c //= 3
            top = k == 7
            if cls == 0:                       # absorb: a+b <= 8
                s = rng.randint(2 if top else 0, 8)
                x = rng.randint(1 if top else 0, min(9, s) - (1 if top and s >= 1 else 0))
                x = max(x, 1) if top else x
                x = min(x, s)
                y = s - x
                if top and (x == 0 or y == 0):
                    x, y = max(1, min(x, s - 1)), s - max(1, min(x, s - 1))
            elif cls == 1:                     # transparent: a+b == 9
                x = rng.randint(1, 8) if top else rng.randint(0, 9)
                y = 9 - x
            else:                              # generate: a+b >= 10
                x = rng.randint(1, 9)
                y = rng.randint(10 - x, 9)
            da.append(x)
            db.append(y)
        a = sum(d * 10 ** k for k, d in enumerate(da))
        b = sum(d * 10 ** k for k, d in enumerate(db))
        if a < 10 ** 7 or b < 10 ** 7:
            continue
        cases.append((a, b))
    return cases


def edge_cases():
    picks = [10 ** 7, 10 ** 8 - 1, 19999999, 10000001, 99999999, 50000000,
             12345678, 87654321, 11111111, 88888888, 10000000, 90000000,
             99999998, 10000002, 55555555, 44444445]
    out = []
    for a in picks:
        for b in picks:
            out.append((a, b))
    return out


def attention_ablation(mod, model, cases):
    """Replace the content-dependent part of the key with its batch mean, so
    attention becomes a fixed distance-only pattern, and re-measure."""
    dev = next(model.parameters()).device
    rows = torch.zeros(len(cases), 10, 2, dtype=torch.long)
    for i, (a, b) in enumerate(cases):
        for k in range(8):
            rows[i, k + 1, 0] = (a // 10 ** k) % 10
            rows[i, k + 1, 1] = (b // 10 ** k) % 10
    rows = rows.to(dev)
    want = torch.tensor([a + b for a, b in cases])

    def score(tokens, freeze):
        with torch.no_grad():
            code = model.code
            x = code[tokens[..., 0]] + code[tokens[..., 1]]
            gate = torch.clamp(model.bank_w * (x.unsqueeze(-1) - model.knee), 0.0, 1.0)
            value = gate[..., 1]
            key = model.key_w * (gate[..., 1] - gate[..., 0])
            if freeze:
                key = key.mean(dim=(0, 1), keepdim=True).expand_as(key)
            p = x.shape[-1]
            pos = torch.arange(p, device=x.device)
            delta = pos.unsqueeze(1) - pos.unsqueeze(0)
            logits = key.unsqueeze(-2) + model.lam * delta
            blocked = torch.finfo(logits.dtype).min
            earlier = (delta > 0) | ((pos.unsqueeze(1) == 0) & (pos.unsqueeze(0) == 0))
            upto = delta >= 0
            w_in = torch.softmax(torch.where(earlier, logits, blocked), -1)
            w_out = torch.softmax(torch.where(upto, logits, blocked), -1)
            stream = (x + model.carry_w * (w_in * value.unsqueeze(-2)).sum(-1)
                      + model.fold * (w_out * value.unsqueeze(-2)).sum(-1))
            pred = (-(stream.unsqueeze(-1) - code) ** 2).argmax(-1)[:, 1:].cpu()
        pw = torch.tensor([10 ** k for k in range(9)], dtype=torch.long)
        return float(((pred * pw).sum(1) == want).float().mean())

    return score(rows, False), score(rows, True)


def robustness(mod, model):
    """The graded interface is add(model, a, b); check it survives the ways an
    unseen evaluator might reasonably hold it."""
    import copy
    rng = random.Random(99)
    pairs = [(rng.randint(10 ** 7, 10 ** 8 - 1), rng.randint(10 ** 7, 10 ** 8 - 1))
             for _ in range(200)]
    out = []

    def ok(m):
        return all(mod.add(m, a, b) == a + b for a, b in pairs)

    out.append(("float64 model", ok(copy.deepcopy(model).double())))
    out.append(("float16 model", ok(copy.deepcopy(model).half())))
    out.append(("left in training mode", ok(copy.deepcopy(model).train())))
    rt = mod.build_model()[0]
    rt.load_state_dict(copy.deepcopy(model).state_dict())
    out.append(("state_dict round trip", ok(rt)))
    with torch.inference_mode():
        out.append(("under inference_mode", ok(model)))

    # Not graded, but a fixed-pattern "attention" could not do this: the block has
    # no position-specific parameters, so other widths must work unchanged.
    widths = []
    for w in (1, 2, 3, 5, 12, 20, 40):
        lo, hi = 10 ** (w - 1), 10 ** w - 1
        g = random.Random(w)
        pp = [(g.randint(lo, hi), g.randint(lo, hi)) for _ in range(300)]
        widths.append((w, sum(mod.add(model, a, b) == a + b for a, b in pp), len(pp)))
    return out, widths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="/workspace/submission.py")
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    mod = load(a.path)
    model, meta = mod.build_model()
    model = model.to(a.device)
    rows, total, buffers = param_report(model)

    print("=" * 66)
    print("AUDIT OF", a.path)
    print("=" * 66)
    print("parameters:")
    for n, shape, num, grad in rows:
        print(f"   {n:12s} {str(shape):8s} {num:3d}  requires_grad={grad}")
    print(f"   TOTAL {total}")
    print("buffers (not counted, architectural constants):")
    for n, shape, val in buffers:
        print(f"   {n:12s} {str(shape):8s} {val}")
    print("metadata:", meta)

    rng = random.Random(1234)
    bad, seen = sweep(mod, model, a.n, rng)
    print(f"\nuniform 8-digit pairs: {seen - bad}/{seen} exact  "
          f"(accuracy {1 - bad / seen:.6f})")

    pairs = [(rng.randint(10 ** 7, 10 ** 8 - 1), rng.randint(10 ** 7, 10 ** 8 - 1))
             for _ in range(3000)]
    w = batched_add(mod, model, pairs)
    print(f"add() interface, one call per pair: {len(pairs) - len(w)}/{len(pairs)} exact")
    if w:
        print("   examples:", w[:5])

    cases = carry_patterns(mod, model)
    w = batched_add(mod, model, cases)
    print(f"carry class patterns (3^8 representatives): "
          f"{len(cases) - len(w)}/{len(cases)} exact")
    if w:
        print("   examples:", w[:5])

    ec = edge_cases()
    w = batched_add(mod, model, ec)
    print(f"edge cases: {len(ec) - len(w)}/{len(ec)} exact")
    if w:
        print("   examples:", w[:5])

    rob, widths = robustness(mod, model)
    print("\nrobustness of the graded interface:")
    for name, good in rob:
        print("   %-24s %s" % (name, "ok" if good else "FAILED"))
    print("digit widths never trained on (block has no position-specific weights):")
    print("   " + "  ".join("%dd %d/%d" % (w, g, n) for w, g, n in widths))

    live, frozen = attention_ablation(mod, model, cases)
    print(f"\nattention ablation on carry-heavy inputs: live {live:.4f} -> "
          f"content-free key {frozen:.4f}")

    print("\nimports in the graded file:")
    for line in open(a.path):
        if line.startswith("import ") or line.startswith("from "):
            print("   ", line.rstrip())


if __name__ == "__main__":
    main()
