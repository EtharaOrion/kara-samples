"""Independent end-to-end check of /workspace/submission.py.

Deliberately does not import anything from the training code: it loads the
graded file the way the grader would, calls the documented interface, and
checks the answers against Python's own arithmetic.
"""
import argparse, ast, importlib.util, itertools, json, random, sys, time
import torch


def load(path):
    spec = importlib.util.spec_from_file_location("graded_submission", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def imports_of(path):
    tree = ast.parse(open(path).read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return sorted(names)


def digits_to_int(d):
    return sum(v * 10 ** i for i, v in enumerate(d))


def tokenize(pairs, n=8):
    A = torch.tensor([[0] + [(a // 10 ** i) % 10 for i in range(n)] + [0]
                      for a, _ in pairs])
    B = torch.tensor([[0] + [(b // 10 ** i) % 10 for i in range(n)] + [0]
                      for _, b in pairs])
    return A, B


def undigit(rows):
    """Little-endian digit rows -> Python ints (exact at any width)."""
    return [sum(v * 10 ** i for i, v in enumerate(r)) for r in rows]


def batched_predict(model, pairs, n=8, chunk=100_000):
    """Same decoding as `add`, run in batches."""
    A, B = tokenize(pairs, n)
    out = []
    with torch.no_grad():
        for i in range(0, len(pairs), chunk):
            d = model(A[i:i + chunk], B[i:i + chunk]).argmax(-1)[:, 1:]
            out += undigit(d.tolist())
    return out


def check(model, pairs, label, n=8):
    got = batched_predict(model, pairs, n)
    bad = [(a, b, g) for (a, b), g in zip(pairs, got) if a + b != g]
    print(f"  {label:<34} {len(pairs)-len(bad):>9d}/{len(pairs):<9d} exact"
          + ("" if not bad else f"   FIRST BAD {bad[0]}"))
    return len(bad), bad[:5]


def heldout_bucket(a, b, n=8):
    """The training split rule, copied from data.py so this file stays
    standalone: bucket 0 of 16 is never trained on."""
    h = 0
    for i in range(n):
        h = (h * 1000003 + ((a // 10 ** i) % 10) * 10 + (b // 10 ** i) % 10) \
            & 0x3FFFFFFFFFFF
    h = (h ^ (h >> 17)) * 2654435761
    return (h & 0x3FFFFFFFFFFF) % 16 == 0


def rand_pairs_uniform(N, rng, n=8):
    lo, hi = 10 ** (n - 1), 10 ** n - 1
    return [(rng.randint(lo, hi), rng.randint(lo, hi)) for _ in range(N)]


def rand_pairs_carry(N, rng, n=8, p=0.55):
    out = []
    for _ in range(N):
        da, db = [], []
        for i in range(n):
            a = rng.randint(0, 9)
            b = 9 - a if rng.random() < p else rng.randint(0, 9)
            da.append(a); db.append(b)
        da[-1] = max(da[-1], 1); db[-1] = max(db[-1], 1)
        out.append((digits_to_int(da), digits_to_int(db)))
    return out


def all_class_patterns(rng, n=8):
    """One concrete operand pair for each of the 3^n carry-class patterns."""
    absorb = [(a, b) for a in range(10) for b in range(10) if a + b <= 8]
    trans = [(a, 9 - a) for a in range(10)]
    gen = [(a, b) for a in range(10) for b in range(10) if a + b >= 10]
    pools = [absorb, trans, gen]
    out = []
    for pat in itertools.product(range(3), repeat=n):
        da, db = [], []
        for i, c in enumerate(pat):
            pool = pools[c]
            if i == n - 1:
                pool = [q for q in pool if q[0] >= 1 and q[1] >= 1]
            a, b = pool[rng.randrange(len(pool))]
            da.append(a); db.append(b)
        out.append((digits_to_int(da), digits_to_int(db)))
    return out


def ablate(model, pairs, mode, n=8):
    """Re-run the block with the attention crippled in one specific way.

    mode='mean'  freeze the attention map at its average over the batch.  If
                 the map were really a fixed function of position, this would
                 change nothing.
    mode='pos'   drop the content term from the logits and keep only the
                 positional (recency) bias.  This is what "a fixed pattern
                 dressed up as attention" would look like; every query would
                 then attend to p-1 regardless of the digits.
    mode='none'  the model as shipped, as a control.
    """
    A, B = tokenize(pairs, n)
    with torch.no_grad():
        code = model.codes()
        x = code[A] + code[B]
        g = torch.clamp(model.bank_w * x.unsqueeze(-1) + model.knee, 0.0, 1.0)
        k = (g * model.key_w).sum(-1)
        v = (g * model.val_w).sum(-1)
        P = x.shape[-1]
        p = torch.arange(P)
        dist = p[:, None] - p[None, :]
        content = torch.zeros_like(k) if mode == "pos" else k
        logit = content.unsqueeze(1) + model.lam * dist
        floor = torch.full_like(logit, -1e30)
        a_in = torch.softmax(torch.where(dist > 0, logit, floor), -1)
        a_out = torch.softmax(torch.where(dist >= 0, logit, floor), -1)
        route = a_in.argmax(-1)                       # who each query reads
        distinct = len({tuple(r) for r in route.tolist()})
        hop = (torch.arange(route.shape[1])[None, :] - route)[:, 1:]
        if mode == "mean":
            a_in = a_in.mean(0, keepdim=True).expand_as(a_in)
            a_out = a_out.mean(0, keepdim=True).expand_as(a_out)
        y = x + model.carry_w * (a_in * v.unsqueeze(1)).sum(-1) \
              + model.fold * (a_out * v.unsqueeze(1)).sum(-1)
        d = (-(y.unsqueeze(-1) - code) ** 2).argmax(-1)[:, 1:]
    got = undigit(d.tolist())
    hit = sum(1 for (a, b), gv in zip(pairs, got) if a + b == gv)
    return hit / len(pairs), distinct, int(hop.max()), float(hop.float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="/workspace/submission.py")
    ap.add_argument("--n_uniform", type=int, default=1_000_000)
    ap.add_argument("--n_add", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=20260907)
    args = ap.parse_args()

    print(f"file      {args.path}")
    print(f"imports   {imports_of(args.path)}")
    mod = load(args.path)
    model, meta = mod.build_model()
    npar = sum(p.numel() for p in model.parameters())
    nbuf = sum(b.numel() for b in model.buffers())
    print(f"module    {type(model).__name__}, "
          f"is nn.Module = {isinstance(model, torch.nn.Module)}")
    print(f"params    {npar} registered parameters "
          f"({[ (n, tuple(p.shape)) for n, p in model.named_parameters() ]})")
    print(f"buffers   {nbuf} values in buffers "
          f"({[ (n, tuple(b.shape)) for n, b in model.named_buffers() ]})")
    print(f"metadata  {json.dumps(meta)[:200]}")

    rng = random.Random(args.seed)
    fails = 0

    print("\nheld-out accuracy (all through one forward pass of the model):")
    big = rand_pairs_uniform(args.n_uniform, rng)
    nb, _ = check(model, big, f"uniform 8-digit x{args.n_uniform}")
    fails += nb
    ho = [p for p in big if heldout_bucket(*p)]
    nb, _ = check(model, ho, f"of those, the held-out bucket only")
    fails += nb
    carry = rand_pairs_carry(200_000, rng)
    nb, _ = check(model, carry, "carry-enriched x200000")
    fails += nb
    nb, _ = check(model, [p for p in carry if heldout_bucket(*p)],
                  "of those, the held-out bucket only")
    fails += nb
    nb, _ = check(model, all_class_patterns(rng), "all 3^8 carry patterns")
    fails += nb

    edge = [(10 ** 7, 10 ** 7), (99999999, 99999999), (99999999, 10000001),
            (19999999, 10000001), (10000000, 99999999), (55555555, 44444445),
            (12345678, 87654321), (50000000, 50000000), (99999999, 99999998),
            (11111111, 88888889), (45454545, 54545455), (10000001, 89999999)]
    nb, _ = check(model, edge, "hand-picked edge cases")
    fails += nb

    print("\nthe documented interface, one call at a time:")
    t0 = time.time()
    sample = rand_pairs_uniform(args.n_add, rng)
    bad = [(a, b) for a, b in sample if mod.add(model, a, b) != a + b]
    print(f"  add(model, a, b) x{args.n_add:<19d} "
          f"{args.n_add-len(bad):>9d}/{args.n_add:<9d} exact"
          + (f"   FIRST BAD {bad[0]}" if bad else "")
          + f"   ({time.time()-t0:.1f}s)")
    fails += len(bad)

    print("\nis the attention doing real work?")
    probe = rand_pairs_carry(4096, rng)
    for mode, label in (("none", "as shipped"),
                        ("mean", "map frozen at its batch mean"),
                        ("pos", "content term dropped, recency bias kept")):
        acc, distinct, mx, mean = ablate(model, probe, mode)
        print(f"  {label:<40} accuracy {acc:.4f}   "
              f"{distinct:>4d} distinct routings   "
              f"hop max {mx}, mean {mean:.2f}")

    print("\nwidths other than 8 (the graded operands are always 8 digits;"
          " this is head-room):")
    for n in (1, 2, 4, 8, 12, 16, 20, 24, 32):
        lo, hi = 10 ** (n - 1), 10 ** n - 1
        wp = [(rng.randint(lo, hi), rng.randint(lo, hi)) for _ in range(4000)]
        wp += rand_pairs_carry(4000, rng, n=n)
        got = batched_predict(model, wp, n=n)
        ok = sum(1 for (a, b), g in zip(wp, got) if a + b == g)
        print(f"  n={n:<3d} {ok:>6d}/{len(wp):<6d} exact")

    print("\nfloat64 agreement:")
    m64 = mod.build_model()[0].double()
    got32 = batched_predict(model, edge)
    got64 = batched_predict(m64, edge)
    print(f"  float32 vs float64 on edge cases      "
          f"{'identical' if got32 == got64 else 'DIFFER'}")

    print(f"\nRESULT: {'PASS' if fails == 0 else f'FAIL ({fails} wrong)'}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
