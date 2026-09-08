"""Independent audit of the shipped /workspace/submission.py.

Imports the graded file fresh (no training code in scope), then checks:
  * what it imports and how many parameters it registers
  * add() on random and adversarial 8-digit pairs
  * batched forward on a large uniform sample and on the held-out hash bucket
  * all 3^8 carry-class patterns
  * edge cases and other digit widths
  * that self-attention is load-bearing (freeze the content term -> collapse)
  * float32 vs float64 and CPU vs CUDA agreement
"""
import argparse, ast, importlib.util, itertools, random, sys, time
import torch


def load(path):
    spec = importlib.util.spec_from_file_location('graded_submission', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def imports_of(path):
    tree = ast.parse(open(path).read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {n.name.split('.')[0] for n in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split('.')[0])
    return sorted(names)


def tokens(a, b, n, device):
    """[B] ints -> token tensors of n places with pads."""
    p = 10 ** torch.arange(n, device=device)
    da = (a[:, None] // p) % 10
    db = (b[:, None] // p) % 10
    z = torch.zeros(len(a), 1, dtype=torch.long, device=device)
    return torch.cat([z, da, z], 1), torch.cat([z, db, z], 1)


def batch_check(model, a, b, n, device, chunk=200000):
    wrong = 0
    p = 10 ** torch.arange(n + 1, device=device)
    for i in range(0, len(a), chunk):
        aa, bb = a[i:i + chunk], b[i:i + chunk]
        da, db = tokens(aa, bb, n, device)
        with torch.no_grad():
            pred = model(da, db).argmax(-1)[:, 1:]
        got = (pred * p).sum(1)
        wrong += int((got != aa + bb).sum())
    return wrong


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', default='/workspace/submission.py')
    ap.add_argument('--n_uniform', type=int, default=2_000_000)
    ap.add_argument('--n_add', type=int, default=20000)
    a_ = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('imports:', imports_of(a_.path))
    m = load(a_.path)
    model, meta = m.build_model()
    nparam = sum(p.numel() for p in model.parameters())
    print('registered parameters:', nparam,
          {n: tuple(p.shape) for n, p in model.named_parameters()})
    print('buffers:', {n: [round(float(x), 4) for x in b.flatten()]
                       for n, b in model.named_buffers()})
    print('metadata:', meta)
    model = model.to(dev)

    g = torch.Generator(device=dev); g.manual_seed(20260907)
    LO, HI = 10_000_000, 99_999_999

    # ---- add() on the graded interface --------------------------------------
    t0 = time.time()
    rng = random.Random(1)
    bad = []
    for _ in range(a_.n_add):
        x = rng.randint(LO, HI); y = rng.randint(LO, HI)
        if m.add(model, x, y) != x + y:
            bad.append((x, y))
    print(f'add() random 8-digit: {a_.n_add - len(bad)}/{a_.n_add} correct '
          f'({time.time()-t0:.1f}s)', bad[:5])

    # ---- big batched sweep ---------------------------------------------------
    tot = 0
    for rep in range(a_.n_uniform // 500_000):
        a = torch.randint(LO, HI + 1, (500_000,), generator=g, device=dev)
        b = torch.randint(LO, HI + 1, (500_000,), generator=g, device=dev)
        tot += batch_check(model, a, b, 8, dev)
    print(f'batched uniform 8-digit: {a_.n_uniform - tot}/{a_.n_uniform} correct')

    # ---- carry-heavy: long transparent runs ---------------------------------
    N = 500_000
    da = torch.randint(0, 10, (N, 8), generator=g, device=dev)
    db = torch.where(torch.rand((N, 8), generator=g, device=dev) < 0.75,
                     9 - da, torch.randint(0, 10, (N, 8), generator=g, device=dev))
    da[:, 7] = da[:, 7].clamp(min=1); db[:, 7] = db[:, 7].clamp(min=1)
    p8 = 10 ** torch.arange(8, device=dev)
    a = (da * p8).sum(1); b = (db * p8).sum(1)
    w = batch_check(model, a, b, 8, dev)
    print(f'carry-heavy (75% transparent places): {N - w}/{N} correct')

    # ---- held-out hash bucket ------------------------------------------------
    sys.path.insert(0, '/workspace')
    import core
    ge = torch.Generator(device=dev); ge.manual_seed(31337)
    hd = core.held_set(200_000, 8, ge, dev)
    with torch.no_grad():
        pred = model(hd[0], hd[1]).argmax(-1)[:, 1:]
    ok = int((pred == hd[2][:, 1:]).all(-1).sum())
    print(f'held-out hash bucket (n=8): {ok}/200000 correct')

    # ---- all 3^8 carry-class patterns ---------------------------------------
    import certify
    got, tot8 = certify.exhaustive_patterns(model, 8, dev)
    print(f'all 3^8 carry-class patterns: {got}/{tot8} correct')

    # ---- edge cases ----------------------------------------------------------
    edges = [(10000000, 10000000), (99999999, 99999999), (99999999, 10000001),
             (19999999, 10000001), (12345678, 87654321), (10000001, 89999999),
             (55555555, 44444445), (99999999, 99999991), (50000000, 50000000),
             (11111111, 88888889), (99999998, 10000002), (45454545, 54545455)]
    bad = [(x, y) for x, y in edges if m.add(model, x, y) != x + y]
    print(f'edge cases: {len(edges)-len(bad)}/{len(edges)} correct', bad)

    # ---- other widths --------------------------------------------------------
    for n in (1, 2, 3, 5, 8, 11, 16):
        lo = 10 ** (n - 1) if n > 1 else 0
        a = torch.randint(lo, 10 ** n, (100_000,), generator=g, device=dev)
        b = torch.randint(lo, 10 ** n, (100_000,), generator=g, device=dev)
        w = batch_check(model, a, b, n, dev)
        print(f'  width {n:2d}: {100_000-w}/100000 correct')

    # ---- attention must be load-bearing --------------------------------------
    # replace the content-dependent key with its batch mean: the recency bias
    # and everything else stay exactly as they are.
    N = 20000
    da = torch.randint(0, 10, (N, 8), generator=g, device=dev)
    db = torch.where(torch.rand((N, 8), generator=g, device=dev) < 0.75,
                     9 - da, torch.randint(0, 10, (N, 8), generator=g, device=dev))
    z = torch.zeros(N, 1, dtype=torch.long, device=dev)
    DA, DB = torch.cat([z, da, z], 1), torch.cat([z, db, z], 1)
    tgt = (da * p8).sum(1) + (db * p8).sum(1)
    p9 = 10 ** torch.arange(9, device=dev)
    with torch.no_grad():
        base = ((model(DA, DB).argmax(-1)[:, 1:] * p9).sum(1) == tgt).float().mean()
        orig = model.key_w.clone()
        code = model.codes()
        x = code[DA] + code[DB]
        u = torch.clamp(model.bank_w * (x[..., None] - model.knee), 0, 1)
        keys = u @ model.key_w
        print(f'\nattention ablation: mean key {float(keys.mean()):.3f}, '
              f'distinct key values {len(torch.unique(keys.round()))}')
        model.key_w.zero_()                      # keys become a constant -> pure recency
        flat = ((model(DA, DB).argmax(-1)[:, 1:] * p9).sum(1) == tgt).float().mean()
        model.key_w.copy_(orig)
    print(f'shipped model {float(base):.4f} -> keys frozen to a constant {float(flat):.4f}')

    # ---- numeric agreement ---------------------------------------------------
    a = torch.randint(LO, HI + 1, (100_000,), generator=g, device=dev)
    b = torch.randint(LO, HI + 1, (100_000,), generator=g, device=dev)
    DA, DB = tokens(a, b, 8, dev)
    with torch.no_grad():
        p32 = model(DA, DB).argmax(-1)
        p64 = model.double()(DA, DB).argmax(-1)
        model.float()
        mc = m.build_model()[0]
        pc = mc(DA.cpu(), DB.cpu()).argmax(-1)
    print(f'float32 vs float64 agreement: {float((p32==p64).float().mean()):.6f}')
    print(f'cuda vs cpu agreement:        {float((p32.cpu()==pc).float().mean()):.6f}')


if __name__ == '__main__':
    main()
