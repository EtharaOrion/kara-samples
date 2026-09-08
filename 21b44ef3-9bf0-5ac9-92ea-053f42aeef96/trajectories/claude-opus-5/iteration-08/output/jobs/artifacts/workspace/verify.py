"""Audit /workspace/submission.py the way the grader would, plus extra probes.

  python verify.py --n 1000000
"""
import argparse, importlib, random, sys
import torch


def digits(x, n):
    return [(x // 10 ** i) % 10 for i in range(n)]


def batched_add(model, A, B, n=8):
    """Same computation add() does, run on a whole batch at once."""
    dev = next(model.parameters()).device
    da = torch.stack([torch.tensor([0] + digits(int(v), n) + [0]) for v in A]).to(dev)
    db = torch.stack([torch.tensor([0] + digits(int(v), n) + [0]) for v in B]).to(dev)
    with torch.no_grad():
        d = model(da, db).argmax(-1)[:, 1:]
    pw = 10 ** torch.arange(n + 1, device=dev, dtype=torch.int64)
    return (d * pw).sum(-1)


def gpu_batch(model, a, b):
    """a,b: (B,n) digit tensors -> predicted answer digits (B,n+1)."""
    dev = next(model.parameters()).device
    z = torch.zeros(a.shape[0], 1, dtype=torch.int64, device=dev)
    da = torch.cat([z, a.to(dev), z], 1)
    db = torch.cat([z, b.to(dev), z], 1)
    with torch.no_grad():
        return model(da, db).argmax(-1)[:, 1:]


def true_digits(a, b):
    n = a.shape[1]
    s = a + b
    out = torch.zeros(a.shape[0], n + 1, dtype=torch.int64, device=a.device)
    c = torch.zeros(a.shape[0], dtype=torch.int64, device=a.device)
    for i in range(n):
        t = s[:, i] + c
        out[:, i] = t % 10
        c = t // 10
    out[:, n] = c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--file", default="/workspace/submission.py")
    args = ap.parse_args()

    sys.path.insert(0, "/workspace")
    sub = importlib.import_module("submission")
    importlib.reload(sub)
    model, meta = sub.build_model()
    npar = sum(p.numel() for p in model.parameters())
    nbuf = sum(b.numel() for b in model.buffers())
    print(f"parameters: {npar}   buffers: {nbuf}   metadata param_count: {meta.get('param_count')}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)

    # --- scalar path: add() itself, on random full-width pairs ---
    rng = random.Random(0)
    bad = []
    for _ in range(2000):
        a = rng.randint(10 ** 7, 10 ** 8 - 1)
        b = rng.randint(10 ** 7, 10 ** 8 - 1)
        if sub.add(model, a, b) != a + b:
            bad.append((a, b))
    print(f"add() on 2000 random pairs: {2000 - len(bad)}/2000" + (f"  e.g. {bad[:3]}" if bad else ""))

    # --- bulk: same decode, batched ---
    g = torch.Generator(device=dev).manual_seed(7)
    tot = ok = 0
    worst = []
    for _ in range(max(1, args.n // 16384)):
        a = torch.randint(0, 10, (16384, 8), device=dev, generator=g)
        b = torch.randint(0, 10, (16384, 8), device=dev, generator=g)
        a[:, -1] = a[:, -1].clamp(min=1)
        b[:, -1] = b[:, -1].clamp(min=1)
        pred = gpu_batch(model, a, b)
        t = true_digits(a, b)
        good = (pred == t).all(1)
        ok += int(good.sum()); tot += good.numel()
        if (~good).any() and len(worst) < 5:
            i = (~good).nonzero()[0, 0]
            worst.append((a[i].tolist(), b[i].tolist()))
    print(f"uniform full-width: {ok}/{tot} = {ok/tot:.6f}" + (f"  fails {worst}" if worst else ""))

    # --- carry-chain stress: every pattern of generate/transparent/absorb ---
    pats = torch.tensor([[(k // 3 ** i) % 3 for i in range(8)] for k in range(3 ** 8)], device=dev)
    accs = []
    for rep in range(8):
        gg = torch.Generator(device=dev).manual_seed(100 + rep)
        a = torch.randint(0, 10, pats.shape, device=dev, generator=gg)
        b = torch.randint(0, 10, pats.shape, device=dev, generator=gg)
        b = torch.where(pats == 0, torch.clamp(8 - a, min=0), b)          # absorb  a+b<=8
        a = torch.where(pats == 1, a, a)
        b = torch.where(pats == 1, 9 - a, b)                              # transparent a+b==9
        lo = torch.clamp(10 - a, min=0, max=9)
        b = torch.where(pats == 2, torch.maximum(b, lo), b)               # generate a+b>=10
        a[:, -1] = a[:, -1].clamp(min=1)
        b[:, -1] = b[:, -1].clamp(min=1)
        b = torch.where(pats == 1, 9 - a, b)
        b = torch.where(pats == 2, torch.maximum(b, torch.clamp(10 - a, min=0, max=9)), b)
        b = torch.where(pats == 0, torch.clamp(torch.minimum(b, 8 - a), min=0), b)
        pred = gpu_batch(model, a, b)
        accs.append(float((pred == true_digits(a, b)).all(1).float().mean()))
    print(f"all 6561 carry patterns x8 draws: min={min(accs):.6f} mean={sum(accs)/len(accs):.6f}")

    # --- edges ---
    edge = [(10 ** 7, 10 ** 7), (99999999, 99999999), (99999999, 10000001),
            (19999999, 10000001), (10000001, 19999999), (12345678, 87654321),
            (55555555, 44444445), (99999999, 99999991), (10000000, 99999999),
            (11111111, 88888889), (50000000, 50000000), (98765432, 12345678)]
    ebad = [(a, b) for a, b in edge if sub.add(model, a, b) != a + b]
    print(f"edge cases: {len(edge) - len(ebad)}/{len(edge)}" + (f"  fail {ebad}" if ebad else ""))

    # --- attention really matters: freeze the map at its batch mean ---
    import types
    a = torch.randint(0, 10, (4096, 8), device=dev, generator=g)
    b = torch.randint(0, 10, (4096, 8), device=dev, generator=g)
    a[:, -1] = a[:, -1].clamp(min=1); b[:, -1] = b[:, -1].clamp(min=1)
    base = float((gpu_batch(model, a, b) == true_digits(a, b)).all(1).float().mean())
    orig = torch.softmax

    class Frozen:
        def __init__(self): self.mean = None
        def __call__(self, x, dim=None, **kw):
            w = orig(x, dim, **kw)
            if w.dim() == 3 and w.shape[-1] == w.shape[-2]:
                if self.mean is None:
                    self.mean = w.mean(0, keepdim=True)
                return self.mean.expand_as(w)
            return w
    f = Frozen()
    torch.softmax = f
    try:
        _ = gpu_batch(model, a[:1024], b[:1024])          # sets the mean
        froz = float((gpu_batch(model, a, b) == true_digits(a, b)).all(1).float().mean())
    finally:
        torch.softmax = orig
    print(f"attention ablation: live={base:.4f}  frozen-at-mean={froz:.4f}")

    # --- how many distinct attention maps does it actually compute? ---
    print("metadata:", meta)


if __name__ == "__main__":
    main()
