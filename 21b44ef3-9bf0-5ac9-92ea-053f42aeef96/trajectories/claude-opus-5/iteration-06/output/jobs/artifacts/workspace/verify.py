"""Audit whatever currently sits at /workspace/submission.py.

Checks accuracy on large held-out samples, on carry-structure corner cases and
on every one of the 3^8 carry patterns; confirms add() agrees with the batched
forward; measures the decision margin; and ablates the attention to show it is
load bearing rather than a fixed pattern.
"""

import argparse, importlib.util, itertools, math, random, sys

import torch

import data


def load(path="/workspace/submission.py"):
    spec = importlib.util.spec_from_file_location("submission", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def digits_to_int(d):
    return sum(int(v) * 10 ** i for i, v in enumerate(d))


@torch.no_grad()
def batched_pred(model, da, db, chunk=65536):
    out = []
    for i in range(0, da.shape[0], chunk):
        out.append(model(da[i:i + chunk], db[i:i + chunk]).argmax(-1)[:, 1:])
    return torch.cat(out)


def acc_on(model, da, db, y):
    return (batched_pred(model, da, db) == y).all(-1).float().mean().item()


# ------------------------------------------------------------------ suites --

def uniform_suite(model, n, dev, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    tot = 0
    got = 0
    while tot < n:
        k = min(200000, n - tot)
        da, db, y = data.batch(k, dev, g, train=False, mix=((1.0, None, None),))
        got += (batched_pred(model, da, db) == y).all(-1).sum().item()
        tot += k
    return got / tot


def hard_suite(model, n, dev, seed=1):
    g = torch.Generator(device=dev).manual_seed(seed)
    mix = ((0.25, 0.30, 0.40), (0.25, 0.10, 0.80), (0.25, 0.02, 0.96), (0.25, 0.0, 1.0))
    da, db, y = data.batch(n, dev, g, train=False, mix=mix)
    return acc_on(model, da, db, y)


def pattern_suite(model, dev, reps=8, seed=0):
    """Every one of the 3^8 assignments of {absorb, transparent, generate} to
    the eight places, each filled `reps` ways."""
    rnd = random.Random(seed)
    A, B = [], []
    for pat in itertools.product((0, 1, 2), repeat=8):
        for _ in range(reps):
            a, b = [], []
            for i, c in enumerate(pat):
                lo = 1 if i == 7 else 0
                while True:
                    if c == 1:
                        x = rnd.randint(lo, 9 - lo if lo else 9)
                        y_ = 9 - x
                    elif c == 0:
                        s = rnd.randint(2 * lo, 8)
                        x = rnd.randint(lo, s - lo)
                        y_ = s - x
                    else:
                        s = rnd.randint(10, 18)
                        x = rnd.randint(max(lo, s - 9), 9)
                        y_ = s - x
                    if x >= lo and y_ >= lo:
                        break
                a.append(x)
                b.append(y_)
            A.append(a)
            B.append(b)
    a = torch.tensor(A, device=dev)
    b = torch.tensor(B, device=dev)
    z = torch.zeros_like(a[:, :1])
    da, db = torch.cat([z, a, z], 1), torch.cat([z, b, z], 1)
    y = data.labels(a, b)
    pred = batched_pred(model, da, db)
    ok = (pred == y).all(-1)
    bad = (~ok).nonzero().flatten()[:5].tolist()
    return ok.float().mean().item(), [(digits_to_int(A[i]), digits_to_int(B[i])) for i in bad]


def edge_cases():
    v = [10000000, 99999999, 10000001, 99999998, 12345678, 87654321,
         19999999, 10000009, 55555555, 44444444, 45454545, 54545454,
         11111111, 88888888, 99999990, 90000000, 50000000, 49999999]
    out = [(x, y) for x in v for y in v]
    for k in range(8):                      # carry chains of every length
        a = 10 ** 7 + (10 ** k - 1)         # 10000000, 10000009, ..., 19999999
        b = 10 ** 7 + 1
        out.append((a, b))
        out.append((b, a))
    assert all(10000000 <= x <= 99999999 and 10000000 <= y <= 99999999
               for x, y in out), "edge cases must stay in the graded range"
    return out


@torch.no_grad()
def margin(model, dev, n=200000, seed=3):
    g = torch.Generator(device=dev).manual_seed(seed)
    mix = ((0.5, None, None), (0.25, 0.10, 0.80), (0.25, 0.02, 0.96))
    da, db, y = data.batch(n, dev, g, train=False, mix=mix)
    lg = []
    for i in range(0, n, 65536):
        lg.append(model(da[i:i + 65536], db[i:i + 65536])[:, 1:])
    lg = torch.cat(lg)
    top2 = lg.topk(2, -1)
    correct = top2.indices[..., 0] == y
    gap = (top2.values[..., 0] - top2.values[..., 1])
    return gap.min().item(), correct.all(-1).float().mean().item()


# --------------------------------------------------------------- ablations --

@torch.no_grad()
def attention_report(model, dev, n=4096, seed=5):
    """Fraction of inputs whose attention argmax pattern differs, and accuracy
    when the attention map is frozen to its mean over the batch."""
    g = torch.Generator(device=dev).manual_seed(seed)
    mix = ((0.4, None, None), (0.3, 0.10, 0.80), (0.3, 0.02, 0.96))
    da, db, y = data.batch(n, dev, g, train=False, mix=mix)

    grabbed = []
    real_softmax = torch.softmax

    def spy(t, dim=None, **kw):
        out = real_softmax(t, dim, **kw)
        if out.dim() >= 3 and out.shape[-1] == out.shape[-2]:
            grabbed.append(out.detach())
        return out

    torch.softmax = spy
    try:
        model(da, db)
    finally:
        torch.softmax = real_softmax
    attn = grabbed[0]                                   # (B, H, P, P)
    pattern = attn.argmax(-1)                           # (B, H, P)
    uniq = torch.unique(pattern.reshape(pattern.shape[0], -1), dim=0).shape[0]
    modal = pattern.mode(0).values
    varies = (pattern != modal[None]).any(-1).any(-1).float().mean().item()
    frozen = attn.mean(0, keepdim=True)

    def fixed(t, dim=None, **kw):
        out = real_softmax(t, dim, **kw)
        if out.dim() >= 3 and out.shape[-1] == out.shape[-2]:
            return frozen.expand_as(out)
        return out

    torch.softmax = fixed
    try:
        pred = model(da, db).argmax(-1)[:, 1:]
    finally:
        torch.softmax = real_softmax
    return uniq, varies, (pred == y).all(-1).float().mean().item()


# -------------------------------------------------------------------- main --

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="/workspace/submission.py")
    ap.add_argument("--n", type=int, default=1000000)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--dev", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    sub = load(a.path)
    model, meta = sub.build_model()
    n_par = sum(p.numel() for p in model.parameters())
    n_buf = sum(b.numel() for b in model.buffers())
    print(f"parameters {n_par}   buffers {n_buf}   meta says {meta.get('n_params')}")
    print("param shapes:", {k: tuple(v.shape) for k, v in model.named_parameters()})

    model = model.to(a.dev)
    print(f"uniform held-out  n={a.n}: {uniform_suite(model, a.n, a.dev):.6f}")
    print(f"carry-heavy mix   n=200000: {hard_suite(model, 200000, a.dev):.6f}")
    pat, bad = pattern_suite(model, a.dev, a.reps)
    print(f"all 6561 carry patterns x{a.reps}: {pat:.6f}  first failures {bad}")

    cpu = model.to("cpu")
    ec = edge_cases()
    wrong = [(x, y) for x, y in ec if sub.add(cpu, x, y) != x + y]
    print(f"edge cases {len(ec)}: {len(ec)-len(wrong)} correct  failures {wrong[:5]}")

    rnd = random.Random(7)
    pairs = [(rnd.randint(10000000, 99999999), rnd.randint(10000000, 99999999))
             for _ in range(2000)]
    bad_api = [(x, y) for x, y in pairs if sub.add(cpu, x, y) != x + y]
    print(f"add() on 2000 random pairs: {2000-len(bad_api)} correct  failures {bad_api[:5]}")

    model = model.to(a.dev)
    m, macc = margin(model, a.dev)
    print(f"min top1-top2 logit gap: {m:.6g}  (acc on that sample {macc:.6f})")

    uniq, varies, frozen_acc = attention_report(model, a.dev)
    print(f"attention: {uniq} distinct argmax patterns over 4096 inputs; "
          f"{varies:.3f} differ from the modal one; accuracy with the map frozen "
          f"at its batch mean: {frozen_acc:.6f}")

    d64 = model.double()
    g = torch.Generator(device=a.dev).manual_seed(11)
    da, db, y = data.batch(100000, a.dev, g, train=False,
                           mix=((0.5, None, None), (0.5, 0.02, 0.96)))
    p64 = batched_pred(d64, da, db)
    print(f"float64 rerun agrees with labels: {(p64 == y).all(-1).float().mean().item():.6f}")


if __name__ == "__main__":
    main()
