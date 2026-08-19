"""Grader-style check of /workspace/submission.py: param count, accuracy, edge cases."""
import importlib.util, random, sys, time, torch

spec = importlib.util.spec_from_file_location("sub", sys.argv[1] if len(sys.argv) > 1 else "/workspace/submission.py")
sub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub)

model, meta = sub.build_model()
npar = sum(p.numel() for p in model.parameters())
print("parameters:", npar, "| metadata:", meta.get("n_parameters"))
print("param names:", [(n, tuple(p.shape)) for n, p in model.named_parameters()])

LIM = 99_999_999_999_999
rng = random.Random(0)

# 1) exact add() on random pairs
t0 = time.time()
bad = []
N = 4000
for _ in range(N):
    a, b = rng.randint(0, LIM), rng.randint(0, LIM)
    if sub.add(model, a, b) != a + b:
        bad.append((a, b))
print(f"random add(): {N - len(bad)}/{N} correct  ({time.time()-t0:.1f}s)")

# 2) edge cases
edges = [(0, 0), (0, 1), (1, 0), (LIM, LIM), (LIM, 1), (1, LIM), (99999999999999, 0),
         (50000000000000, 50000000000000), (12345678901234, 98765432109876),
         (9, 1), (99, 1), (999999999999, 1), (10000000000000, 10000000000000),
         (11111111111111, 88888888888889), (55555555555555, 44444444444445),
         (1, 999999999999), (123, 456), (0, LIM), (LIM, 0), (98765432109876, 12345678901234)]
ebad = [(a, b) for a, b in edges if sub.add(model, a, b) != a + b]
print(f"edge cases: {len(edges)-len(ebad)}/{len(edges)} correct", ebad[:5])

# 3) large batched sweep through forward() (equivalent path, much faster)
dev = next(model.parameters()).device
g = torch.Generator(device="cpu").manual_seed(7)


def digits(x, n=15):
    return torch.stack([(x // 10 ** i) % 10 for i in range(n)], 1)


def sweep(name, sampler, total=2_000_000, bs=100_000):
    ok = 0
    for _ in range(total // bs):
        A, B = sampler(bs)
        S = A + B
        ad = torch.cat([torch.zeros(bs, 1, dtype=torch.long), digits(A)], 1)
        bd = torch.cat([torch.zeros(bs, 1, dtype=torch.long), digits(B)], 1)
        pred = model(ad.to(dev), bd.to(dev)).argmax(-1).cpu()
        ok += (pred == digits(S)).all(1).sum().item()
    print(f"{name}: {ok}/{total} = {ok/total:.6f}")
    return ok / total


def uni(n):
    return (torch.randint(0, LIM + 1, (n,), generator=g), torch.randint(0, LIM + 1, (n,), generator=g))


def stress(n):
    ad = torch.randint(0, 10, (n, 14), generator=g)
    rate = 0.3 + 0.7 * torch.rand(n, 1, generator=g)
    m = torch.rand(n, 14, generator=g) < rate
    bd = torch.where(m, 9 - ad, torch.randint(0, 10, (n, 14), generator=g))
    p10 = 10 ** torch.arange(14)
    return ((ad * p10).sum(1), (bd * p10).sum(1))


def varlen(n):
    out = []
    for _ in range(2):
        d = torch.randint(0, 10, (n, 14), generator=g)
        L = torch.randint(1, 15, (n, 1), generator=g)
        d = torch.where(torch.arange(14)[None] < L, d, torch.zeros_like(d))
        out.append((d * 10 ** torch.arange(14)).sum(1))
    return out[0], out[1]


with torch.no_grad():
    u = sweep("uniform ", uni)
    s = sweep("propagate-stress", stress)
    v = sweep("variable-length", varlen)
print(f"\nSUMMARY params={npar} uniform={u:.6f} stress={s:.6f} varlen={v:.6f} "
      f"add_ok={N-len(bad)}/{N} edges={len(edges)-len(ebad)}/{len(edges)}")
