"""Independent verification of /workspace/submission.py.

Generates its own test data (no shared code with train.py beyond torch) and
measures exact-match accuracy on the eval distribution plus several stress
regimes, using the shipped module's own forward pass.
"""
import sys, time, torch

sys.path.insert(0, '/workspace')
import submission as S

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
MAXOP = 10 ** 14
NB = S.N_BITS


def tok(x):                                   # x: (B,) int64 -> (B, N_SLOTS)
    bits = (x.unsqueeze(-1) >> torch.arange(NB, device=x.device)) & 1
    return torch.cat([torch.zeros_like(bits[:, :1]), bits], 1)


@torch.no_grad()
def run(model, a, b):
    logits = model(tok(a), tok(b))[:, 1:]
    d = logits.argmax(-1)
    val = (d * (1 << torch.arange(NB, device=a.device))).sum(-1)
    return val, logits


def gen_uniform(n, g):
    return (torch.randint(0, MAXOP, (n,), generator=g, device=DEV),
            torch.randint(0, MAXOP, (n,), generator=g, device=DEV))


def gen_prop(n, g, p):
    """Per-slot propagate probability p (a_i != b_i), operands < 2**47."""
    pr = torch.rand(n, NB, generator=g, device=DEV) < p
    one = torch.randint(0, 2, (n, NB), generator=g, device=DEV)
    eq = torch.randint(0, 2, (n, NB), generator=g, device=DEV)
    ab = torch.where(pr, one, eq); bb = torch.where(pr, 1 - one, eq)
    ab[:, NB - 1] = 0; bb[:, NB - 1] = 0
    w = (1 << torch.arange(NB, device=DEV))
    return (ab * w).sum(1), (bb * w).sum(1)


def gen_run(n, g, length):
    """A forced propagate run of `length` slots at a random offset."""
    a, b = gen_prop(n, g, 0.5)
    w = (1 << torch.arange(NB, device=DEV))
    ab = (a.unsqueeze(-1) >> torch.arange(NB, device=DEV)) & 1
    bb = (b.unsqueeze(-1) >> torch.arange(NB, device=DEV)) & 1
    st = torch.randint(0, max(1, NB - length), (n, 1), generator=g, device=DEV)
    idx = torch.arange(NB, device=DEV)[None, :]
    inrun = (idx >= st) & (idx < st + length)
    bb = torch.where(inrun, 1 - ab, bb)
    ab[:, NB - 1] = 0; bb[:, NB - 1] = 0
    return (ab * w).sum(1), (bb * w).sum(1)


@torch.no_grad()
def measure(model, gen_fn, total, bs=8192, seed=7):
    g = torch.Generator(device=DEV); g.manual_seed(seed)
    wrong, margin, n = 0, float('inf'), 0
    for _ in range(total // bs):
        a, b = gen_fn(bs, g)
        val, lg = run(model, a, b)
        bad = val != (a + b)
        wrong += int(bad.sum()); n += bs
        m = (lg.max(-1).values - lg.min(-1).values).min()
        margin = min(margin, float(m))
    return 1.0 - wrong / n, wrong, margin


def main():
    model, meta = S.build_model()
    model.to(DEV)
    npar = sum(p.numel() for p in model.parameters())
    print(f'parameters: {npar}   (metadata claims {meta.get("n_parameters")})')
    for name, p in model.named_parameters():
        print(f'   {name:4s} {p.detach().cpu().numpy().round(5)}')

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
    t0 = time.time()
    acc, wrong, marg = measure(model, gen_uniform, n)
    print(f'uniform 14-digit : acc {acc:.6f}  wrong {wrong}/{n}  min margin {marg:.3f}')
    for p in (0.5, 0.9, 0.99, 1.0):
        acc2, w2, m2 = measure(model, lambda k, g, p=p: gen_prop(k, g, p), n // 4)
        print(f'propagate p={p:<5} : acc {acc2:.6f}  wrong {w2}/{n//4}  min margin {m2:.3f}')
    worst = 1.0
    for ln in range(0, NB):
        acc3, w3, _ = measure(model, lambda k, g, ln=ln: gen_run(k, g, ln), 65536)
        worst = min(worst, acc3)
        if acc3 < 1.0:
            print(f'   run length {ln:2d}: acc {acc3:.6f} ({w3} wrong)')
    print(f'forced runs 0..47: worst acc {worst:.6f}')

    # edge cases through the public add() entry point
    edges = [(0, 0), (0, 1), (1, 0), (99999999999999, 99999999999999), (99999999999999, 1),
             (1, 99999999999999), (0, 99999999999999), (50000000000000, 50000000000000),
             (99999999999999, 0), (12345678901234, 98765432109876), (2**46, 2**46),
             (999999999999, 1), (10**13, 10**13), (7, 8), (99999999999998, 1)]
    bad = [(a, b, S.add(model.cpu(), a, b)) for a, b in edges if S.add(model.cpu(), a, b) != a + b]
    print('edge cases:', 'all correct' if not bad else bad)

    # add() agrees with the batched path on random pairs
    g = torch.Generator(device='cpu'); g.manual_seed(99)
    ab = torch.randint(0, MAXOP, (2, 3000), generator=g)
    mism = sum(1 for i in range(3000)
               if S.add(model, int(ab[0, i]), int(ab[1, i])) != int(ab[0, i]) + int(ab[1, i]))
    print(f'add() on 3000 random pairs: {3000-mism}/3000 correct')

    # attention must depend on the input
    model.to(DEV)
    gd = torch.Generator(device=DEV); gd.manual_seed(3)
    a1, b1 = gen_prop(64, gd, 0.5)
    with torch.no_grad():
        z = model.U[tok(a1)] + model.U[tok(b1)]
        c1 = model.w1 * z + model.w2 * torch.relu(z + model.b)
        s = (c1 + model.bq).unsqueeze(-1) * c1.unsqueeze(-2) + model.slope * model.pos
        pa = torch.softmax(s + model.mask_in, -1)
    spread = (pa.amax(0) - pa.amin(0)).max()
    print(f'attention over 64 inputs: max per-cell spread {float(spread):.4f} '
          f'(0 would mean a fixed pattern)')
    print('   argmax target of slot 40, first 8 inputs:',
          pa[:8, 40].argmax(-1).tolist())
    print(f'({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
