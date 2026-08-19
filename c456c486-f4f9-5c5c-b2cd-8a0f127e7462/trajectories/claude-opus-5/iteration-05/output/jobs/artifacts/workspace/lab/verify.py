"""Independent check of /workspace/submission.py: parameter count + exact-match accuracy."""
import importlib.util, random, sys, torch

spec = importlib.util.spec_from_file_location('sub', '/workspace/submission.py')
sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
model, meta = sub.build_model()
NP = sum(p.numel() for p in model.parameters())
print('metadata:', meta)
print('registered parameters:', NP,
      {n: tuple(p.shape) for n, p in model.named_parameters()})
print('buffers:', {n: tuple(b.shape) for n, b in model.named_buffers()})

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(dev)
P10 = torch.tensor([10 ** i for i in range(15)], dtype=torch.long, device=dev)


def batch_exact(a, b):
    """a,b: (B,15) digit tensors. Returns exact-match mask and digit-accuracy."""
    B = a.shape[0]
    oh = torch.zeros(B, 16, 10, device=dev)
    oh[:, 1:, :].scatter_add_(2, a.unsqueeze(-1), torch.ones(B, 15, 1, device=dev))
    oh[:, 1:, :].scatter_add_(2, b.unsqueeze(-1), torch.ones(B, 15, 1, device=dev))
    oh[:, 0, 0] = 2.0
    with torch.no_grad():
        pred = model(oh).argmax(-1)
    s = (a * P10).sum(-1) + (b * P10).sum(-1)
    tgt = (s.unsqueeze(-1) // P10) % 10
    ok = (pred == tgt)
    return ok.all(-1), ok.float().mean().item(), pred, tgt


def rand_digits(B, g, maxlen=14):
    d = torch.randint(0, 10, (B, 15), device=dev, generator=g)
    d[:, 14:] = 0
    if maxlen < 14:
        d[:, maxlen:] = 0
    return d


g = torch.Generator(device=dev).manual_seed(4242)
total, wrong = 0, 0
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
CH = 100_000
report = []
for name in ('uniform', 'varlen', 'skew', 'propagate', 'runs'):
    bad = 0; n = 0
    for _ in range(max(1, N // CH)):
        if name == 'uniform':
            a, b = rand_digits(CH, g), rand_digits(CH, g)
        elif name == 'varlen':
            a, b = rand_digits(CH, g), rand_digits(CH, g)
            for d in (a, b):
                L = torch.randint(1, 15, (CH, 1), device=dev, generator=g)
                d.mul_((torch.arange(15, device=dev).view(1, -1) < L).long())
        elif name == 'skew':
            a, b = rand_digits(CH, g), rand_digits(CH, g)
            for d in (a, b):
                pk = torch.rand(CH, 15, device=dev, generator=g)
                d.copy_(torch.where(pk < 0.3, torch.zeros_like(d), torch.where(pk > 0.7, torch.full_like(d, 9), d)))
                d[:, 14] = 0
        elif name == 'propagate':
            a = rand_digits(CH, g); b = rand_digits(CH, g)
            rate = 0.3 + 0.7 * torch.rand(CH, 1, device=dev, generator=g)
            m = torch.rand(CH, 15, device=dev, generator=g) < rate
            b = torch.where(m, 9 - a, b); b[:, 14] = 0
        else:
            a = rand_digits(CH, g); b = rand_digits(CH, g)
            rl = torch.randint(0, 15, (CH, 1), device=dev, generator=g)
            st = torch.randint(0, 15, (CH, 1), device=dev, generator=g)
            ix = torch.arange(15, device=dev).view(1, -1)
            b = torch.where((ix >= st) & (ix < st + rl), 9 - a, b); b[:, 14] = 0
        ok, dacc, pred, tgt = batch_exact(a, b)
        bad += int((~ok).sum()); n += CH
    report.append((name, n, bad, 1 - bad / n))
    print(f'{name:10s} n={n:9d} wrong={bad:7d} exact={1-bad/n:.6f}', flush=True)

# per-propagate-run-length sweep
print('\nexact accuracy by forced-propagate run length:')
for rl in range(0, 15):
    a = rand_digits(200_000, g); b = rand_digits(200_000, g)
    st = torch.randint(0, 15, (200_000, 1), device=dev, generator=g)
    ix = torch.arange(15, device=dev).view(1, -1)
    b = torch.where((ix >= st) & (ix < st + rl), 9 - a, b); b[:, 14] = 0
    ok, _, _, _ = batch_exact(a, b)
    print(f'  runlen {rl:2d}: {ok.float().mean().item():.6f}', flush=True)

# the graded interface itself, on edge cases and random ints
edge = [(0, 0), (0, 1), (1, 0), (99999999999999, 99999999999999), (99999999999999, 1),
        (1, 99999999999999), (50000000000000, 50000000000000), (9999999, 1), (999999999999, 1),
        (12345678901234, 98765432109876), (10000000000000, 10000000000000), (0, 99999999999999),
        (55555555555555, 44444444444444), (99999999999998, 2), (1, 1), (10, 90), (999, 1), (1, 999)]
bad = [(a, b) for a, b in edge if sub.add(model.cpu(), a, b) != a + b]
print('\nedge cases wrong:', bad if bad else 'none', f'({len(edge)} tested)')
rnd = random.Random(0)
n_api = 3000
bad2 = 0
for _ in range(n_api):
    a = rnd.randint(0, 99999999999999); b = rnd.randint(0, 99999999999999)
    if sub.add(model, a, b) != a + b:
        bad2 += 1
print(f'add() API on {n_api} random pairs: wrong={bad2}')
print(f'\nPARAMS={NP}  worst-regime exact={min(r[3] for r in report):.6f}')
