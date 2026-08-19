"""How much slack does the readout have?  A large minimum margin means the exact-match
result is structural, not a float32 coincidence."""
import importlib.util, torch
spec = importlib.util.spec_from_file_location('sub', '/workspace/submission.py')
sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
model, _ = sub.build_model()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(dev)
P10 = torch.tensor([10 ** i for i in range(15)], dtype=torch.long, device=dev)

def run(a, b):
    B = a.shape[0]
    oh = torch.zeros(B, 16, 10, device=dev)
    oh[:, 1:, :].scatter_add_(2, a.unsqueeze(-1), torch.ones(B, 15, 1, device=dev))
    oh[:, 1:, :].scatter_add_(2, b.unsqueeze(-1), torch.ones(B, 15, 1, device=dev))
    oh[:, 0, 0] = 2.0
    with torch.no_grad():
        lg = model(oh)
    s = (a * P10).sum(-1) + (b * P10).sum(-1)
    tgt = ((s.unsqueeze(-1) // P10) % 10)
    corr = lg.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    other = lg.masked_fill(torch.nn.functional.one_hot(tgt, 10).bool(), -1e30).max(-1).values
    return corr - other, tgt, lg

g = torch.Generator(device=dev).manual_seed(1)
worst = 1e30
for name, mk in [
    ('uniform', lambda n: (torch.randint(0, 10, (n, 15), device=dev, generator=g),
                           torch.randint(0, 10, (n, 15), device=dev, generator=g))),
    ('all-propagate', lambda n: (lambda a: (a, 9 - a))(torch.randint(0, 10, (n, 15), device=dev, generator=g))),
]:
    for _ in range(20):
        a, b = mk(50_000); a[:, 14] = 0; b[:, 14] = 0
        m, _, _ = run(a, b)
        worst = min(worst, float(m.min()))
    print(f'{name:14s} min margin so far {worst:.4f}')

# exhaustive over the local pattern: every digit pair at every slot, worst-case context
print('\nexhaustive: all 100 digit pairs x all 15 slots, inside a full propagate chain')
mn = 1e30
for pos in range(15):
    a = torch.zeros(100, 15, dtype=torch.long, device=dev)
    b = torch.zeros(100, 15, dtype=torch.long, device=dev)
    base = torch.arange(100, device=dev)
    ctx = torch.randint(0, 10, (100, 15), device=dev, generator=g)
    a.copy_(ctx); b.copy_(9 - ctx)                 # everything else propagates
    a[:, pos] = base // 10; b[:, pos] = base % 10
    a[:, 14] = 0; b[:, 14] = 0
    m, t, _ = run(a, b)
    ok = (m > 0).all().item()
    mn = min(mn, float(m.min()))
    if not ok:
        print('  FAIL at pos', pos)
print('  all correct, min margin', round(mn, 4))
print(f'\nOVERALL min margin {min(worst, mn):.4f}  (logit units; centre spacing '
      f'~{float((2*model.U[:1]*(model.U[1]-model.U[0])).abs().mean()):.1f})')
