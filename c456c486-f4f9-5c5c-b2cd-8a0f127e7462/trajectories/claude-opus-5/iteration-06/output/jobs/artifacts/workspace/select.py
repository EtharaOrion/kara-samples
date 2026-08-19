"""Score the top cells of a checkpoint over large samples and export the best."""
import sys, torch, train as T

ck = torch.load(sys.argv[1])
K = int(sys.argv[2]) if len(sys.argv) > 2 else 24
p = {n: v[:K].to(T.DEV) for n, v in ck.items()}
M = p['U'].shape[0]
g = torch.Generator(device=T.DEV); g.manual_seed(20240814)

wrong = torch.zeros(M, device=T.DEV)
margin = torch.full((M,), float('inf'), device=T.DEV)
n_tot = 0
for it in range(120):
    if it % 3 == 2:
        at, bt, tg = T.eval_batch(4096, g)
    elif it % 3 == 1:
        at, bt, tg = T.hard_batch(4096, g)
    else:                                    # every forced run length in turn
        ln = (it // 3) % T.NBITS
        n = 4096
        pr = torch.rand(n, T.NBITS, generator=g, device=T.DEV) < 0.5
        one = torch.randint(0, 2, (n, T.NBITS), generator=g, device=T.DEV)
        eq = torch.randint(0, 2, (n, T.NBITS), generator=g, device=T.DEV)
        A = torch.where(pr, one, eq); B = torch.where(pr, 1 - one, eq)
        st = torch.randint(0, max(1, T.NBITS - ln), (n, 1), generator=g, device=T.DEV)
        idx = torch.arange(T.NBITS, device=T.DEV)[None, :]
        inrun = (idx >= st) & (idx < st + ln)
        B = torch.where(inrun, 1 - A, B)
        A[:, -1] = 0; B[:, -1] = 0
        at, bt, tg = T.to_tokens((A * T.POW).sum(1), (B * T.POW).sum(1))
    with torch.no_grad():
        lg = T.forward(p, at, bt)[:, :, 1:, :]
        t = tg.unsqueeze(0).expand_as(lg[..., 0])
        wrong += (lg.argmax(-1) != t).any(-1).float().sum(1)
        corr = lg.gather(-1, t.unsqueeze(-1)).squeeze(-1)
        other = lg.sum(-1) - corr
        margin = torch.minimum(margin, (corr - other).amin(-1).amin(-1))
    n_tot += at.shape[0]

for i in range(M):
    print(f'cell {i:2d}: wrong {int(wrong[i]):6d}/{n_tot}  min logit margin {margin[i]:+9.3f}')
good = (wrong == 0).nonzero().flatten()
if good.numel() == 0:
    print('no perfect cell'); sys.exit(1)
best = good[margin[good].argmax()].item()
print(f'\nbest cell {best}: margin {margin[best]:.3f}, {n_tot} pairs, 0 errors')
if len(sys.argv) > 3 and sys.argv[3] == 'export':
    T.export(p, best)
