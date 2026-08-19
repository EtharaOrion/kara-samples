"""Diagnose a trained cell against the conditions the carry-lookahead needs."""
import sys, torch, train7 as T

ck = torch.load(sys.argv[1])
k = int(sys.argv[2]) if len(sys.argv) > 2 else 0
p = {n: v[k].to(T.DEV) for n, v in ck.items()}
U, b, w2, bq, wo = p['U'], p['b'], p['w2'], p['bq'], p['wo']
print('U', U.tolist(), 'b', float(b), 'w2', float(w2), 'bq', float(bq), 'wo', wo.tolist())

z = torch.tensor([2 * U[0], U[0] + U[1], 2 * U[1]], device=T.DEV)   # kill, propagate, generate
c1 = T.W1 * z + w2 * torch.relu(z + b)
c2 = z
q = c1 + bq
names = ['kill(0,0)', 'prop(0,1)', 'gen (1,1)']
for i, n in enumerate(names):
    print(f'  {n}: z {z[i]:+9.4f}  c1(key) {c1[i]:+10.4f}  c2(val) {c2[i]:+9.4f}  q {q[i]:+10.4f}')

notch = min(c1[0], c1[2]) - c1[1]          # how far the propagate key sits below the others
spread = abs(c1[2] - c1[0])                # key difference between kill and generate
qmin = q.min()
print(f'\n  notch depth      {notch:+.5f}   -> q*notch = {qmin*notch:9.2f}  (needs > {4*48} '
      f'to beat recency over a full-length chain)')
print(f'  kill/gen spread  {spread:+.5f}   -> q*spread = {qmin*spread:9.4f}  (needs < 4 '
      f'so recency, not content, picks the nearest)')
print(f'  q > 0 everywhere: {bool((q > 0).all())}')
G = c2[2] - c2[0]
print(f'  wo0*G {float(wo[0]*G):+.5f} (want U1-U0 = {float(U[1]-U[0]):+.5f})')
print(f'  wo1*G {float(wo[1]*G):+.5f} (want -2*(U1-U0) = {float(-2*(U[1]-U[0])):+.5f})')
print(f'  c2(prop) {float(c2[1]):+.5f} vs c2(kill) {float(c2[0]):+.5f}')

g = torch.Generator(device=T.DEV); g.manual_seed(5)
pp = {n: v[None] for n, v in p.items()}
for ln in [0, 1, 2, 4, 8, 16, 24, 32, 40, 47]:
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
    _, ok = T.loss_and_acc(pp, at, bt, tg)
    print(f'  forced run {ln:2d}: exact {float(ok):.4f}')
a = torch.randint(0, T.MAXOP, (8192,), generator=g, device=T.DEV)
bq2 = torch.randint(0, T.MAXOP, (8192,), generator=g, device=T.DEV)
at, bt, tg = T.to_tokens(a, bq2)
_, ok = T.loss_and_acc(pp, at, bt, tg)
print(f'  uniform 14-digit: exact {float(ok):.5f}')
