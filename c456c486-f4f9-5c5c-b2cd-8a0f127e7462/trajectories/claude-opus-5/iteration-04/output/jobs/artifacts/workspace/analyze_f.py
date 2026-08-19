"""Check what the shipped model actually computes: is the key a notch on the
propagate sums, and does attention really select the nearest settling slot?"""
import argparse, importlib.util
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--sub', default='/workspace/submission.py')
args = ap.parse_args()

spec = importlib.util.spec_from_file_location('sub', args.sub)
sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
m, meta = sub.build_model()
print('parameters', meta['parameters'])

U = m.U.detach()
d = torch.arange(10.0)
A = torch.stack([d, torch.ones(10)], 1)
coef, *_ = torch.linalg.lstsq(A, U[:, None])
print('U:', ' '.join(f'{u:+.3f}' for u in U.tolist()))
print(f'U affine fit {float(coef[0]):+.4f}*d {float(coef[1]):+.4f}, '
      f'resid {float((A @ coef - U[:, None]).abs().max()):.4f}')

z = U[:, None] + U[None, :]
ha = torch.relu(z[..., None] * m.W1a + m.b1a)
c1 = (ha * m.W2a0).sum(-1)
c2 = (ha * m.W2a1).sum(-1) + (m.k2 if hasattr(m, 'k2') else 0.0)
print('\n  s   c1 (attention key)     c2 (attention value)')
for s in range(19):
    p = [(a, s - a) for a in range(10) if 0 <= s - a <= 9]
    k = [float(c1[a, b]) for a, b in p]; v = [float(c2[a, b]) for a, b in p]
    tag = ' <- propagate' if s == 9 else (' <- generate' if s >= 10 else '')
    print(f'{s:3d}  {min(k):+8.3f}..{max(k):+8.3f}  {min(v):+8.3f}..{max(v):+8.3f}{tag}')

# attention on a long propagate chain: 9999999999999 + 1
a_tok = torch.tensor([[0] + [9] * 13 + [0, 0]])
b_tok = torch.tensor([[0, 1] + [0] * 14])
onehot = (torch.nn.functional.one_hot(a_tok, 10) + torch.nn.functional.one_hot(b_tok, 10)).float()
zz = onehot @ U
h = torch.relu(zz[..., None] * m.W1a + m.b1a)
k1 = (h * m.W2a0).sum(-1)
score = (k1[..., :, None] + m.bq) * k1[..., None, :] + m.slope * m.rel
att = torch.softmax(score.masked_fill(m.blocked, float('-inf')), -1)[0]
print('\n9999999999999 + 1  (slots 1..14 are propagate except slot 1, which generates)')
for r in (1, 2, 5, 10, 14):
    t = att[r].topk(2)
    print(f'  slot {r:2d} attends ' + ' '.join(f'{int(j)}:{float(w):.3f}' for j, w in zip(t.indices, t.values)))

# and a case where the same slot must attend somewhere else entirely
a2 = torch.tensor([[0] + [4] * 13 + [0, 0]])
b2 = torch.tensor([[0, 1] + [0] * 14])
onehot2 = (torch.nn.functional.one_hot(a2, 10) + torch.nn.functional.one_hot(b2, 10)).float()
h2 = torch.relu((onehot2 @ U)[..., None] * m.W1a + m.b1a)
k2 = (h2 * m.W2a0).sum(-1)
s2 = (k2[..., :, None] + m.bq) * k2[..., None, :] + m.slope * m.rel
att2 = torch.softmax(s2.masked_fill(m.blocked, float('-inf')), -1)[0]
print('4444444444444 + 1  (no propagates: every slot should just look back one)')
for r in (1, 2, 5, 10, 14):
    t = att2[r].topk(2)
    print(f'  slot {r:2d} attends ' + ' '.join(f'{int(j)}:{float(w):.3f}' for j, w in zip(t.indices, t.values)))

print('\nadd(99999999999999, 1) =', sub.add(m, 99999999999999, 1))
print('add(4444444444444, 1)  =', sub.add(m, 4444444444444, 1))
