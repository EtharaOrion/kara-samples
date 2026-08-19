"""Inspect a trained checkpoint: what the code, the key feature and attention do."""
import argparse, torch
import lab_b2, lab_c
from lab import make_bias

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='/workspace/ckpt_b2.pt')
ap.add_argument('--arch', default='b')
ap.add_argument('--index', type=int, default=-1)
args = ap.parse_args()

ck = torch.load(args.ckpt, map_location='cpu')
i = ck['best'] if args.index < 0 else args.index
P = {k: v[i:i + 1] for k, v in ck['params'].items()}
print('model', i, 'acc', float(ck['acc'][i]), 'stress', float(ck['acc_stress'][i]))
print('slope', float(P['slope']), 'wq', float(P['wq']), 'bq', float(P['bq']), 'wv', float(P['wv']))

U = P['U'][0]
print('U:', ' '.join(f'{u:+.3f}' for u in U.tolist()))
print('U[a]+U[9-a]:', ' '.join(f'{(U[a]+U[9-a]):+.4f}' for a in range(10)))

z = torch.tensor([[U[a] + U[b] for b in range(10)] for a in range(10)])
ha = torch.relu(z[..., None] * P['W1a'][0] + P['b1a'][0])
c1 = (ha * P['W2a0'][0]).sum(-1)
c2 = (ha * P['W2a1'][0]).sum(-1)
print('\n  s   z(min..max)      c1(key)          c2(value)')
for s in range(19):
    pairs = [(a, s - a) for a in range(10) if 0 <= s - a <= 9]
    zs = [float(z[a, b]) for a, b in pairs]
    k = [float(c1[a, b]) for a, b in pairs]
    v = [float(c2[a, b]) for a, b in pairs]
    flag = ' <- propagate' if s == 9 else (' <- generate' if s >= 10 else '')
    print(f'{s:3d}  {min(zs):+.3f}..{max(zs):+.3f}   {min(k):+.3f}..{max(k):+.3f}   '
          f'{min(v):+.3f}..{max(v):+.3f}{flag}')

# attention on a hard example: 9999999999999 + 1  (long propagate chain)
fwd = lab_b2.forward if args.arch == 'b' else lab_c.forward
rel, block = make_bias('cpu')
a_d = torch.tensor([[9] * 13 + [0, 0]])
b_d = torch.tensor([[1] + [0] * 14])
s1 = lab_b2.encode_b(a_d, b_d)
zz = torch.einsum('blk,mk->mbl', s1, P['U'])
haa = torch.relu(zz[..., None] * P['W1a'][:, None, None, :] + P['b1a'][:, None, None, :])
cc1 = (haa * P['W2a0'][:, None, None, :]).sum(-1)
q = cc1 * P['wq'][:, None, None] + P['bq'][:, None, None]
sc = q[..., :, None] * cc1[..., None, :] + P['slope'][:, None, None, None] * rel
A = torch.softmax(sc.masked_fill(block, float('-inf')), -1)[0, 0]
print('\nattention rows (slot -> attended slot) for 9999999999999 + 1:')
for r in [1, 2, 5, 13, 14]:
    top = A[r].topk(3)
    print(f'  slot {r:2d}: ' + ' '.join(f'{int(j)}:{float(w):.3f}' for j, w in zip(top.indices, top.values)))
print('pred', fwd(P, s1, rel, block).argmax(-1)[0, 0, 1:].tolist())
print('true', lab_b2.sum_digits(a_d, b_d)[0].tolist())
