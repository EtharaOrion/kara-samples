"""Drop a redundant feature-MLP unit without changing what the model computes.

The feature MLP's hidden units are only ever evaluated at the 19 possible digit
sums, so their activations live on a 19-point lattice.  When more units fire on
the same stretch of that lattice than the two features need, one of them is
linearly redundant: the key and value read-outs can be re-solved on the
remaining units and reproduce c1 and c2 exactly.  This writes out the reduced
checkpoint, which is then re-trained in the smaller parametrisation.
"""
import argparse, itertools
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='/workspace/ckpt_ft_g47.pt')
ap.add_argument('--index', type=int, default=-1)
ap.add_argument('--keep', type=int, default=3)
ap.add_argument('--out', default='/workspace/ckpt_refit.pt')
args = ap.parse_args()

ck = torch.load(args.ckpt, map_location='cpu')
i = ck['best'] if args.index < 0 else args.index
P = {k: v[i] for k, v in ck['params'].items()}
U = P['U']
z = torch.tensor([float(U[a] + U[s - a]) for s in range(19) for a in [max(0, s - 9)]])
H = torch.relu(z[:, None] * P['W1a'] + P['b1a'])
c1, c2 = H @ P['W2a0'], H @ P['W2a1']
usable = [s for s in range(19) if s != 9]          # propagate slots are never attended

best = None
for S in itertools.combinations(range(P['W1a'].numel()), args.keep):
    Hs = H[:, list(S)]
    w = torch.linalg.lstsq(Hs, c1[:, None]).solution
    v = torch.linalg.lstsq(Hs[usable], c2[usable, None]).solution
    err = max(float((Hs @ w - c1[:, None]).abs().max()),
              float((Hs[usable] @ v - c2[usable, None]).abs().max()))
    scale = max(float(w.abs().max()), float(v.abs().max()))
    print(f'units {S}: max feature error {err:.5f}, max |read-out weight| {scale:.2f}')
    if best is None or (err, scale) < best[0]:
        best = ((err, scale), S, w.flatten(), v.flatten())

(err, _), S, w, v = best
print('keeping units', S, 'with max feature error', round(err, 6))
Q = {k: val.clone() for k, val in P.items()}
for k in ('W1a', 'b1a'):
    Q[k] = Q[k][list(S)]
Q['W2a0'] = w
Q['W2a1'] = v
torch.save({'params': {k: val[None] for k, val in Q.items()}, 'best': 0,
            'acc': torch.zeros(1), 'acc_stress': torch.zeros(1),
            'cfg': dict(ck.get('cfg', {}))}, args.out)
print('wrote', args.out, 'parameters', sum(val.numel() for val in Q.values()))
