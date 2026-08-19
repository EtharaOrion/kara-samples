"""Fold a trained arch-H model into arch I: the always-on output-MLP unit becomes
the attention's scalar output projection (see lab_i.fold for why it is exact up
to that unit's negligible dependence on z)."""
import argparse
import torch

import lab_h, lab_i

ap = argparse.ArgumentParser()
ap.add_argument('--src', default='/workspace/ckpt_ft_h36.pt')
ap.add_argument('--index', type=int, default=-1)
ap.add_argument('--out', default='/workspace/ckpt_fold33.pt')
args = ap.parse_args()

ck = torch.load(args.src, map_location='cpu')
i = ck['best'] if args.index < 0 else args.index
P = {k: v[i] for k, v in ck['params'].items()}
Q = lab_i.fold(lab_h.fold(P))
torch.save({'params': {k: v[None] for k, v in Q.items()}, 'best': 0,
            'acc': torch.zeros(1), 'acc_stress': torch.zeros(1), 'cfg': dict(ck.get('cfg', {}))},
           args.out)
print('wrote', args.out, 'parameters', sum(v.numel() for v in Q.values()),
      'wo', float(Q['wo']))
