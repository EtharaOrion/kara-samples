"""Evaluate a checkpoint's cells on uniform and stress pairs."""
import argparse
import torch

import lab_b2, lab_e, lab_f, lab_g, lab_h, lab_i
from lab import make_bias
from lab_b2 import evaluate

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--arch', default='g', choices=['b', 'e', 'f', 'g', 'h', 'i'])
ap.add_argument('--mlp_to', type=int, default=0)
ap.add_argument('--n', type=int, default=1_000_000)
args = ap.parse_args()

dev = 'cuda'
g = torch.Generator(device=dev); g.manual_seed(1234)
ck = torch.load(args.ckpt, map_location='cpu')
P = {k: v.to(dev) for k, v in ck['params'].items()}
fwd = (lab_i.forward if args.arch == 'i' else
       lab_h.forward if args.arch == 'h' else
       lab_g.make_forward(args.mlp_to) if args.arch == 'g' else
       lab_f.make_forward(args.mlp_to) if args.arch == 'f' else
       lab_e.forward if args.arch == 'e' else lab_b2.forward)
rel, block = make_bias(dev)
print('parameters', sum(v[0].numel() for v in P.values()))
print('uniform ' + ' '.join(f'{a:.5f}' for a in
                            evaluate(P, rel, block, dev, g, n=args.n, fwd=fwd).tolist()))
print('stress  ' + ' '.join(f'{a:.5f}' for a in
                            evaluate(P, rel, block, dev, g, n=args.n, stress=True, fwd=fwd).tolist()))
