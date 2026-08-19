"""Arch C: arch B' with the output ReLU fold replaced by a periodic feature layer.

The answer digit is (digit sum + carry) mod 10, and "mod 10" is a rotation, so a
sine activation expresses the fold with two units instead of a ReLU staircase:

  z_i    = U[a_i] + U[b_i]                              additive digit code (10)
  hA     = relu(z * W1a + b1a)                          feature MLP     (2*dffa)
  c1, c2 = hA.W2a0, hA.W2a1                             key / value     (2*dffa)
  score  = (wq*c1_i + bq) * c1_j + slope*(j-i)          carry lookahead (3)
  carry  = wv * sum_j A_ij c2_j                                         (1)
  f      = sin(z*W1b0 + carry*W1b1 + b1b)               periodic fold   (3*dffb)
  logits = f . Wout^T                                   learned readout (10*dffb)

params = 10 + 4*dffa + 4 + 13*dffb
"""
import argparse
import torch

from lab import make_bias
from lab_b2 import run

torch.backends.cuda.matmul.allow_tf32 = True


def n_params(cfg):
    return 10 + 4 * cfg['d_ffa'] + 4 + 13 * cfg['d_ffb']


def init_params(M, dev, cfg, g):
    def r(*shape, s=1.0):
        return (torch.randn(M, *shape, device=dev, generator=g) * s).requires_grad_()
    da, db = cfg['d_ffa'], cfg['d_ffb']
    freq = cfg['freq'].to(dev)[:, None]
    P = {
        'U': r(10, s=0.8),
        'W1a': r(da, s=1.0), 'b1a': r(da, s=1.0), 'W2a0': r(da, s=1.0), 'W2a1': r(da, s=1.0),
        'wq': r(s=1.0), 'bq': r(s=1.0), 'wv': r(s=1.0),
        'W1b0': (torch.randn(M, db, device=dev, generator=g) * freq).requires_grad_(),
        'W1b1': (torch.randn(M, db, device=dev, generator=g) * freq).requires_grad_(),
        'b1b': (torch.rand(M, db, device=dev, generator=g) * 6.283).requires_grad_(),
        'Wout': r(10, db, s=0.8),
    }
    P['slope'] = cfg['slope_init'].clone().to(dev).requires_grad_()
    return P


def forward(P, s1, rel, block, sigma=None, gen=None):
    z = torch.einsum('blk,mk->mbl', s1, P['U'])
    ha = torch.relu(z[..., None] * P['W1a'][:, None, None, :] + P['b1a'][:, None, None, :])
    c1 = (ha * P['W2a0'][:, None, None, :]).sum(-1)
    c2 = (ha * P['W2a1'][:, None, None, :]).sum(-1)

    q = c1 * P['wq'][:, None, None] + P['bq'][:, None, None]
    sc = q[..., :, None] * c1[..., None, :] + P['slope'][:, None, None, None] * rel
    if sigma is not None:
        sc = sc + sigma[:, None, None, None] * torch.randn(sc.shape, device=sc.device, generator=gen)
    A = torch.softmax(sc.masked_fill(block, float('-inf')), -1)
    carry = P['wv'][:, None, None] * torch.einsum('mbij,mbj->mbi', A, c2)

    f = torch.sin(z[..., None] * P['W1b0'][:, None, None, :]
                  + carry[..., None] * P['W1b1'][:, None, None, :]
                  + P['b1b'][:, None, None, :])
    return torch.einsum('mblf,mcf->mblc', f, P['Wout'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=60000)
    ap.add_argument('--bs', type=int, default=2048)
    ap.add_argument('--lr', type=float, default=3.5e-3)
    ap.add_argument('--d_ffa', type=int, default=3)
    ap.add_argument('--d_ffb', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='/workspace/ckpt_c.pt')
    ap.add_argument('--slopes', default='2,3,4,6')
    ap.add_argument('--freqs', default='0.3,0.7,1.5,3.0')
    ap.add_argument('--sigma', type=float, default=0.05)
    args = ap.parse_args()

    slopes = torch.tensor([float(s) for s in args.slopes.split(',')]).repeat_interleave(4)
    freqs = torch.tensor([float(s) for s in args.freqs.split(',')]).repeat(4)
    sigmas = torch.full((16,), args.sigma)
    cfg = {'d_ffa': args.d_ffa, 'd_ffb': args.d_ffb, 'freq': freqs}
    run(args, cfg, slopes, sigmas, fwd=forward, initf=init_params, npf=n_params)


if __name__ == '__main__':
    main()
