"""Arch D: arch B' with the readout's first column tied to the input digit code.

The readout needs w_d proportional to the digit value plus a free bias-like
column, and the input code U is already affine in the digit, so column 0 of the
readout is shared with U (classic tied input/output embeddings) and only the
second column stays free.

params = 10(U) + 4*dffa + 4(attn) + 5*dffb + 10(Wout1)
"""
import torch
from lab_b2 import forward as _fwd_b, init_params as _init_b


def n_params(cfg):
    return 24 + 4 * cfg['d_ffa'] + 5 * cfg['d_ffb']


def init_params(M, dev, cfg, g):
    P = _init_b(M, dev, cfg, g)
    del P['Wout']
    P['Wout1'] = (torch.randn(M, 10, device=dev, generator=g) * 0.8).requires_grad_()
    return P


def forward(P, s1, rel, block, sigma=None, gen=None):
    P = dict(P)
    P['Wout'] = torch.stack([P['U'], P['Wout1']], -1)      # (M, 10, 2), column 0 tied to U
    return _fwd_b(P, s1, rel, block, sigma, gen)
