"""Arch F: arch E with the output MLP writing into only one residual channel.

Arch E's residual is (z_i, carry_i) and the width-{d_ffb} fold MLP adds into
both channels, costing 2*d_ffb weights.  The readout is 2-dimensional, so the
MLP arguably only has to move one channel; this variant keeps a single row of
W2b (which channel is a config choice) and saves d_ffb parameters.
"""
import torch
from lab_e import init_params as _init_e


def n_params(cfg):
    head = 10 if cfg.get('tied') else 20
    return 10 + 4 * cfg['d_ffa'] + 2 + 4 * cfg['d_ffb'] + head


def init_params(M, dev, cfg, g):
    P = _init_e(M, dev, cfg, g)
    P['W2b'] = P['W2b'][:, cfg['mlp_to']].detach().clone().requires_grad_()
    return P


def make_forward(mlp_to):
    """mlp_to: which residual channel (0 = digit code, 1 = carry) the fold writes."""
    def forward(P, s1, rel, block, sigma=None, gen=None):
        z = torch.einsum('blk,mk->mbl', s1, P['U'])
        ha = torch.relu(z[..., None] * P['W1a'][:, None, None, :] + P['b1a'][:, None, None, :])
        c1 = (ha * P['W2a0'][:, None, None, :]).sum(-1)
        c2 = (ha * P['W2a1'][:, None, None, :]).sum(-1)

        sc = (c1[..., :, None] + P['bq'][:, None, None, None]) * c1[..., None, :] \
            + P['slope'][:, None, None, None] * rel
        if sigma is not None:
            sc = sc + sigma[:, None, None, None] * torch.randn(sc.shape, device=sc.device, generator=gen)
        A = torch.softmax(sc.masked_fill(block, float('-inf')), -1)
        carry = torch.einsum('mbij,mbj->mbi', A, c2)

        # without b1b the fold's thresholds come from the carry channel itself,
        # which is bounded away from zero and differs by carry class
        bb = P['b1b'][:, None, None, :] if 'b1b' in P else 0.0
        hb = torch.relu(z[..., None] * P['W1b0'][:, None, None, :]
                        + carry[..., None] * P['W1b1'][:, None, None, :] + bb)
        fold = torch.einsum('mblf,mf->mbl', hb, P['W2b'])
        if mlp_to == 0:
            x = torch.stack([z + fold, carry], -1)
        else:
            x = torch.stack([z, carry + fold], -1)
        Wout = torch.stack([P['U'], P['Wout1']], -1) if 'Wout1' in P else P['Wout']
        return torch.einsum('mbld,mcd->mblc', x, Wout)
    return forward
