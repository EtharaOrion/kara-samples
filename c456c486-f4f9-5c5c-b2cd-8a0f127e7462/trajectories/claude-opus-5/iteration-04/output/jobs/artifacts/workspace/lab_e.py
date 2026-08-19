"""Arch E: arch B'/D with the two redundant attention scalars removed.

Arch B' scores attention with  (wq*c1_i + bq) * c1_j + slope*(j - i)  and scales
the attention output by wv.  Both scalars are absorbable:

  wv   carry = wv * (A @ c2) feeds the readout only through Wout[:, 1] and the
       output MLP's W1b1 / W2b[1], so multiplying those three by wv (and W2b[1]
       by 1/wv) leaves every logit unchanged.  Exactly redundant.
  wq   c1 is used only in the score, so rescaling W2a0 by sqrt(wq) turns
       wq*c1_i*c1_j into c1_i*c1_j and bq into bq/sqrt(wq).  Redundant up to the
       sign of wq, which the architecture fixes at +1.

So attention keeps two learned scalars, bq and slope, and the score is
    score(i, j) = c1_i * c1_j + bq * c1_j + slope * (j - i).
fold() maps trained arch-B'/D weights into this parametrisation exactly (the
logits are unchanged); training then continues in the smaller parametrisation.
"""
import torch
from lab_b2 import init_params as _init_b


def n_params(cfg):
    head = 10 if cfg.get('tied') else 20
    return 10 + 4 * cfg['d_ffa'] + 2 + 5 * cfg['d_ffb'] + head


def init_params(M, dev, cfg, g):
    P = _init_b(M, dev, cfg, g)
    del P['wq'], P['wv']
    if cfg.get('tied'):
        del P['Wout']
        P['Wout1'] = (torch.randn(M, 10, device=dev, generator=g) * 0.8).requires_grad_()
    return P


def fold(src, tied):
    """Exact reparametrisation of arch B'/D weights (no leading model dim)."""
    P = {k: v.clone() for k, v in src.items()}
    wq = P.pop('wq'); wv = P.pop('wv')
    a = wq.clamp_min(1e-6).sqrt()
    P['W2a0'] = P['W2a0'] * a
    P['bq'] = P['bq'] / a
    P['W1b1'] = P['W1b1'] * wv
    P['W2b'] = torch.stack([P['W2b'][0], P['W2b'][1] / wv])
    if tied or 'Wout1' in P:
        P['Wout1'] = P['Wout1'] * wv
    else:
        P['Wout'] = torch.stack([P['Wout'][:, 0], P['Wout'][:, 1] * wv], -1)
    return P


def forward(P, s1, rel, block, sigma=None, gen=None):
    z = torch.einsum('blk,mk->mbl', s1, P['U'])
    ha = torch.relu(z[..., None] * P['W1a'][:, None, None, :] + P['b1a'][:, None, None, :])
    c1 = (ha * P['W2a0'][:, None, None, :]).sum(-1)                   # attention key
    c2 = (ha * P['W2a1'][:, None, None, :]).sum(-1)                   # attention value

    sc = (c1[..., :, None] + P['bq'][:, None, None, None]) * c1[..., None, :] \
        + P['slope'][:, None, None, None] * rel
    if sigma is not None:
        sc = sc + sigma[:, None, None, None] * torch.randn(sc.shape, device=sc.device, generator=gen)
    A = torch.softmax(sc.masked_fill(block, float('-inf')), -1)
    carry = torch.einsum('mbij,mbj->mbi', A, c2)

    x = torch.stack([z, carry], -1)
    hb = torch.relu(z[..., None] * P['W1b0'][:, None, None, :]
                    + carry[..., None] * P['W1b1'][:, None, None, :]
                    + P['b1b'][:, None, None, :])
    x = x + torch.einsum('mblf,mdf->mbld', hb, P['W2b'])
    Wout = torch.stack([P['U'], P['Wout1']], -1) if 'Wout1' in P else P['Wout']
    return torch.einsum('mbld,mcd->mblc', x, Wout)
