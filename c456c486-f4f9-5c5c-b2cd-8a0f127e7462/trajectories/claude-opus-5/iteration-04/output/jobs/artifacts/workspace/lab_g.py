"""Arch G: arch F with the feature MLP's constant unit removed.

Training reliably drives one unit of the feature MLP to W1a = 0, i.e. to a
constant relu(b1a) that only supplies offsets to the key c1 and the value c2.
Both offsets are cheaper than a hidden unit:

  key    a constant k1 in c1 shifts the score by (k1 + bq) * c1_j plus terms
         that are constant along j, and those cancel in the softmax -- so k1 is
         redundant with bq and can simply be dropped.
  value  a constant k2 in c2 survives as a constant in the attention output
         (the attention weights sum to one), so it is kept as one scalar.

That trades a 4-parameter hidden unit for a single parameter.  fold() maps arch
F weights into this parametrisation; it is exact when the unit really is
constant, which is the case to about 1e-7 in the models this is applied to.
"""
import torch
from lab_e import init_params as _init_e


def n_params(cfg):
    head = 10 if cfg.get('tied') else 20
    return 10 + 4 * cfg['d_ffa'] + 2 + 1 + 4 * cfg['d_ffb'] + head


def init_params(M, dev, cfg, g):
    P = _init_e(M, dev, cfg, g)
    P['W2b'] = P['W2b'][:, cfg['mlp_to']].detach().clone().requires_grad_()
    P['k2'] = (torch.randn(M, device=dev, generator=g)).requires_grad_()
    return P


def fold(src, unit=0):
    """Drop the constant hidden unit `unit`, keeping its two offsets."""
    P = {k: v.clone() for k, v in src.items()}
    const = torch.relu(P['b1a'][unit])
    P['bq'] = P['bq'] + P['W2a0'][unit] * const          # k1 merges into bq
    P['k2'] = P['W2a1'][unit] * const
    keep = [i for i in range(P['W1a'].numel()) if i != unit]
    for k in ('W1a', 'b1a', 'W2a0', 'W2a1'):
        P[k] = P[k][keep]
    return P


def make_forward(mlp_to, carry_out=True):
    """carry_out=False drops the readout's carry column: the carry then reaches
    the logits only through the fold MLP, and the residual stream the readout
    sees is just the digit-code channel."""
    def forward(P, s1, rel, block, sigma=None, gen=None):
        z = torch.einsum('blk,mk->mbl', s1, P['U'])
        ha = torch.relu(z[..., None] * P['W1a'][:, None, None, :] + P['b1a'][:, None, None, :])
        c1 = (ha * P['W2a0'][:, None, None, :]).sum(-1)
        c2 = (ha * P['W2a1'][:, None, None, :]).sum(-1) + P['k2'][:, None, None]

        sc = (c1[..., :, None] + P['bq'][:, None, None, None]) * c1[..., None, :] \
            + P['slope'][:, None, None, None] * rel
        if sigma is not None:
            sc = sc + sigma[:, None, None, None] * torch.randn(sc.shape, device=sc.device,
                                                               generator=gen)
        A = torch.softmax(sc.masked_fill(block, float('-inf')), -1)
        carry = torch.einsum('mbij,mbj->mbi', A, c2)

        bb = P['b1b'][:, None, None, :] if 'b1b' in P else 0.0
        hb = torch.relu(z[..., None] * P['W1b0'][:, None, None, :]
                        + carry[..., None] * P['W1b1'][:, None, None, :] + bb)
        fold_out = torch.einsum('mblf,mf->mbl', hb, P['W2b'])
        if not carry_out:
            return torch.einsum('mbl,mc->mblc', z + fold_out, P['U'])
        x = (torch.stack([z + fold_out, carry], -1) if mlp_to == 0
             else torch.stack([z, carry + fold_out], -1))
        Wout = torch.stack([P['U'], P['Wout1']], -1) if 'Wout1' in P else P['Wout']
        return torch.einsum('mbld,mcd->mblc', x, Wout)
    return forward
