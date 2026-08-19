"""Arch I: give the attention an output projection, so the fold MLP can be narrower.

In arch H the attention's carry reaches the residual stream only through the fold
MLP, and one of that MLP's units is always active -- it exists purely to add a
term proportional to the carry.  The residual stream here is one channel wide,
so an ordinary attention output projection is a single scalar wo:

    y = z + wo * carry + mlp(z, carry)

That replaces a 4-parameter always-on unit with 1 parameter, and leaves the MLP
to do the part that actually needs a nonlinearity: the wrap at ten, a step in z
whose threshold moves by one digit when a carry comes in (two ReLU units).
"""
import torch
from lab_h import init_params as _init_h


def n_params(cfg):
    return 10 + 4 * cfg['d_ffa'] + 2 + 1 + 4 * cfg['d_ffb']


def init_params(M, dev, cfg, g):
    P = _init_h(M, dev, cfg, g)
    P['wo'] = (0.5 * torch.randn(M, device=dev, generator=g)).requires_grad_()
    return P


def fold(src):
    """Turn the always-on fold unit into the output projection wo.

    The always-on unit is the one that barely looks at z: it passes the carry
    through linearly, which is what wo does, plus a constant.  The constant is
    not thrown away -- shifting the whole digit code U by d shifts y by 2d and
    the target U[digit] by d, so a shift of +constant cancels it exactly (with
    b1a and b1b compensated so the two MLPs see what they saw before).  Only the
    unit's residual dependence on z is dropped, which is what training then
    cleans up.
    """
    P = {k: v.clone() for k, v in src.items()}
    if 'wo' in P:
        return P
    j = int((P['W1b0'] * P['W2b']).abs().argmin())
    P['wo'] = P['W1b1'][j] * P['W2b'][j]
    shift = P['b1b'][j] * P['W2b'][j]
    keep = [i for i in range(P['W1b0'].numel()) if i != j]
    for k in ('W1b0', 'W1b1', 'b1b', 'W2b'):
        if k in P:
            P[k] = P[k][keep]
    P['b1a'] = P['b1a'] - 2 * shift * P['W1a']
    P['b1b'] = P['b1b'] - 2 * shift * P['W1b0']
    P['U'] = P['U'] + shift
    return P


def forward(P, s1, rel, block, sigma=None, gen=None):
    z = torch.einsum('blk,mk->mbl', s1, P['U'])
    ha = torch.relu(z[..., None] * P['W1a'][:, None, None, :] + P['b1a'][:, None, None, :])
    c1 = (ha * P['W2a0'][:, None, None, :]).sum(-1)
    c2 = (ha * P['W2a1'][:, None, None, :]).sum(-1)

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
    y = z + P['wo'][:, None, None] * carry + torch.einsum('mblf,mf->mbl', hb, P['W2b'])
    U = P['U'][:, None, None, :]
    return 2.0 * y[..., None] * U - U * U
