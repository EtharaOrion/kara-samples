"""Arch H: arch G with a distance-based unembedding, so the readout costs nothing.

Arch G's readout is  logit_c = x0 * U_c + x1 * Wout1_c  over a two-channel
residual stream.  Both columns are needed, but not for the reason it looks:
with a single column the logits are *linear* in the class embedding, so the
argmax is whichever class has the extreme embedding value, no matter what the
model computes -- an interior digit can never win.  Wout1 exists to supply a
second, curved direction that turns the head into a proper decoder.

A distance head does that with no parameters of its own:

    logit_c = -(y - U_c)^2  =  2*y*U_c - U_c^2   (dropping the -y^2 that is
                                                  constant across classes)

i.e. the model predicts a point y in the same embedding space the digits are
encoded in, and the readout picks the nearest digit.  U is still learned, and
nothing about addition is built in -- the fold MLP must still learn to produce
the code of (digit sum + carry) mod 10, wrap-around included.  The carry now
reaches the logits only through that MLP, which also makes the value offset k2
redundant (a constant shift of the carry is absorbed by the MLP's bias b1b).
"""
import torch
from lab_e import init_params as _init_e


def n_params(cfg):
    return 10 + 4 * cfg['d_ffa'] + 2 + 4 * cfg['d_ffb']


def init_params(M, dev, cfg, g):
    P = _init_e(M, dev, cfg, g)
    P.pop('Wout1', None); P.pop('Wout', None); P.pop('k2', None)
    P['W2b'] = P['W2b'][:, cfg['mlp_to']].detach().clone().requires_grad_()
    return P


def fold(src):
    """Drop k2 and the readout's carry column; k2 moves into the fold bias."""
    P = {k: v.clone() for k, v in src.items()}
    k2 = P.pop('k2', None)
    if k2 is not None and 'b1b' in P:
        P['b1b'] = P['b1b'] + k2 * P['W1b1']
    P.pop('Wout1', None); P.pop('Wout', None)
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
    y = z + torch.einsum('mblf,mf->mbl', hb, P['W2b'])
    U = P['U'][:, None, None, :]
    return 2.0 * y[..., None] * U - U * U
