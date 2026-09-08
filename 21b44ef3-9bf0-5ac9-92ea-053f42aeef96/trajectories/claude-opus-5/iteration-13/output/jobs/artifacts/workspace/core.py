"""Architecture: single-block digit-pair transformer with a 1-D (scalar) residual stream.

This is the *training-side* functional mirror of the model that ships in submission.py.
Every tensor in the parameter dict carries a leading ensemble axis E so that E
independent members train simultaneously on one GPU (parameters never mix across E;
the only shared thing is the data batch).

Token layout (LSB-first), for n digit places:
    P = n + 2 positions.
    pos 0        : (0, 0) pad  -> supplies "carry into place 0 is absent"
    pos 1 .. n   : digit pair (a_i, b_i) of place i-1
    pos n+1      : (0, 0) pad  -> emits the leading carry digit
Position i predicts the answer digit of place i-1, so one forward pass yields all
n+1 answer digits.

Residual stream (scalar per position):
    u_i = code[a_i] + code[b_i] + rb

Bank (U clamp units, the only nonlinearity):
    g_i = clamp(bw * u_i + bb, 0, 1)                       [U]

One shared key/value stream feeding two masked heads:
    k_i = <g_i, kw>            v_i = <g_i, vw> + vb
    logit[h,i,j] = q * k_j + lam[h] * (i - j)      for j in mask[h]
    head 0 = strictly causal (j < i), head 1 = inclusively causal (j <= i)
    c[h,i]  = sum_j softmax(logit[h,i,:])_j * v_j

Read-out (tied: the code table is both the embedding and the prototype set):
    z_i        = u_i + e[0] * c[0,i] + e[1] * c[1,i]
    logits_i,d = -ls * (z_i - code[d])^2
"""

import torch


def make_geom(P, device):
    """Distance matrix and the two head masks. Pure architecture, no parameters."""
    i = torch.arange(P, device=device)
    dist = (i[:, None] - i[None, :]).to(torch.float32)      # i - j
    strict = dist >= 1.0
    # row 0 has no strictly-earlier position; let it look at itself so the softmax is
    # well defined. pos 0 is a (0,0) pad whose own prediction is never read, and no
    # other row is affected.
    strict = strict.clone()
    strict[0, 0] = True
    incl = dist >= 0.0
    return dist, torch.stack([strict, incl], 0)             # [2,P,P]


def forward_core(p, A, B, need_attn=False):
    """p: dict of parameters, each [E, ...]. A, B: int64 [Bs, P] digit ids.

    Returns logits [E, Bs, P, 10] (and optionally the attention maps).
    """
    dist, mask = make_geom(A.shape[1], A.device)
    dt = p["code"].dtype

    u = p["code"][:, A] + p["code"][:, B] + p["rb"][:, None, None]          # [E,Bs,P]
    g = torch.clamp(u.unsqueeze(-1) * p["bw"][:, None, None, :]
                    + p["bb"][:, None, None, :], 0.0, 1.0)                  # [E,Bs,P,U]
    k = (g * p["kw"][:, None, None, :]).sum(-1)                             # [E,Bs,P]
    v = (g * p["vw"][:, None, None, :]).sum(-1) + p["vb"][:, None, None]    # [E,Bs,P]

    lg = (p["q"][:, None, None, None, None] * k[:, :, None, None, :]
          + p["lam"][:, None, :, None, None] * dist.to(dt)[None, None, None])  # [E,Bs,2,P,P]
    lg = lg.masked_fill(~mask[None, None], torch.finfo(dt).min / 4)
    at = torch.softmax(lg, dim=-1)
    c = (at * v[:, :, None, None, :]).sum(-1)                               # [E,Bs,2,P]

    z = u + p["e"][:, 0, None, None] * c[:, :, 0, :] + p["e"][:, 1, None, None] * c[:, :, 1, :]
    out = -p["ls"][:, None, None, None] * (z.unsqueeze(-1) - p["code"][:, None, None, :]) ** 2
    if need_attn:
        return out, at, z, k, v, u
    return out


# ---------------------------------------------------------------- parameter set

# name -> trailing shape (U is substituted at build time)
SPEC = {
    "code": (10,),
    "rb":   (),
    "bw":   ("U",),
    "bb":   ("U",),
    "kw":   ("U",),
    "vw":   ("U",),
    "vb":   (),
    "q":    (),
    "lam":  (2,),
    "e":    (2,),
    "ls":   (),
}


def shapes(U):
    return {k: tuple(U if s == "U" else s for s in sh) for k, sh in SPEC.items()}


def init_params(E, U, device, gen, dtype=torch.float32):
    """Random init. Bank pre-activations start inside (0,1) so clamp gradients are alive."""
    def rn(*sh, scale=1.0):
        return torch.randn(*sh, generator=gen, device=device, dtype=dtype) * scale

    p = {
        "code": rn(E, 10, scale=0.8),
        "rb":   rn(E, scale=0.3),
        "bw":   rn(E, U, scale=0.35),
        "bb":   0.5 + rn(E, U, scale=0.25),
        "kw":   rn(E, U, scale=1.0),
        "vw":   rn(E, U, scale=1.0),
        "vb":   rn(E, scale=0.2),
        "q":    torch.ones(E, device=device, dtype=dtype),
        "lam": -torch.rand(E, 2, generator=gen, device=device, dtype=dtype) * 1.5 - 0.1,
        "e":    rn(E, 2, scale=1.0),
        "ls":   torch.ones(E, device=device, dtype=dtype),
    }
    return p


def n_free(p, frozen):
    """Number of scalar entries per ensemble member that are trainable parameters."""
    tot = 0
    for k, t in p.items():
        if k in frozen:
            continue
        tot += t[0].numel()
    return tot
