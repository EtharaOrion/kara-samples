"""Architecture for the digit-pair adder, written as a pure function over a
parameter dict whose tensors all carry a leading ensemble axis E.

Training E independent members at once (each with its own random init) is the
only practical way to search this parameter regime: at a few dozen parameters
the outcome is dominated by seed luck, so we run a large "seed lottery" in a
single batched forward pass.

Token layout (LSB first), for n-digit operands:

    pos 0        : pad, digit pair (0, 0)
    pos 1 .. n   : digit pair (a_i, b_i) for place i-1
    pos n+1      : pad, digit pair (0, 0)  -- carries the final carry out

Position p predicts answer digit p-1, so the whole sum comes from a single
forward pass.

The residual stream is C-dimensional (C=1 in the shipped model).  A place token
embeds as code[a] + code[b] from one learned 10-entry table that also serves as
the read-out prototypes.  A small clamp bank turns that scalar into a
content-dependent key/value pair, which two masked attention heads read:

    head A (strictly causal, j < i)   -> carry INTO place i
    head B (inclusively causal, j<=i) -> carry OUT of place i

Read-out is negative squared distance to the ten prototypes.
"""

import torch

PARAM_NAMES = ("code", "Wb", "bb", "kw", "vw", "vb", "uA", "uB", "q", "lam", "ls")


def shapes(C, U):
    return {
        "code": (10, C),   # digit -> residual code, also read-out prototypes
        "Wb": (U, C),      # bank input weights
        "bb": (U,),        # bank biases (knee positions)
        "kw": (U,),        # bank -> attention key
        "vw": (U,),        # bank -> attention value
        "vb": (),          # value bias
        "uA": (C,),        # write-back direction for carry-in  (head A)
        "uB": (C,),        # write-back direction for carry-out (head B)
        "q": (),           # query scale on the key
        "lam": (),         # relative-distance bias slope
        "ls": (),          # read-out temperature
    }


def init_params(E, C, U, device, gen, scale=1.0):
    """Random init for E independent members."""
    def rn(*shape, s=1.0):
        return torch.randn(E, *shape, generator=gen, device=device) * s * scale

    p = {
        "code": rn(10, C, s=0.6),
        "Wb": rn(U, C, s=1.0),
        "bb": rn(U, s=1.0),
        "kw": rn(U, s=1.0),
        "vw": rn(U, s=1.0),
        "vb": torch.zeros(E, device=device),
        "uA": rn(C, s=0.6),
        "uB": rn(C, s=0.6),
        "q": torch.ones(E, device=device),
        "lam": torch.full((E,), -0.5, device=device) + 0.2 * rn(s=1.0),
        "ls": torch.full((E,), 2.0, device=device),
    }
    return p


def masks(P, device):
    """Attention masks.  Row 0 of the strict mask has no legal predecessor;
    it is allowed to attend to itself (a (0,0) pad, value 0) so the softmax is
    well defined.  Position 0's prediction is never used."""
    i = torch.arange(P, device=device)
    dist = (i[:, None] - i[None, :]).float()
    mA = dist > 0
    mA = mA.clone()
    mA[0, 0] = True
    mB = dist >= 0
    return dist, mA, mB


def forward(p, a, b, want_y=False):
    """a, b: int64 [B, P] digit tensors.  Returns logits [E, B, P, 10]."""
    code = p["code"]                                  # [E,10,C]
    x = code[:, a] + code[:, b]                       # [E,B,P,C]

    e = torch.einsum("euc,ebpc->ebpu", p["Wb"], x) + p["bb"][:, None, None, :]
    g = e.clamp(0.0, 1.0)                             # [E,B,P,U]

    k = (g * p["kw"][:, None, None, :]).sum(-1)       # [E,B,P]
    v = (g * p["vw"][:, None, None, :]).sum(-1) + p["vb"][:, None, None]

    P = a.shape[-1]
    dist, mA, mB = masks(P, a.device)
    s = (p["q"][:, None, None, None] * k[:, :, None, :]
         + p["lam"][:, None, None, None] * dist)      # [E,B,P(query),P(key)]

    neg = torch.finfo(s.dtype).min
    wA = torch.softmax(s.masked_fill(~mA, neg), -1)
    wB = torch.softmax(s.masked_fill(~mB, neg), -1)
    cA = (wA * v[:, :, None, :]).sum(-1)              # [E,B,P] carry in
    cB = (wB * v[:, :, None, :]).sum(-1)              # [E,B,P] carry out

    y = (x + cA[..., None] * p["uA"][:, None, None, :]
           + cB[..., None] * p["uB"][:, None, None, :])

    d2 = ((y[:, :, :, None, :] - code[:, None, None, :, :]) ** 2).sum(-1)
    logits = -p["ls"][:, None, None, None] * d2
    return (logits, y) if want_y else logits


def attn_maps(p, a, b):
    """Same computation, returning the two attention maps (for auditing)."""
    code = p["code"]
    x = code[:, a] + code[:, b]
    e = torch.einsum("euc,ebpc->ebpu", p["Wb"], x) + p["bb"][:, None, None, :]
    g = e.clamp(0.0, 1.0)
    k = (g * p["kw"][:, None, None, :]).sum(-1)
    P = a.shape[-1]
    dist, mA, mB = masks(P, a.device)
    s = (p["q"][:, None, None, None] * k[:, :, None, :]
         + p["lam"][:, None, None, None] * dist)
    neg = torch.finfo(s.dtype).min
    return (torch.softmax(s.masked_fill(~mA, neg), -1),
            torch.softmax(s.masked_fill(~mB, neg), -1))


def n_free(C, U):
    return sum(int(torch.tensor(s).prod()) if s else 1 for s in shapes(C, U).values())
