"""Ensemble-batched mirror of the shipped forward pass.

Training runs E independent members at once: every parameter carries a leading E axis and
the batch is shared, so one kernel launch advances the whole lottery.  `check_mirror()`
asserts this reproduces `model_src.DigitPairAdder` exactly, member by member.
"""
import torch

import model_src

BANK_W = model_src.DigitPairAdder.BANK_W
KEY_W = model_src.DigitPairAdder.KEY_W
LAM = model_src.DigitPairAdder.LAM

PARAM_SHAPES = {"code_free": (8,), "carry_w": (1,), "knee": (2,), "fold": (1,)}
N_PARAMS = sum(int(torch.tensor(s).prod()) for s in PARAM_SHAPES.values())
PARAM_KEYS = tuple(PARAM_SHAPES)


def init_params(e, device, seed=0, dtype=torch.float32):
    """Random restarts.  code[2..9] are drawn i.i.d. across the span the axis will end up
    occupying -- no ordering, no ramp, nothing about which digit means what.  The knees
    are drawn across the whole span x = code[a] + code[b] can occupy."""
    g = torch.Generator(device=device).manual_seed(seed)
    r = lambda *s: torch.rand(*s, device=device, generator=g, dtype=dtype)
    p = {
        "code_free": r(e, 8) * 10.0,                # U(0, 10)
        "carry_w": r(e, 1) * 4.0 - 2.0,             # U(-2, 2)
        "knee": r(e, 2) * 21.0 - 1.0,               # U(-1, 20)
        "fold": r(e, 1) * 24.0 - 12.0,              # U(-12, 12)
    }
    return {k: v.requires_grad_(True) for k, v in p.items()}


def forward_full(p, tok):
    """p: dict of (E, ...) tensors.  tok: (B, P, 2) long, shared by all members.
    Returns the answer axis, the code, and the block's internals."""
    e = p["code_free"].shape[0]
    unit = p["code_free"].new_tensor([0.0, 1.0]).expand(e, 2)
    code = torch.cat([unit, p["code_free"]], dim=1)                 # (E, 10)

    x = code[:, tok[..., 0]] + code[:, tok[..., 1]]                 # (E, B, P)

    u = torch.clamp(BANK_W * (x.unsqueeze(-1) - p["knee"][:, None, None, :]), 0.0, 1.0)
    key = KEY_W * (u[..., 1] - u[..., 0])                           # (E, B, P)
    val = u[..., 1]

    np_ = x.shape[-1]
    idx = torch.arange(np_, device=x.device)
    rel = LAM * (idx[:, None] - idx[None, :]).to(x.dtype)           # (P, P)
    logit = key.unsqueeze(-2) + rel                                 # (E, B, i, j)

    after = idx[:, None] > idx[None, :]
    at_or_after = idx[:, None] >= idx[None, :]
    strict = after | ((idx[:, None] == 0) & (idx[None, :] == 0))

    neg = torch.tensor(-1e30, dtype=x.dtype, device=x.device)
    a_strict = torch.softmax(torch.where(strict, logit, neg), dim=-1)
    a_incl = torch.softmax(torch.where(at_or_after, logit, neg), dim=-1)

    carry_in = (a_strict * val.unsqueeze(-2)).sum(-1)
    carry_out = (a_incl * val.unsqueeze(-2)).sum(-1)

    y = (x + p["carry_w"][:, None, :] * carry_in
         + p["fold"][:, None, :] * carry_out)
    return y, code, key, val


def forward(p, tok):
    """(E, B, P, 10) read-out logits: minus the squared distance to each prototype."""
    y, code, _, _ = forward_full(p, tok)
    return -(y.unsqueeze(-1) - code[:, None, None, :]) ** 2


def loss_and_acc(p, tok, tgt, tau=4.0, margin_w=0.0, margin_target=0.5):
    """Per-member loss, whole-sum exact match, and worst geometric read-out margin.

    The loss is cross entropy on the read-out plus, optionally, a hinge on the *geometric*
    margin -- how much closer the answer axis sits to the right prototype than to any
    other, measured in code units.  Cross entropy alone is happy with a code whose steps
    are barely resolvable; the hinge is what forces `code[a] + code[b]` to land on
    `code[a+b]` squarely, and hence forces the code to become linear on its own.
    """
    y, code, _, _ = forward_full(p, tok)
    e = y.shape[0]
    dist = (y.unsqueeze(-1) - code[:, None, None, :]).abs()         # (E, B, P, 10)
    logits = -dist ** 2

    lsm = torch.log_softmax(tau * logits, dim=-1)
    t = tgt[None, :, :, None].expand(e, -1, -1, 1)
    loss = -lsm.gather(-1, t).squeeze(-1).mean(dim=(1, 2))

    d_true = dist.gather(-1, t).squeeze(-1)                         # (E, B, P)
    d_other = dist.scatter(-1, t, 1e30).amin(-1)                    # (E, B, P)
    gm = d_other - d_true
    if margin_w:
        loss = loss + margin_w * torch.relu(margin_target - gm).mean(dim=(1, 2))

    with torch.no_grad():
        exact = (logits.argmax(-1) == tgt[None]).all(-1).to(y.dtype).mean(-1)
        margin = gm.amin(dim=(1, 2))
    return loss, exact, margin


def check_mirror(device="cpu", e=5, seed=1234):
    """Assert the E-batched forward and the shipped nn.Module agree numerically."""
    import data
    p = init_params(e, device, seed=seed)
    g = torch.Generator(device=device).manual_seed(7)
    tok, _ = data.make_batch(64, 8, device, g)
    ref = forward(p, tok).detach()

    for m_i in range(e):
        mod = model_src.DigitPairAdder().to(device)
        with torch.no_grad():
            for k in PARAM_KEYS:
                getattr(mod, k).copy_(p[k][m_i])
        got = mod(tok).detach()
        d = (got - ref[m_i]).abs().max().item()
        assert d < 1e-4, f"member {m_i}: mirror mismatch {d}"
    return True


if __name__ == "__main__":
    print("params per member:", N_PARAMS)
    print("mirror ok:", check_mirror())
