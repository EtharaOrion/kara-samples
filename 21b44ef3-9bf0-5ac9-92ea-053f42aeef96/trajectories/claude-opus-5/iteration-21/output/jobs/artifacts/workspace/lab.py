"""Training-side machinery: data sampler and an ensemble-batched mirror of the model.

Nothing in here is shipped.  The mirror `ens_forward` computes exactly the same
function as model_src.DigitPairAdder.forward, but with a leading ensemble axis E
so that E independent members train simultaneously on one GPU.  `check_mirror`
asserts the two agree numerically.
"""

import torch

PIN = (0.0, 1.0)        # code[0], code[1]: origin and unit of the residual axis
GATE_SLOPE = 8.0
KEY_SCALE = 400.0
RECENCY = -12.0
KNEE_LO, KNEE_HI = -1.0, 19.0

PARAM_SHAPES = {"code_free": (8,), "carry_w": (), "knee": (2,), "fold": ()}
N_PARAMS = 12


def is_exact(rate):
    """A per-member exact-match rate that means "got every one of them".

    Not `rate == 1`.  These rates are means of boolean tensors, and for most
    batch sizes float32 rounds n * (1 / n) to either side of one, so an exact
    member reads back as 0.99999994 and an equality test silently discards it.
    """
    return rate >= 1.0 - 1e-6


# --------------------------------------------------------------------------- data

def _value(d):
    """(B, n) digits, least significant first -> (B,) integer value."""
    p10 = torch.pow(10, torch.arange(d.shape[1], device=d.device, dtype=torch.int64))
    return (d.to(torch.int64) * p10).sum(1)


def bucket(a, b):
    """Deterministic 1-in-16 hash bucket of an operand pair."""
    h = torch.bitwise_xor(_value(a) * 1000003, _value(b) * 15485863)
    return torch.remainder(h, 16)


def answer_digits(a, b):
    """(B, n) digit pairs -> (B, n + 1) answer digits, least significant first."""
    s = a + b
    carry = torch.zeros_like(s[:, 0])
    out = []
    for i in range(s.shape[1]):
        t = s[:, i] + carry
        out.append(torch.remainder(t, 10))
        carry = torch.div(t, 10, rounding_mode="floor")
    out.append(carry)
    return torch.stack(out, dim=1)


# per-sample regime: (probability, transparent rate, generate rate)
REGIMES = ((0.35, 0.0, 0.0), (0.40, 0.40, 0.25), (0.25, 0.85, 0.10))


def raw_pairs(B, n, device, full_width=0.5, nocarry=0.0):
    """Digit pairs with carry structure mixed in, so carries chain.

    `nocarry` is the fraction of samples drawn with a + b <= 9 in every place,
    i.e. problems where no carry ever happens.  Those are the easy end of the
    curriculum: they ask only for a digit code that adds.
    """
    r = torch.rand(B, 1, device=device)
    p_tr = torch.zeros(B, 1, device=device)
    p_gen = torch.zeros(B, 1, device=device)
    lo = 0.0
    for prob, tr, gen in REGIMES:
        sel = (r >= lo) & (r < lo + prob + 1e-9)
        p_tr = torch.where(sel, torch.full_like(p_tr, tr), p_tr)
        p_gen = torch.where(sel, torch.full_like(p_gen, gen), p_gen)
        lo += prob

    a = torch.randint(0, 10, (B, n), device=device)
    b = torch.randint(0, 10, (B, n), device=device)
    u = torch.rand(B, n, device=device)

    transparent = u < p_tr
    generate = (u >= p_tr) & (u < p_tr + p_gen)
    # a generating place needs a >= 1, then b uniform on [10 - a, 9]
    a_gen = torch.randint(1, 10, (B, n), device=device)
    b_gen = (10 - a_gen) + (torch.rand(B, n, device=device) * a_gen).floor().long()
    a = torch.where(generate, a_gen, a)
    b = torch.where(generate, b_gen, torch.where(transparent, 9 - a, b))

    if nocarry > 0:
        easy = torch.rand(B, 1, device=device) < nocarry
        b_easy = (torch.rand(B, n, device=device) * (10 - a)).floor().long()
        b = torch.where(easy, b_easy, b)

    if full_width > 0 and n > 1:
        force = torch.rand(B, 1, device=device) < full_width
        top = torch.zeros(B, n, dtype=torch.bool, device=device)
        top[:, n - 1] = True
        keep = force & top
        a = torch.where(keep, a.clamp(min=1), a)
        b = torch.where(keep, b.clamp(min=1), b)
    return a, b


def batch(B, n, device, split="train", full_width=0.5, nocarry=0.0):
    """Sampled batch restricted to the requested hash split."""
    if split == "any":
        a, b = raw_pairs(B, n, device, full_width, nocarry)
        return torch.stack([a, b], dim=-1), answer_digits(a, b)
    want_heldout = split == "eval"
    keep_a = torch.zeros(0, n, dtype=torch.long, device=device)
    keep_b = torch.zeros(0, n, dtype=torch.long, device=device)
    for _ in range(40):
        a, b = raw_pairs(max(B * 2, 1024), n, device, full_width, nocarry)
        m = (bucket(a, b) == 0) if want_heldout else (bucket(a, b) != 0)
        keep_a = torch.cat([keep_a, a[m]])
        keep_b = torch.cat([keep_b, b[m]])
        if keep_a.shape[0] >= B:
            break
    a, b = keep_a[:B], keep_b[:B]
    return torch.stack([a, b], dim=-1), answer_digits(a, b)


def all_single_place(device):
    """All 100 one-place problems."""
    d = torch.arange(10, device=device)
    a = d.repeat_interleave(10).unsqueeze(1)
    b = d.repeat(10).unsqueeze(1)
    return torch.stack([a, b], dim=-1), answer_digits(a, b)


# --------------------------------------------------------------------------- model mirror

def init_params(E, device, gen=None, free_unit=True):
    """Random independent members.

    `free_unit` leaves code[1] trainable, so the residual axis has a free scale
    during training.  That matters because the gate bank saturates: its two
    thresholds get essentially no gradient and stay near wherever they start, so
    something has to adapt to them, and with a free scale the code can.  The
    scale is spent afterwards (rescale.py) to reach the 12-parameter form.

    Both thresholds start positive because the padding token sits at the origin
    of the residual axis and has to fall below both of them, and knee[0] starts
    below knee[1] purely as a naming convention for the two gate units.
    """
    n_free = 9 if free_unit else 8
    hi = torch.empty(E, device=device).uniform_(0.5, 12.0, generator=gen)
    lo = hi * torch.empty(E, device=device).uniform_(0.3, 1.0, generator=gen)
    p = {
        "code_free": torch.randn(E, n_free, device=device, generator=gen) * 4.0,
        "carry_w": torch.empty(E, device=device).uniform_(-2, 2, generator=gen),
        "knee": torch.stack([lo, hi], dim=1),
        "fold": torch.randn(E, device=device, generator=gen) * 6.0,
        # Training-only readout temperature.  The shipped model has no such
        # value: multiplying every logit by a positive constant cannot change an
        # argmax, so this affects the loss surface only, never a prediction.
        # It exists so that "become less confident" is available to the
        # optimiser as an alternative to "shrink the digit code", which is the
        # collapse that otherwise eats every member.
        "lsr": torch.zeros(E, device=device),
    }
    return {k: v.requires_grad_(True) for k, v in p.items()}


SHIP_KEYS = ("code_free", "carry_w", "knee", "fold")


def code_of(p):
    """(E, 10) digit code.  code[0] is always the origin of the residual axis;
    code[1] is the unit and is trainable until the scale gauge is spent."""
    cf = p["code_free"]
    E = cf.shape[0]
    head = cf.new_zeros(E, 1)
    if cf.shape[1] == 8:
        head = torch.cat([head, cf.new_ones(E, 1)], dim=1)
    return torch.cat([head, cf], dim=1)                     # (E, 10)


def masks(places, device, dtype=torch.float32):
    pos = torch.arange(places, device=device)
    gap = pos.unsqueeze(1) - pos.unsqueeze(0)
    blocked = torch.finfo(dtype).min
    future = torch.where(gap < 0, blocked, 0.0).to(dtype)
    strict = torch.where(gap <= 0, blocked, 0.0).to(dtype)
    strict[0, 0] = 0.0
    return gap.to(dtype), strict, future


def ens_forward(p, pairs, square=True, temp=1.0, want_gates=False, norm=False):
    """(E, ...) mirror of DigitPairAdder.forward.

    pairs: (B, n, 2) shared across members.  Returns (E, B, P, 10) logits.
    """
    code = code_of(p)                                        # (E, 10)
    E = code.shape[0]
    x = code[:, pairs[..., 0]] + code[:, pairs[..., 1]]      # (E, B, n)
    pad = x.new_zeros(E, x.shape[1], 1)
    x = torch.cat([pad, x, pad], dim=2)                      # (E, B, P)
    places = x.shape[2]

    gate = torch.clamp(GATE_SLOPE * (x.unsqueeze(-1) - p["knee"].view(E, 1, 1, 2)), 0.0, 1.0)
    value = gate[..., 1]
    key = KEY_SCALE * (gate[..., 1] - gate[..., 0])

    gap, strict, future = masks(places, x.device, x.dtype)
    scores = key.unsqueeze(2) + RECENCY * gap                # (E, B, P, P)
    w_in = torch.softmax(scores + strict, dim=-1)
    w_out = torch.softmax(scores + future, dim=-1)
    carry_in = torch.einsum("ebij,ebj->ebi", w_in, value)
    carry_out = torch.einsum("ebij,ebj->ebi", w_out, value)

    z = x + p["carry_w"].view(E, 1, 1) * carry_in + p["fold"].view(E, 1, 1) * carry_out
    d = z.unsqueeze(-1) - code[:, None, None, :]
    if norm:
        # Measure read-out distance in units of the code's own spread.  This
        # makes the training loss exactly invariant to rescaling the residual
        # axis, so "shrink the digit code" stops being a descent direction --
        # it is the collapse that otherwise eats every member.  The shipped
        # model needs no such term: an argmax over distances does not care.
        d = d / code.std(dim=1).clamp_min(1e-4).view(E, 1, 1, 1)
    scale = temp
    if "lsr" in p:
        scale = temp * torch.exp(p["lsr"].clamp(-6.0, 6.0)).view(E, 1, 1, 1)
    logits = -(d.pow(2) if square else d.abs()) * scale
    if want_gates:
        return logits, gate
    return logits


def check_mirror(device="cuda"):
    """Assert the ensemble mirror equals the shipped module."""
    import model_src

    torch.manual_seed(0)
    m = model_src.DigitPairAdder().to(device)
    with torch.no_grad():
        for q in m.parameters():
            q.copy_(torch.randn_like(q))
    p = {
        "code_free": m.code_free.detach()[None],
        "carry_w": m.carry_w.detach()[None],
        "knee": m.knee.detach()[None],
        "fold": m.fold.detach()[None],
    }
    pairs, _ = batch(64, 8, device, split="any")
    with torch.no_grad():
        a = m(pairs)
        b = ens_forward(p, pairs)[0]
    err = (a - b).abs().max().item()
    assert err < 1e-4, f"mirror mismatch {err}"
    return err
