"""Ensemble mirror of the shipped forward pass.

`forward_ens` computes exactly what `model_src.DigitPairAdder.forward` computes,
but for E independent members at once (leading axis E).  Adam is elementwise,
so E members stacked this way train completely independently -- which is what
makes a large random-restart lottery affordable.  `check_mirror` asserts the two
implementations agree bit-for-bit-ish on random weights.
"""

import torch

CONST = {"code0": 0.0, "code1": 1.0, "bank_w": 8.0, "key_w": 400.0, "lam": -12.0}
PARAM_SHAPES = {"code_free": (8,), "carry_w": (), "knee": (2,), "fold": ()}


def stream_ens(params, tokens, const=CONST):
    """params: dict of (E, ...) tensors; tokens: (B,P,2) -> stream (E,B,P), code (E,10)."""
    code_free = params["code_free"]
    e = code_free.shape[0]
    dev = code_free.device
    head = torch.tensor([const["code0"], const["code1"]], device=dev,
                        dtype=code_free.dtype).expand(e, 2)
    code = torch.cat([head, code_free], dim=1)

    x = code[:, tokens[..., 0]] + code[:, tokens[..., 1]]
    gate = torch.clamp(const["bank_w"] * (x.unsqueeze(-1) - params["knee"][:, None, None, :]),
                       0.0, 1.0)
    value = gate[..., 1]
    key = const["key_w"] * (gate[..., 1] - gate[..., 0])

    p = tokens.shape[-2]
    pos = torch.arange(p, device=dev)
    delta = pos.unsqueeze(1) - pos.unsqueeze(0)
    logits = key.unsqueeze(-2) + const["lam"] * delta
    blocked = torch.finfo(logits.dtype).min
    earlier = (delta > 0) | ((pos.unsqueeze(1) == 0) & (pos.unsqueeze(0) == 0))
    upto = delta >= 0
    w_in = torch.softmax(torch.where(earlier, logits, blocked), dim=-1)
    w_out = torch.softmax(torch.where(upto, logits, blocked), dim=-1)
    carry_in = (w_in * value.unsqueeze(-2)).sum(-1)
    carry_out = (w_out * value.unsqueeze(-2)).sum(-1)

    stream = (x + params["carry_w"][:, None, None] * carry_in
              + params["fold"][:, None, None] * carry_out)
    return stream, code


def digit_logits(params, tokens, scale=1.0, const=CONST):
    stream, code = stream_ens(params, tokens, const)
    return -scale * (stream.unsqueeze(-1) - code[:, None, None, :]) ** 2


def init_params(e, device, gen, code_sigma=3.0, knee_lo=-10.0, knee_hi=20.0,
                carry_w=2.0, fold=12.0, dtype=torch.float32):
    def u(shape, lo, hi):
        return torch.rand(*shape, device=device, generator=gen, dtype=dtype) * (hi - lo) + lo
    return {
        "code_free": torch.randn(e, 8, device=device, generator=gen, dtype=dtype) * code_sigma,
        "carry_w": u((e,), -carry_w, carry_w),
        "knee": u((e, 2), knee_lo, knee_hi),
        "fold": u((e,), -fold, fold),
    }


def check_mirror(device="cpu", seed=0):
    """Assert the ensemble mirror reproduces the shipped module exactly."""
    import model_src
    torch.manual_seed(seed)
    e, b, p = 5, 7, 10
    gen = torch.Generator(device=device).manual_seed(seed)
    params = init_params(e, device, gen)
    tokens = torch.randint(0, 10, (b, p, 2), device=device)
    ref = torch.empty(e, b, p, 10, device=device)
    for i in range(e):
        m = model_src.DigitPairAdder().to(device)
        with torch.no_grad():
            m.code_free.copy_(params["code_free"][i])
            m.carry_w.copy_(params["carry_w"][i])
            m.knee.copy_(params["knee"][i])
            m.fold.copy_(params["fold"][i])
            ref[i] = m(tokens)
    got = digit_logits(params, tokens)
    err = (ref - got).abs().max().item()
    assert err == 0.0, f"mirror mismatch {err}"
    return err


if __name__ == "__main__":
    print("mirror max abs error:", check_mirror("cpu"), check_mirror("cuda"))
