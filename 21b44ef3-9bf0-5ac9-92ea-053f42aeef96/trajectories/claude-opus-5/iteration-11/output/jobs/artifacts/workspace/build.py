"""Emit /workspace/submission.py: model class + trained weights as float literals.

The graded file must contain the model and its inference path only, and import
nothing but torch, so weights go in as plain Python floats (no base64, no pickle,
no data generation).
"""

import argparse
import json

import torch

import adder

TEMPLATE = '''"""Minimal transformer that adds two 8-digit numbers.

One attention block over per-place digit-pair tokens.  For n digit places the
sequence is n+2 long, LSB first:

    pos 0      (0, 0) pad     anchors "the carry into place 0 is zero"
    pos 1..n   (a_i, b_i)     the i-th place of the two operands
    pos n+1    (0, 0) pad     slot for the final carry-out digit

and position i predicts answer digit i-1, so a single forward pass emits all n+1
answer digits.  The residual stream is one scalar per position: a token embeds as
code[a] + code[b] from a learned 10-entry table, and the same table supplies the
read-out prototypes.

Training discovered this mechanism.  A place is "transparent" when a_i + b_i is
the special value theta (it passes an incoming carry through unchanged), and the
learned code puts every other place at least one clamp-width away from theta.  So
the bank

    upos = clamp(alpha * (x - theta), 0, 1)     uneg = clamp(alpha * (theta - x), 0, 1)

gives upos + uneg = 1 everywhere except a notch to 0 at transparent places, and
upos alone = 1 exactly at the places that generate a carry.  Used as the attention
key, the notch hides transparent places, so each position attends to the nearest
earlier place that is not transparent and reads off whether it generated a carry
-- that is the carry into this place.  The two heads share that one key/value
stream and differ only in their mask: the strictly-causal head returns the carry
in, the inclusively-causal head returns the carry out, and the carry out is what
folds the digit sum back below ten.

Parameters ({NP} of them): the 9 free code entries, theta, and the fold weight e2.
"""

import torch
import torch.nn as nn


class DigitPairAdder(nn.Module):
    def __init__(self):
        super().__init__()
        # --- learned ---------------------------------------------------------
        self.code_free = nn.Parameter(torch.zeros({NFREE}))
        self.theta = nn.Parameter(torch.zeros(()))
        self.e2 = nn.Parameter(torch.zeros(()))
        # --- fixed -----------------------------------------------------------
        # code_pin: origin of the 1-D residual stream (a coordinate choice).
        # e1:       scale of that stream (a coordinate choice).
        # alpha, kw, lam: shape constants with wide working bands, see NOTES.md.
        self.register_buffer("code_pin", torch.tensor({CODE_PIN}))
        self.register_buffer("e1", torch.tensor({E1}))
        self.register_buffer("alpha", torch.tensor({ALPHA}))
        self.register_buffer("kw", torch.tensor({KW}))
        self.register_buffer("lam", torch.tensor({LAM}))

    def code(self):
        return torch.cat([self.code_pin, self.code_free])

    def forward(self, ab):
        """ab: int64 [B, P, 2] digit pairs -> logits [B, P, 10]."""
        P = ab.shape[1]
        code = self.code()
        x = code[ab].sum(-1)                                     # [B, P]

        t = self.alpha * (x - self.theta)
        upos = t.clamp(0.0, 1.0)
        uneg = (-t).clamp(0.0, 1.0)
        key = self.kw * (upos + uneg)                            # [B, P]

        idx = torch.arange(P, device=ab.device)
        dist = (idx[:, None] - idx[None, :]).to(x.dtype)
        neg = torch.finfo(x.dtype).min / 4
        strict = idx[None, :] < idx[:, None]
        strict = strict.clone()
        strict[0, 0] = True              # row 0 has no earlier place; it reads the pad
        incl = idx[None, :] <= idx[:, None]
        base = key[:, None, :] + self.lam * dist

        val = upos[:, None, :]
        a1 = torch.softmax(base.masked_fill(~strict, neg), dim=-1)
        a2 = torch.softmax(base.masked_fill(~incl, neg), dim=-1)
        o1 = (a1 * val).sum(-1)                                  # carry in
        o2 = (a2 * val).sum(-1)                                  # carry out

        y = x + self.e1 * o1 + self.e2 * o2
        d = y[..., None] - code
        return -d * d


_CODE_FREE = {CODE_FREE}
_THETA = {THETA}
_E2 = {E2}


def build_model():
    model = DigitPairAdder()
    with torch.no_grad():
        model.code_free.copy_(torch.tensor(_CODE_FREE))
        model.theta.copy_(torch.tensor(_THETA))
        model.e2.copy_(torch.tensor(_E2))
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    meta = {
        "name": "digit-pair-adder",
        "n_parameters": n,
        "architecture": "1 block: clamp bank -> 2-head causal self-attention "
                        "(shared key/value) -> tied 1-D digit-code read-out",
        "digits": 8,
    }
    return model, meta


@torch.no_grad()
def add(model, a: int, b: int) -> int:
    """Exact sum of two integers, decoded from one forward pass of `model`."""
    a, b = int(a), int(b)
    n = max(len(str(a)), len(str(b)), 1)
    da = [(a // 10 ** i) % 10 for i in range(n)]
    db = [(b // 10 ** i) % 10 for i in range(n)]
    dev = next(model.parameters()).device
    ab = torch.zeros(1, n + 2, 2, dtype=torch.long, device=dev)
    ab[0, 1:n + 1, 0] = torch.tensor(da, dtype=torch.long, device=dev)
    ab[0, 1:n + 1, 1] = torch.tensor(db, dtype=torch.long, device=dev)
    digits = model(ab).argmax(-1)[0, 1:n + 2].tolist()
    return sum(d * 10 ** i for i, d in enumerate(digits))
'''


def fmt(x):
    return repr(float(x))


def fmt_list(v):
    return "[" + ", ".join(fmt(x) for x in v) + "]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--member", type=int, default=0)
    ap.add_argument("--out", default="/workspace/submission.py")
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="ship this alpha instead of the checkpoint's; it must lie "
                         "strictly inside the band derived from the weights")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = adder.default_cfg(**ck["cfg"])
    p = {k: v[args.member] for k, v in ck["params"].items()}
    if args.alpha > 0:
        cfg = dict(cfg, alpha=args.alpha)
        p = dict(p, alpha=torch.tensor(args.alpha))
    nfix = cfg["code_fix"]
    assert nfix == 1, "the shipped model pins exactly one code entry"
    assert not cfg["free_theta_neg"], "the shipped model has one shared threshold"
    assert abs(float(p["rb"])) < 1e-9, "residual bias must be gauged to zero"

    # the fixed constants ship as the exact round values they are meant to be;
    # the checkpoint may hold float32 residue from the gauge division
    exact = {"e1": 1.0, "alpha": cfg["alpha"], "kw": cfg["kw"], "lam": cfg["lam"]}
    for k, v in exact.items():
        got = float(p[k])
        assert abs(got - v) <= 1e-5 * max(1.0, abs(v)), (k, got, v)
        p[k] = torch.tensor(v)
    assert abs(float(p["code"][0])) < 1e-9, "code[0] must be gauged to zero"
    p["code"] = p["code"].clone()
    p["code"][0] = 0.0

    # the shipped constants must not merely work, they must be provably inside the
    # admissible region implied by the learned weights
    from certify import certify
    r = certify(p["code"], p["theta"], p["e2"], p["e1"], p["alpha"], p["kw"],
                p["lam"], P=10, verbose=False)
    assert r["ok"], "member does not certify at the shipped constants"
    a = float(p["alpha"])
    assert r["alpha_lo"] * 1.1 < a < r["alpha_hi"] / 1.1, (
        f"alpha {a} is not comfortably inside the derived band "
        f"[{r['alpha_lo']:.3f}, {r['alpha_hi']:.3f}]")

    code = p["code"].tolist()
    src = TEMPLATE
    subs = {
        "{NP}": str(adder.n_params(cfg)),
        "{NFREE}": str(10 - nfix),
        "{CODE_PIN}": fmt_list(code[:nfix]),
        "{CODE_FREE}": fmt_list(code[nfix:]),
        "{THETA}": fmt(p["theta"]),
        "{E2}": fmt(p["e2"]),
        "{E1}": fmt(p["e1"]),
        "{ALPHA}": fmt(p["alpha"]),
        "{KW}": fmt(p["kw"]),
        "{LAM}": fmt(p["lam"]),
    }
    for k, v in subs.items():
        src = src.replace(k, v)
    with open(args.out, "w") as f:
        f.write(src)
    print("wrote", args.out, "n_params", adder.n_params(cfg),
          "member acc", float(ck["acc"][args.member]))


if __name__ == "__main__":
    main()
