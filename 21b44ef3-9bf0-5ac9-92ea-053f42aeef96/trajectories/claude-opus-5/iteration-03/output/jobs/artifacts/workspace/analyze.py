"""Read out what the trained model actually learned.

Two things are printed:

  * the geometry of the ten digit codes in the readout plane, together with a
    least-squares fit to the moment curve (t, t^2), and
  * the attention distribution on a hand-made carry chain, next to the place
    each query *should* be looking at.

Nothing here is used by the submission; it exists so the mechanism can be
checked rather than assumed.
"""
import argparse
import importlib.util

import torch


def load(path):
    spec = importlib.util.spec_from_file_location("cand", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    model, _ = m.build_model()
    return m, model


def digit_geometry(model):
    e = model._emb().detach()
    d = torch.arange(10, dtype=torch.float32)
    print("digit codes in the readout plane")
    print(f"{'digit':>5} {'x':>9} {'y':>9}")
    for i in range(10):
        print(f"{i:>5} {e[i, 0]:9.4f} {e[i, 1]:9.4f}")

    # Fit x ~ a*d + b and y ~ c*d^2 + e*d + f, and report how much of each
    # coordinate the fit explains.
    def fit(y, cols):
        A = torch.stack(cols, 1)
        coef = torch.linalg.lstsq(A, y.unsqueeze(1)).solution.squeeze(1)
        resid = y - A @ coef
        r2 = 1 - resid.var() / y.var()
        return coef, float(r2)

    one = torch.ones(10)
    cx, r2x = fit(e[:, 0], [d, one])
    cy, r2y = fit(e[:, 1], [d * d, d, one])
    print(f"\nx = {cx[0]:+.4f}*digit {cx[1]:+.4f}          R^2 = {r2x:.5f}")
    print(f"y = {cy[0]:+.4f}*digit^2 {cy[1]:+.4f}*digit {cy[2]:+.4f}  "
          f"R^2 = {r2y:.5f}")
    print("\nThe first coordinate is linear in the digit, so the token "
          "emb[a]+emb[b]\ncarries a+b directly.  The second is quadratic, "
          "which is what puts the ten\ncodes on a convex curve: every code is "
          "then a vertex of their convex hull,\nso a linear readout can select "
          "any digit.  A collinear code could not --\nits arg max would be "
          "monotone and only the two extreme digits could win.")

    # How cleanly does the token's first coordinate encode a+b?  Compare the
    # spread of emb[a].x + emb[b].x within a fixed a+b against the step
    # between neighbouring values of a+b.
    x = e[:, 0]
    tok = x.view(-1, 1) + x.view(1, -1)
    tot = torch.arange(10).view(-1, 1) + torch.arange(10).view(1, -1)
    spread = max(float(tok[tot == s].max() - tok[tot == s].min())
                 for s in range(19))
    print(f"\ntoken x-coordinate vs a+b: within one value of a+b it varies by "
          f"at most {spread:.4f},\nagainst a step of {float(x[1] - x[0]):.4f} "
          f"per unit of a+b -- so the pre-attention\nFFN sees a+b, and only "
          f"a+b, along this direction.")


def relu_roles(model, n=40000, seed=5):
    """What do the two single-unit FFNs actually compute?

    The carry-lookahead algorithm needs exactly two predicates: whether a place
    is transparent (a+b == 9), which is what attention has to key on, and
    whether a place overflows (a+b+carry >= 10), which is the mod-10 wrap in
    the answer.  There is one ReLU available for each.  This tabulates each
    ReLU's pre-activation against the predicate it ought to be computing.
    """
    from data import sample
    g = torch.Generator().manual_seed(seed)
    da, db = sample(n, "cpu", g)
    s = da + db

    carry = torch.zeros_like(s)
    c = torch.zeros(s.shape[0], dtype=s.dtype)
    for i in range(s.shape[1]):
        carry[:, i] = c
        c = ((s[:, i] + c) >= 10).to(s.dtype)

    with torch.no_grad():
        x = model.embed(da, db)
        h = model._n(x)
        pre = (h[..., :1] if model.ffn_in_axis
               else h[..., :model.d_io] @ model.w_in1)
        z_in = pre + model.b_in1
        d = model._nl(z_in) @ model.w_in2
        x = x + (d + model.b_in2 if model.rb_in else d)
        h = model._n(x)
        o = model.attend(h) @ h
        x = x + o
        z_out = model._n(x) @ model.w_out1 + model.b_out1

    zi = z_in[:, 1:9, 0]
    zo = z_out[:, 1:9, 0]

    print("\npre-attention ReLU, by a+b at that place "
          "(it should isolate a+b == 9):")
    print(f"{'a+b':>5} {'mean pre-act':>13} {'fires':>8}")
    for v in range(19):
        m = s == v
        if m.any():
            print(f"{v:>5} {float(zi[m].mean()):13.3f} "
                  f"{float((zi[m] > 0).float().mean()):7.1%}")

    print("\npost-attention ReLU, by (a+b, carry-in) "
          "(it should isolate a+b+carry >= 10):")
    print(f"{'a+b':>5} {'carry 0':>18} {'carry 1':>18}")
    for v in range(19):
        row = f"{v:>5}"
        for cv in (0, 1):
            m = (s == v) & (carry == cv)
            row += (f"   {float(zo[m].mean()):8.3f} {float((zo[m] > 0).float().mean()):6.1%}"
                    if m.any() else f"{'-':>18}")
        print(row)

    fire_i = (zi > 0)
    want_i = (s == 9)
    fire_o = (zo > 0)
    want_o = ((s + carry) >= 10)
    print(f"\npre-attention ReLU fires iff a+b >= 10 on "
          f"{float((zi > 0).eq((s >= 10)).float().mean()):.4%} of places")
    print(f"  (it does *not* isolate a+b == 9: {float((fire_i == want_i).float().mean()):.2%})")
    print(f"post-attention ReLU fires iff a+b+carry >= 10 on "
          f"{float((fire_o == want_o).float().mean()):.4%} of places")


def attention(mod, model, a, b):
    da = torch.tensor([[int(c) for c in f"{a:08d}"[::-1]]])
    db = torch.tensor([[int(c) for c in f"{b:08d}"[::-1]]])
    with torch.no_grad():
        _, att = model(da, db, return_attn=True)
    att = att[0]
    s = [int(x) + int(y) for x, y in zip(f"{a:08d}"[::-1], f"{b:08d}"[::-1])]
    kind = ["".join("G" if v > 9 else ("T" if v == 9 else "K") for v in s)]
    print(f"\n{a} + {b} = {a + b}")
    print(f"  per-place a+b (LSB first): {s}")
    print(f"  generate/transparent/kill: {kind[0]}  (pos 1..8)")
    print("  each query's most-attended key, and how much mass it gets:")
    for q in range(1, 10):
        j = int(att[q].argmax())
        print(f"    pos {q} (digit {q - 1} of the sum) -> pos {j}"
              f"   mass {float(att[q, j]):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="/workspace/submission.py")
    args = ap.parse_args()
    mod, model = load(args.path)
    n = sum(p.numel() for p in model.parameters())
    print(f"{args.path}: {n} parameters\n")
    digit_geometry(model)
    relu_roles(model)
    for a, b in ((19999999, 10000001), (12345678, 87654321),
                 (55555555, 44444445)):
        attention(mod, model, a, b)


if __name__ == "__main__":
    main()
