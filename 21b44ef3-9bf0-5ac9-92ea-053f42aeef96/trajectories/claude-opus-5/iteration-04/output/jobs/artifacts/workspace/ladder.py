"""The config ladder for the parameter cascade.

Family: a 1-D digit code that is at once the input embedding and the set of
output prototypes (`out_mode="proto"`, `code_dim=1`), so the ten learned code
numbers are the *whole* vocabulary cost.  Everything else is one Macaron block.

The lean layout confines every read and write to a named residual axis:

    axis 0   the running sum / the answer -- what the readout looks at
    axis 1   attention key   (something that marks "this place is not a 9")
    axis 2   attention value (something that marks "this place carries out")

which is what makes the small rungs small: the attention reads one scalar for
its score and one scalar for its value, and the readout reads axis 0 directly.
"""
from model_src import Adder, BASE, count_params


def CFG(d, f1, f2, **kw):
    """Unconfined rung: every block reads and writes the whole residual."""
    c = dict(BASE)
    c.update(d=d, T=10, code_dim=1, code_fix=0, out_mode="proto", out_dim=1, norm="",
             f1=f1, f1_in=(0, d), f1_out=(0, d), f1_ob=True, f1_fixed_in=False,
             qk_in=(0, d), sep_qk=True, q_bias=True, k_bias=False,
             alibi=True, self_bias=True, v_in=(0, d), o_out=(0, d), o_fixed=False,
             f2=f2, f2_in=(0, d), f2_out=(0, d), f2_ob=True,
             r_in=(0, 1), read_id=True, r_bias=False,
             logit_scale=True, lsc0=1.0)
    c.update(kw)
    return c


def LEAN(f1, f2, d=3, **kw):
    """Confined rung: one axis per role, identity readout, no spare scales."""
    c = CFG(d, f1, f2,
            f1_in=(0, 1), f1_out=(1, 3), f1_ob=False,
            qk_in=(1, 2), sep_qk=False, q_bias=True, q_fixed=True,
            v_in=(2, 3), v_fixed=True, o_out=(0, 1), o_fixed=True,
            f2_in=(0, 1), f2_out=(0, 1), f2_ob=False,
            r_in=(0, 1), read_id=True, r_bias=False, logit_scale=False)
    c.update(kw)
    return c


def LEAN1(f1, f2, **kw):
    """Two residual axes: the attention key and its value share axis 1.

    A single scalar feature phi(t) can serve both roles.  With phi(t) = 0 for
    t <= 8, phi(9) = -delta and phi(t) = e for t >= 10, the score prefers any
    non-nine over a nine by delta and prefers a generator over an absorber by
    only e, so a large delta/e ratio lets the relative-position bias sit in
    between and pick the *most recent* non-nine -- whose phi is then exactly
    e times the carry it produces.

    That ratio is why this layout keeps the attention output scale: e has to be
    small against the position bias, so that the value's stray contribution to
    the *score* cannot outrank recency, while the carry it writes has to be a
    whole step of the code.  One scale factor between them buys both, and the
    code's unit gauge (`code_fix`) pays it back.
    """
    c = CFG(2, f1, f2,
            f1_in=(0, 1), f1_out=(1, 2), f1_ob=False,
            qk_in=(1, 2), sep_qk=False, q_bias=True, q_fixed=True,
            v_in=(1, 2), v_fixed=True, o_out=(0, 1), o_fixed=False,
            f2_in=(0, 1), f2_out=(0, 1), f2_ob=False,
            r_in=(0, 1), read_id=True, r_bias=False, logit_scale=False)
    c.update(kw)
    return c


def n(cfg):
    return count_params(Adder(cfg))


if __name__ == "__main__":
    for name, c in [
        ("wide d16 f32", CFG(16, 32, 32)),
        ("wide d8  f16", CFG(8, 16, 16)),
        ("wide d6  f10", CFG(6, 10, 8)),
        ("wide d5  f6",  CFG(5, 6, 5)),
        ("wide d4  f5",  CFG(4, 5, 4)),
        ("lean d4  f5",  LEAN(5, 4, d=4, f1_out=(1, 4), v_in=(3, 4))),
        ("lean d3  f4",  LEAN(4, 3)),
        ("lean d3  f3",  LEAN(3, 3)),
        ("lean d3  f32", LEAN(3, 2)),
        ("lean d3  f22", LEAN(2, 2)),
    ]:
        print(f"{name:14s} {n(c):5d}")
