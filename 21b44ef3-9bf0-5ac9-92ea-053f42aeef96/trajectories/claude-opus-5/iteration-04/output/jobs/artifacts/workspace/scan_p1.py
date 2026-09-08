"""Can a 1-D digit code with a nearest-prototype readout be trained at all?

The tied dot-product readout needs a curved (>=2-D) code, which costs 20
parameters.  A squared-distance readout does not: with cd=1 the same 10 numbers
serve as the input embedding *and* as the output prototypes, so the whole
code costs 10.  Capacity is clearly sufficient (a monotone code turns the task
into `s = a+b`, a threshold for the carry and one subtract-10 kink), the open
question is purely whether SGD finds it.  These runs measure that.
"""
import sys
import torch
import scan
from model_src import Adder, BASE, count_params


def P(d, f1, f2, cd=1, mode="proto", **kw):
    c = dict(BASE)
    c.update(d=d, T=10, code_dim=cd, code_fix=0, out_mode=mode, out_dim=cd, norm="",
             f1=f1, f1_in=(0, d), f1_out=(0, d), f1_ob=True, f1_fixed_in=False,
             qk_in=(0, d), sep_qk=True, q_bias=True, k_bias=False,
             alibi=True, self_bias=True, v_in=(0, d), o_out=(0, d), o_fixed=False,
             f2=f2, f2_in=(0, d), f2_out=(0, d), f2_ob=True,
             r_in=(0, d), read_id=False, r_bias=True,
             logit_scale=True, lsc0=1.0)
    c.update(kw)
    return c


RAMP = "/workspace/ck/_ramp.pt"


def make_ramp():
    """Diagnostic only: a monotone code as a warm start, to separate 'the
    architecture cannot express it' from 'the optimiser cannot find it'."""
    torch.save({"cfg": {}, "params": {"code_p": torch.arange(10.) - 4.5},
                "n_params": 0, "acc": 0.0, "u": 0.0, "h": 0.0}, RAMP)


JOBS = [
    ("p1_8_16",   P(8, 16, 16)),
    ("p1_8_16r",  P(8, 16, 16), ["--init_from", RAMP, "--sigma", "0.4"]),
    ("p1_16_32",  P(16, 32, 32)),
    ("p2_8_16",   P(8, 16, 16, cd=2, mode="tied")),
]

if __name__ == "__main__":
    make_ramp()
    for j in JOBS:
        print(j[0], count_params(Adder(j[1])), flush=True)
    for tag, line in scan.run(JOBS, E=256, steps=20000, par=int(sys.argv[1]) if len(sys.argv) > 1 else 2):
        print(line, flush=True)
