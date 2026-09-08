import scan
from model_src import Adder, BASE, count_params

def C(**kw):
    c = dict(BASE); c.update(kw); return c

def lean(d=3, f1=2, f2=2, cf=0, norm="", **kw):
    c = C(d=d, code_dim=2, code_fix=cf, out_mode="tied", norm=norm,
          f1=f1, f1_in=(0, 2), f1_out=(d - 1, d), f1_ob=True,
          qk_in=(0, d), q_bias=True, alibi=True, self_bias=True,
          v_in=(0, d), o_out=(d - 1, d), o_fixed=False,
          f2=f2, f2_in=(0, d), f2_out=(0, 2), f2_ob=True,
          r_in=(0, 2), read_id=True, r_bias=False, logit_scale=True)
    c.update(kw); return c

def wide(d, f1, f2, **kw):
    c = C(d=d, code_dim=2, out_mode="tied",
          f1=f1, f1_in=(0, d), f1_out=(0, d), f1_ob=True,
          qk_in=(0, d), q_bias=True, v_in=(0, d), o_out=(0, d),
          f2=f2, f2_in=(0, d), f2_out=(0, d), f2_ob=True,
          r_in=(0, d), read_id=False, r_bias=True, logit_scale=True)
    c.update(kw); return c

JOBS = [
    ("w4_6_norm",  wide(4, 6, 6, norm="a")),
    ("w4_6_nonorm", wide(4, 6, 6, norm="")),
    ("w3_4_nonorm", wide(3, 4, 4, norm="")),
    ("lean3_33",   lean(3, 3, 3)),
    ("lean3_22",   lean(3, 2, 2)),
    ("lean3_22_cf6", lean(3, 2, 2, cf=6)),
    ("lean2_22",   lean(2, 2, 2, f1_out=(0, 2), o_out=(0, 2), f2_in=(0, 2))),
    ("lean3_22_norm", lean(3, 2, 2, norm="a")),
]

if __name__ == "__main__":
    for t, c in JOBS:
        print(t, count_params(Adder(c)))
    for tag, line in scan.run(JOBS, E=256, steps=25000, par=2):
        print(line)
