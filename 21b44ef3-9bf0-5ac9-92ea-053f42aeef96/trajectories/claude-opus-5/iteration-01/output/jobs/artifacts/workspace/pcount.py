"""Parameter count for a configuration, without building anything: used to pick
which shrink steps are worth a training run."""
import sys, itertools
import model_src
from model_src import AdderTransformer
from train import parse_layers

def count(d, layers, pos="rank1", tie=True, hb=True, qb=True, ls=True):
    m = AdderTransformer(d, parse_layers(layers), pos_mode=pos, tie_head=tie,
                         head_bias=hb, q_bias=qb, learn_scale=ls)
    return sum(p.numel() for p in m.parameters())

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for spec in sys.argv[1:]:
            d, layers, pos, hb = spec.split("|")
            print("%4d  d=%s %-14s %-5s hb=%s" % (count(int(d), layers, pos, True, hb=="1"), d, layers, pos, hb))
    else:
        for d in (5, 6):
            for dh1, ff1 in ((1,4),(2,4),(2,3)):
                for dh2, ff2 in ((3,14),(3,10),(2,12),(2,10),(2,9),(1,12),(1,10)):
                    L = "1,%d,%d;1,%d,%d" % (dh1, ff1, dh2, ff2)
                    for pos in ("rank1","ramp"):
                        for hb in (True, False):
                            print("%4d  d=%d %-14s %-5s hb=%d" % (count(d,L,pos,hb=hb), d, L, pos, hb))
