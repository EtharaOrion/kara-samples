"""Generate sweep specs and report results."""
import json
import os
import sys

from sweep import param_count

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")


def blocks_str(bl):
    return ";".join("%d,%d,%d" % b for b in bl)


def report(prefix=""):
    rows = []
    for fn in sorted(os.listdir(RUNS)):
        if not fn.endswith(".json") or not fn.startswith(prefix):
            continue
        try:
            r = json.load(open(os.path.join(RUNS, fn)))
        except Exception:
            continue
        if "acc" not in r:
            continue
        rows.append((r["params"], r["acc"]["uniform"], r["acc"]["chain"],
                     r["acc"]["maxchain"], fn[:-5],
                     r["cfg"]["blocks"], r["cfg"]["d_model"],
                     [round(s["argmax_varies"], 3) for s in r["attn"]]))
    rows.sort(key=lambda x: (x[0], -x[1]))
    print("%-6s %-9s %-9s %-9s %-28s %s" % ("par", "uniform", "chain", "maxchn", "tag", "d/blocks"))
    for p, u, c, m, tag, bl, d, av in rows:
        flag = "OK " if u >= 0.999 else ("ok " if u >= 0.99 else "   ")
        print("%s%-6d %-9.5f %-9.5f %-9.5f %-28s d%d %s %s"
              % (flag, p, u, c, m, tag, d, blocks_str([tuple(b) for b in bl]), av))
    return rows


if __name__ == "__main__":
    if sys.argv[1] == "report":
        report(sys.argv[2] if len(sys.argv) > 2 else "")
