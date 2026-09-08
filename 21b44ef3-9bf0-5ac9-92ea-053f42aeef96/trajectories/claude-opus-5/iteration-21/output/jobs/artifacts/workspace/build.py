"""Write /workspace/submission.py from a trained checkpoint.

The shipped file is model_src.py verbatim with the weight block replaced by
plain Python float literals, so the file that is graded is exactly the source
that was tested here.  Only `torch` is imported.
"""

import argparse
import re

import torch

import lab

HEAD = "# --- weights ---"
TAIL = "# --- /weights ---"


def fmt(v):
    return repr(float(v))


def emit(vals, out_path="/workspace/submission.py", src_path="/workspace/model_src.py"):
    block = [
        HEAD,
        "_WEIGHTS = {",
        '    "code_free": [' + ", ".join(fmt(v) for v in vals["code_free"]) + "],",
        '    "carry_w": ' + fmt(vals["carry_w"]) + ",",
        '    "knee": [' + ", ".join(fmt(v) for v in vals["knee"]) + "],",
        '    "fold": ' + fmt(vals["fold"]) + ",",
        "}",
        TAIL,
    ]
    src = open(src_path).read()
    new = re.sub(re.escape(HEAD) + r".*?" + re.escape(TAIL), "\n".join(block), src,
                 flags=re.S)
    assert new != src and HEAD in new
    open(out_path, "w").write(new)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/p2.pt")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--out", default="/workspace/submission.py")
    args = ap.parse_args()

    ck = torch.load(args.ckpt)
    p = ck["params"]
    i = args.index
    vals = {
        "code_free": p["code_free"][i].tolist(),
        "carry_w": p["carry_w"][i].item(),
        "knee": p["knee"][i].tolist(),
        "fold": p["fold"][i].item(),
    }
    path = emit(vals, args.out)
    print("wrote", path)
    print("  code ", [round(v, 5) for v in lab.PIN + tuple(vals["code_free"])])
    print("  carry_w", round(vals["carry_w"], 5),
          " knee", [round(v, 5) for v in vals["knee"]],
          " fold", round(vals["fold"], 5))


if __name__ == "__main__":
    main()
