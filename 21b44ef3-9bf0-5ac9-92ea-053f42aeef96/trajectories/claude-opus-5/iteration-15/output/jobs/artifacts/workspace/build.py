"""Emit /workspace/submission.py from a trained shipped-form checkpoint.

The graded file gets the model class verbatim out of `arch.py` plus the twelve
trained numbers written as plain Python float literals, and imports nothing but
`torch`.  No training code, no data generation, no encoders of any kind.
"""
import argparse, json, os
import torch

import data, ens, reduce as red

HEAD = '''"""An 8-digit adder: one transformer block over digit-pair tokens.

Twelve trained parameters -- the codes of the digits 1..9, the two thresholds
of the feed-forward bank, and the write-back weight of the base fold.  They
were fitted by gradient descent on sampled addition problems; the training code
lives beside this file and is not needed to run it.
"""
'''


def source_block(path="arch.py"):
    txt = open(path).read()
    a = txt.index("# --- BEGIN SHIPPED ---") + len("# --- BEGIN SHIPPED ---")
    b = txt.index("# --- END SHIPPED ---")
    return txt[a:b].strip("\n")


def fmt(vals):
    return "[\n    " + ",\n    ".join(repr(float(v)) for v in vals) + ",\n]"


def emit(w, out, meta_extra=None):
    body = [HEAD, source_block(), "", "",
            "# The twelve trained values.",
            f"_CODE = {fmt(w['code'])}", "",
            f"_KNEE = {fmt(w['knee'])}", "",
            f"_FOLD = {fmt(w['fold'])}", "", ""]
    meta = dict(architecture="one transformer block: point-wise bank -> "
                             "single-head self-attention read through a "
                             "strictly-causal and an inclusively-causal mask "
                             "-> tied nearest-code read-out",
                residual_channels=1, attention_heads=1, bank_units=2,
                vocabulary=10, trained_parameters=12,
                digits_per_forward_pass="all of them")
    if meta_extra:
        meta.update(meta_extra)
    body.append(f'''def build_model():
    """Return the trained model and a short description of it."""
    model = DigitPairAdder()
    with torch.no_grad():
        model.code.copy_(torch.tensor(_CODE, dtype=torch.float32))
        model.knee.copy_(torch.tensor(_KNEE, dtype=torch.float32))
        model.fold.copy_(torch.tensor(_FOLD, dtype=torch.float32))
    model.eval()
    meta = {json.dumps(meta, indent=8)[:-1]}        }}
    meta["parameter_count"] = sum(p.numel() for p in model.parameters())
    return model, meta
''')
    txt = "\n".join(body)
    with open(out, "w") as f:
        f.write(txt)
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--member", type=int, default=-1)
    ap.add_argument("--out", default="/workspace/submission.py")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    pr = {k: v.to(dev) for k, v in ck["params"].items()}
    m = args.member if args.member >= 0 else int(ck["order"][0])
    p = ens.single(pr, m)
    w = red.to_shipped(p)
    print(f"member {m}: acc {float(ck['acc'][m]):.6f} "
          f"margin {float(ck['margin'][m]):.4f}")
    print("  code", [round(v, 5) for v in w["code"]])
    print("  knee", [round(v, 5) for v in w["knee"]],
          " -> thresholds", [round(-v / red.SLOPE, 5) for v in w["knee"]])
    print("  fold", [round(v, 5) for v in w["fold"]])
    emit(w, args.out, dict(source_checkpoint=os.path.basename(args.ckpt),
                           source_member=m))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
