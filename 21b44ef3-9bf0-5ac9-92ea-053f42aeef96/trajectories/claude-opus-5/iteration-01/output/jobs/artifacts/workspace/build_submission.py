"""Emit /workspace/submission.py from a trained checkpoint.

The graded file is assembled from two pieces:
  * the verbatim text of model_src.py (architecture + tokenisation), so the
    shipped forward pass is byte-for-byte the one that was trained, and
  * the trained parameters, base64-encoded float32 in state_dict order.

No training code, data generation or file I/O ends up in the output.
"""

import argparse
import base64
import io
import json

import torch

import model_src
from model_src import AdderTransformer

TEMPLATE = '''"""Minimal transformer that adds two 8-digit integers.

{summary}

Interface:
    model, meta = build_model()
    add(model, 12345678, 87654321) -> 99999999

Every returned sum is read straight off one forward pass of `model`; the
parameters below are the ones produced by training (see train.py, which is not
imported here).
"""

import base64

{model_source}

# ---------------------------------------------------------------------------
# Trained parameters: float32, little-endian, concatenated in state_dict order.
# ---------------------------------------------------------------------------

_CONFIG = {config}

_WEIGHTS = (
{weights}
)

_META = {meta}


def build_model():
    """Rebuild the trained model.  Returns (model, metadata)."""
    model = AdderTransformer(**_CONFIG)
    flat = torch.frombuffer(bytearray(base64.b64decode(_WEIGHTS)), dtype=torch.float32)
    state, off = {{}}, 0
    for name, ref in model.state_dict().items():
        n = ref.numel()
        state[name] = flat[off:off + n].view_as(ref).clone()
        off += n
    if off != flat.numel():
        raise RuntimeError("weight blob does not match architecture")
    model.load_state_dict(state)
    model.eval()
    meta = dict(_META)
    meta["n_parameters"] = sum(p.numel() for p in model.parameters())
    return model, meta


@torch.no_grad()
def add(model, a, b):
    """Return a + b for 8-digit operands, decoded from one forward pass."""
    device = next(model.parameters()).device
    ta, tb = encode(int(a), int(b), device=device)
    logits = model(ta, tb)
    digits = logits[0, 1:, :].argmax(-1).tolist()
    return decode(digits)

'''


def wrap_b64(blob, width=88):
    lines = [blob[i:i + width] for i in range(0, len(blob), width)]
    return "\n".join('    "%s"' % ln for ln in lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", default="/workspace/submission.py")
    ap.add_argument("--holdout-acc", type=float, default=None)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = dict(d_model=ck["d_model"], layers=ck["layers"],
                  pos_mode=ck.get("pos_mode", "rank1"),
                  tie_head=ck.get("tie_head", True),
                  head_bias=ck.get("head_bias", True),
                  q_bias=ck.get("q_bias", False),
                  learn_scale=ck.get("learn_scale", False))
    model = AdderTransformer(**config)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    buf = io.BytesIO()
    for name, ref in model.state_dict().items():
        buf.write(ref.detach().to(torch.float32).contiguous().numpy().tobytes())
    blob = base64.b64encode(buf.getvalue()).decode("ascii")

    acc = args.holdout_acc if args.holdout_acc is not None else ck.get("holdout_acc", 0.0)
    layer_desc = " -> ".join(
        ("attn(h={n_heads}x{d_head}) + mlp({d_ff})" if l["n_heads"] * l["d_head"] else "mlp({d_ff})").format(**l)
        for l in ck["layers"])
    meta = {
        "task": "8-digit decimal addition",
        "architecture": "causal transformer, d_model=%d, %s" % (ck["d_model"], layer_desc),
        "n_parameters": n_params,
        "positional_encoding": config["pos_mode"],
        "tied_embedding_head": config["tie_head"],
        "sequence_length": model_src.SEQ,
        "decoding": "single forward pass, argmax per output digit",
        "holdout_exact_match": round(float(acc), 6),
    }
    if args.notes:
        meta["notes"] = args.notes

    n_attn = sum(1 for l in ck["layers"] if l["n_heads"] * l["d_head"])
    summary = ("A causal, %d-parameter transformer trained from scratch on randomly\n"
               "generated operand pairs.  Digits are fed least-significant-first, one\n"
               "place per position, and the carry is resolved by %s\n"
               "attending to the nearest lower place that is not carry-transparent\n"
               "(the nearest place whose digits do not sum to exactly 9).\n"
               "Held-out exact-match accuracy: %.3f%%."
               % (n_params, "the attention layer" if n_attn == 1 else "attention",
                  100.0 * acc))

    src = model_src.__file__
    with open(src) as fh:
        model_source = fh.read()
    # Drop the module docstring (re-stated in the submission header) and keep the rest.
    body = model_source.split('"""', 2)[2].lstrip("\n")

    out = TEMPLATE.format(
        summary=summary,
        model_source=body.rstrip(),
        config=json.dumps(config, indent=4).replace("true", "True").replace("false", "False"),
        weights=wrap_b64(blob),
        meta=json.dumps(meta, indent=4).replace("true", "True").replace("false", "False"),
    )
    with open(args.out, "w") as fh:
        fh.write(out)
    print("wrote %s (%d params, %.4f%% holdout)" % (args.out, n_params, 100 * acc))


if __name__ == "__main__":
    main()
