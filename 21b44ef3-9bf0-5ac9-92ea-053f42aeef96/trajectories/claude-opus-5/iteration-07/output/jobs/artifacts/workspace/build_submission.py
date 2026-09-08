"""Emit the graded submission.py: model source + trained weights as literals.

The graded file must contain the model and its inference path only, so the
weights are written as plain Python float literals and the only import is
torch.  Training lives in train_ens.py / cascade.py.
"""
import argparse, json, os, re, sys
import torch

import data
from model_src import DigitPairAdder, default_cfg, n_params

HERE = os.path.dirname(os.path.abspath(__file__))
BEGIN = "# ---- BEGIN SUBMISSION SOURCE ----"
END = "# ---- END SUBMISSION SOURCE ----"

HEADER = '''"""Minimal transformer that adds two 8-digit numbers.

One Macaron block (FFN -> single-head strictly-causal self-attention -> FFN)
over ten LSB-first digit-pair tokens; every position predicts its own answer
digit against a tied code table, so the whole sum comes from one forward pass.
All {NP} floating-point parameters below were produced by gradient descent
(see train_ens.py / cascade.py in the training workspace).
"""
'''

TAIL = '''

_CFG = {CFG}

_WEIGHTS = {WEIGHTS}


def build_model():
    """Return (model, metadata) with the trained weights loaded."""
    model = DigitPairAdder(_CFG)
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in _WEIGHTS.items():
            t = torch.tensor(value, dtype=torch.float32)
            params[name].copy_(t.reshape(params[name].shape))
    model.eval()
    metadata = {META}
    return model, metadata
'''


def fmt(x, indent):
    """Nested Python float literals, round-trip exact for float32."""
    if isinstance(x, list):
        inner = ", ".join(fmt(v, indent) for v in x)
        return "[" + inner + "]"
    return repr(float(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", default=os.path.join(HERE, "submission.py"))
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = dict(default_cfg())
    cfg.update(ck["cfg"])                      # fill keys added after the run
    model = DigitPairAdder(cfg)
    sd = {k: v.float() for k, v in ck["state"].items()}
    missing = [k for k, _ in model.named_parameters() if k not in sd]
    if missing:
        raise SystemExit(f"checkpoint missing parameters: {missing}")
    with torch.no_grad():
        for k, p in model.named_parameters():
            p.copy_(sd[k].reshape(p.shape))
    model.eval()
    npar = n_params(model)

    src = open(os.path.join(HERE, "model_src.py")).read()
    body = src[src.index(BEGIN) + len(BEGIN):src.index(END)].strip("\n")

    weights = {k: p.detach().cpu().tolist() for k, p in model.named_parameters()}
    wtxt = "{\n" + "".join(
        f"    {k!r}: {fmt(v, 4)},\n" for k, v in weights.items()) + "}"
    meta = {
        "name": "digit-pair-adder",
        "n_parameters": npar,
        "task": "exact addition of two 8-digit integers",
        "architecture": ("1 Macaron transformer block: FFN -> 1-head "
                         "strictly-causal self-attention with a relative "
                         "distance bias -> FFN"),
        "d_model": cfg["D"],
        "n_layers": 1,
        "n_heads": 1,
        "sequence": "10 LSB-first digit-pair tokens, tied code table readout",
        "trained": True,
    }
    txt = (HEADER.replace("{NP}", str(npar)) + "\n" + body + "\n"
           + TAIL.replace("{CFG}", json.dumps(cfg, sort_keys=True)
                          .replace("true", "True").replace("false", "False"))
           .replace("{WEIGHTS}", wtxt)
           .replace("{META}", json.dumps(meta, sort_keys=True)
                    .replace("true", "True").replace("false", "False")))
    tmp = os.path.join(os.path.dirname(os.path.abspath(args.out)),
                       "_build_check.py")
    open(tmp, "w").write(txt)

    # --- the emitted file must reproduce the checkpoint exactly -------------
    sys.path.insert(0, os.path.dirname(os.path.abspath(tmp)))
    import importlib.util
    spec = importlib.util.spec_from_file_location("_cand", tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m2, md = mod.build_model()
    assert sum(p.numel() for p in m2.parameters()) == npar, "param count drift"
    g = torch.Generator(device="cpu").manual_seed(7)
    tok, tgt = data.sample(20000, "cpu", g, split="all")
    with torch.no_grad():
        d = (model(tok) - m2(tok)).abs().max()
    assert float(d) == 0.0, f"emitted weights differ from checkpoint ({d})"
    acc = float((m2(tok)[:, 1:].argmax(-1) == tgt[:, 1:]).all(-1).float().mean())
    # add() must decode exactly what the batched forward pass predicts
    with torch.no_grad():
        pred = m2(tok[:300])[:, 1:].argmax(-1)
    dec = (pred * torch.tensor([10 ** i for i in range(pred.shape[1])])).sum(-1)
    aa, bb = data.to_int(tok[:300, 1:9, 0]), data.to_int(tok[:300, 1:9, 1])
    bad = [i for i in range(300)
           if mod.add(m2, int(aa[i]), int(bb[i])) != int(dec[i])]
    assert not bad, f"add() disagrees with the batched forward at {bad[:5]}"
    os.replace(tmp, args.out)
    print(f"WROTE {args.out}  params={npar}  spotcheck_acc={acc:.5f}  "
          f"bytes={len(txt)}")


if __name__ == "__main__":
    main()
