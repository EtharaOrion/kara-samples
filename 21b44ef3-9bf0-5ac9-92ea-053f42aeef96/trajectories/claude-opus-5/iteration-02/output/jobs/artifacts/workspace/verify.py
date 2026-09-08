"""Independent checks on /workspace/submission.py.

  * dependency surface (what the file imports)
  * parameter count, counted from the module itself
  * exact-match accuracy through the graded add() entry point, on pairs from
    the held-out 1% that training never saw
  * the answer must actually come from the forward pass (weight corruption
    must change it)
  * self-attention must do real work (freezing the pattern must break it)
"""
import argparse
import ast
import importlib.util
import os
import sys
import time

import torch

import data

HERE = os.path.dirname(os.path.abspath(__file__))
ALLOWED_IMPORTS = {"torch", "math"}


def load(path):
    spec = importlib.util.spec_from_file_location("submission_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["submission_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def check_imports(path):
    tree = ast.parse(open(path).read())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                found.add(n.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                found.add(node.module.split(".")[0])
    return found, found - ALLOWED_IMPORTS


def batched_predict(mod, model, tok):
    with torch.no_grad():
        return model(tok)[:, 1:, :].argmax(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=os.path.join(HERE, "submission.py"))
    ap.add_argument("--n_api", type=int, default=5000)
    ap.add_argument("--n_batch", type=int, default=400000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    found, bad = check_imports(args.path)
    print("imports:", sorted(found), "| outside allowlist:", sorted(bad) or "none")

    mod = load(args.path)
    model, meta = mod.build_model()
    npar = sum(p.numel() for p in model.parameters())
    nbuf = sum(b.numel() for b in model.buffers())
    print("parameters: %d   buffers(floats): %d" % (npar, nbuf))
    print("metadata:", meta)
    assert isinstance(model, torch.nn.Module)

    dev = args.device
    model = model.to(dev)

    # ---- accuracy through the graded interface -------------------------------
    tok, tgt = data.heldout_set(args.n_api, dev, 777001, "uniform")
    av = data.digits_to_value(tok[:, 1:9, 0]).tolist()
    bv = data.digits_to_value(tok[:, 1:9, 1]).tolist()
    t0 = time.time()
    wrong = 0
    for a, b in zip(av, bv):
        if mod.add(model, a, b) != a + b:
            wrong += 1
    print("add() uniform: %d/%d correct = %.4f%%  (%.1fs)"
          % (len(av) - wrong, len(av), 100 * (1 - wrong / len(av)), time.time() - t0))

    # ---- large batched accuracy ---------------------------------------------
    for kind, n in (("uniform", args.n_batch), ("chain", args.n_batch // 4),
                    ("maxchain", 50000)):
        tok, tgt = data.heldout_set(n, dev, 777002, kind)
        ok = 0
        for i in range(0, tok.shape[0], 65536):
            p = batched_predict(mod, model, tok[i:i + 65536])
            ok += (p == tgt[i:i + 65536]).all(-1).sum().item()
        print("batched %-9s %d/%d = %.5f" % (kind, ok, tok.shape[0], ok / tok.shape[0]))

    # ---- edge cases ----------------------------------------------------------
    edges = [(10000000, 10000000), (99999999, 99999999), (19999995, 80000005),
             (99999999, 10000001), (11111111, 88888889), (50000000, 49999999),
             (12345678, 87654321), (99999998, 10000002), (10000001, 89999999),
             (55555555, 44444445), (98765432, 12345678), (19999999, 10000001)]
    bad_edge = [(a, b, mod.add(model, a, b), a + b) for a, b in edges
                if mod.add(model, a, b) != a + b]
    print("edge cases: %d/%d correct" % (len(edges) - len(bad_edge), len(edges)),
          bad_edge if bad_edge else "")

    # ---- does the answer come from the forward pass? -------------------------
    ref = [mod.add(model, a, b) for a, b in edges]
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 2.0)
    pert = [mod.add(model, a, b) for a, b in edges]
    model.load_state_dict(sd)
    changed = sum(1 for x, y in zip(ref, pert) if x != y)
    print("weight corruption changed %d/%d answers" % (changed, len(edges)))

    # ---- is the attention doing real work? -----------------------------------
    tok_c, tgt_c = data.heldout_set(20000, dev, 777003, "chain")
    tok_u, tgt_u = data.heldout_set(20000, dev, 777004, "uniform")
    with torch.no_grad():
        _, attns = model(tok_c, return_attn=True)
    fixed = [a.mean(0, keepdim=True) for a in attns]
    for li, a in enumerate(attns):
        mad = (a - a.mean(0, keepdim=True)).abs().mean().item()
        am = a.argmax(-1)
        varies = (am != am.mode(0).values.unsqueeze(0)).float().mean().item()
        print("attn layer %d: mean|dev from input-average| = %.4f, "
              "argmax differs from modal pattern on %.1f%% of query rows"
              % (li, mad, 100 * varies))

    orig = mod.SelfAttention.forward
    slot = {"maps": None, "i": 0}

    def frozen_forward(self, x, return_attn=False):
        B, T, _ = x.shape
        H, K = self.n_heads, self.d_head
        v = self.w_v(x).view(B, T, H, K).transpose(1, 2)
        att = slot["maps"][slot["i"]].expand(B, -1, -1, -1)
        slot["i"] += 1
        y = (att @ v).transpose(1, 2).reshape(B, T, H * K)
        y = self.w_o(y)
        return (y, att) if return_attn else y

    mod.SelfAttention.forward = frozen_forward
    slot["maps"] = fixed
    for name, tk, tg in (("chain", tok_c, tgt_c), ("uniform", tok_u, tgt_u)):
        slot["i"] = 0
        with torch.no_grad():
            p = model(tk)[:, 1:, :].argmax(-1)
        # forward consumed len(fixed) maps; reset counter each call
        acc = (p == tg).all(-1).float().mean().item()
        print("frozen-attention accuracy on %-8s %.5f" % (name, acc))
    mod.SelfAttention.forward = orig

    print("\nOK")


if __name__ == "__main__":
    main()
