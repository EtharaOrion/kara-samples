"""Emit /workspace/submission.py.

The graded file is the class source from ship_model.py, verbatim, plus the
trained weights written as plain Python float literals.  Nothing is imported
beyond torch, and no training or data-generation code goes into it.
"""
import argparse
import importlib.util
import sys

import torch

W_MARK = "# ---- 8< ---- WEIGHTS ---- 8< ----"
M_MARK = "# ---- 8< ---- MODEL ---- 8< ----"

DOC = '''"""8-digit addition with a 12-parameter transformer.

Weights were produced by the training pipeline in this workspace
(two_phase.py -> reduce.py -> finetune.py) and are inlined here as plain
literals.  This file contains the model and its inference path only.
"""
'''


def fmt(name, vals):
    body = ", ".join(repr(float(v)) for v in vals)
    return f'    "{name}": [{body}],\n'


def emit(params, buffers, path="/workspace/submission.py"):
    src = open("ship_model.py").read()
    model_src = src.split(M_MARK, 1)[1].lstrip("\n")
    w = "_PARAMS = {\n" + "".join(fmt(k, v) for k, v in params.items()) + "}\n"
    w += "_BUFFERS = {\n" + "".join(fmt(k, v) for k, v in buffers.items()) + "}\n"
    open(path, "w").write(DOC + "\n" + w + "\n" + model_src)
    return path


def load_fresh(path="/workspace/submission.py", name="submission_check"):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="ship.pt")
    ap.add_argument("--out", default="/workspace/submission.py")
    a = ap.parse_args()
    d = torch.load(a.inp, map_location="cpu", weights_only=False)
    path = emit(d["params"], d["buffers"], a.out)

    mod = load_fresh(path)
    model, meta = mod.build_model()
    n = sum(p.numel() for p in model.parameters())
    print(f"wrote {path}")
    print(f"registered parameters: {n}  ({[ (k, tuple(v.shape)) for k, v in model.named_parameters() ]})")
    print(f"buffers: {[(k, tuple(v.shape)) for k, v in model.named_buffers()]}")
    print("metadata:", meta)

    import random
    random.seed(0)
    bad = 0
    for _ in range(2000):
        x = random.randint(10 ** 7, 10 ** 8 - 1)
        y = random.randint(10 ** 7, 10 ** 8 - 1)
        if mod.add(model, x, y) != x + y:
            bad += 1
    print(f"quick check on 2000 random 8-digit pairs: {bad} wrong")


if __name__ == "__main__":
    main()
