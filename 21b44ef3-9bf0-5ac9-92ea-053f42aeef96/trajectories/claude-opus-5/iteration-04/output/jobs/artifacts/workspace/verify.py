"""Audit /workspace/submission.py exactly the way the grader would use it.

Checks:
  1. import surface of the graded file
  2. build_model() / add() on large held-out uniform samples
  3. structured edge cases (long carry propagation runs, extremes)
  4. `add` agrees with the batched path, and every answer comes from the model
  5. attention really does work: the pattern is input dependent, freezing it
     destroys accuracy, and corrupting the model output changes the answers
"""
import argparse, ast, importlib.util, random, sys
import torch
import torch.nn.functional as F

import data
import probe

DEV = "cpu"


def load(path):
    spec = importlib.util.spec_from_file_location("subm", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def import_surface(path):
    tree = ast.parse(open(path).read())
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            mods.add(n.module or "")
    return sorted(mods)


# ---------------------------------------------------------------- instrumented
def manual_forward(m, x, attn_mode="normal", frozen=None):
    """The block's forward pass re-run outside the shipped module, so the
    attention can be intervened on.  Any drift between the two would show up as
    a mismatch in the check below."""
    lg, raw, _ = probe.run(m, x, attn_mode, frozen)
    return lg, raw


def acc_of(sub, model, dig, chunk=20000):
    ok = 0
    for i in range(0, dig.shape[0], chunk):
        d = dig[i:i + chunk]
        y = data.targets(d)
        pw = torch.tensor([10 ** j for j in range(9)])
        true = (y * pw).sum(-1).tolist()
        pairs = _pairs(d)
        got = sub.add_batch(model, pairs)
        ok += sum(int(g == t) for g, t in zip(got, true))
    return ok / dig.shape[0]


def _pairs(d):
    pw = torch.tensor([10 ** j for j in range(8)])
    a = (d[..., 0] * pw).sum(-1).tolist()
    b = (d[..., 1] * pw).sum(-1).tolist()
    return list(zip(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="/workspace/submission.py")
    ap.add_argument("--n", type=int, default=300000)
    a = ap.parse_args()

    print("imports:", import_surface(a.path))
    sub = load(a.path)
    model, meta = sub.build_model()
    n_par = sum(p.numel() for p in model.parameters())
    n_buf = sum(b.numel() for b in model.buffers())
    print(f"params={n_par}  buffers={n_buf}  meta.n_params={meta.get('n_params')}")
    print("buffers:", {k: list(v.shape) for k, v in model.named_buffers()})

    # ---- accuracy -----------------------------------------------------------
    u = data.eval_set(a.n, DEV, 20240001, data.UNIFORM)
    h = data.eval_set(a.n // 2, DEV, 20240002, data.HARD)
    au, ah = acc_of(sub, model, u), acc_of(sub, model, h)
    print(f"held-out uniform  n={u.shape[0]}  exact-match={au:.6f}")
    print(f"held-out hard     n={h.shape[0]}  exact-match={ah:.6f}")

    # ---- explicit edge cases ------------------------------------------------
    edges = [(99999999, 99999999), (10000000, 10000000), (19999999, 10000001),
             (99999999, 10000001), (55555555, 44444445), (12345678, 87654322),
             (98765432, 12345678), (11111111, 88888889), (50000000, 50000000),
             (99999999, 10000000), (10000001, 19999999), (45454545, 54545455)]
    bad = [(x, y, sub.add(model, x, y)) for x, y in edges if sub.add(model, x, y) != x + y]
    print(f"hand edge cases: {len(edges)-len(bad)}/{len(edges)} correct", bad[:5])

    # exhaustive over carry structure: every one of the 3^8 generate/propagate/
    # absorb patterns, one instance each
    pats = []
    rng = random.Random(0)
    for code in range(3 ** 8):
        da, db, c = [], [], code
        for i in range(8):
            k = c % 3; c //= 3
            msb = (i == 7)
            lo = 1 if msb else 0
            if k == 0:      # propagate
                x = rng.randint(lo, 9 - lo); y = 9 - x
            elif k == 1:    # generate
                s = rng.randint(max(10, 2 * lo), 18)
                x = rng.randint(max(lo, s - 9), min(9, s - lo)); y = s - x
            else:           # absorb
                s = rng.randint(2 * lo, 8)
                x = rng.randint(lo, min(9, s - lo)); y = s - x
            da.append(x); db.append(y)
        pats.append([da, db])
    pt = torch.tensor(pats).permute(0, 2, 1)
    ap_ = acc_of(sub, model, pt)
    print(f"all {pt.shape[0]} carry-structure patterns: exact-match={ap_:.6f}")

    # ---- add() consistency --------------------------------------------------
    pr = _pairs(u[:2000])
    one = [sub.add(model, x, y) for x, y in pr]
    many = sub.add_batch(model, pr)
    print("add == add_batch:", one == many)

    # ---- attention audit ----------------------------------------------------
    d = data.eval_set(4096, DEV, 777, data.HARD)
    x = data.tokens(d)
    lg, att = manual_forward(model, x)
    with torch.no_grad():
        ref = model(x)
    print(f"instrumented forward matches model: max|d|={float((lg-ref).abs().max()):.2e}")
    am = att.argmax(-1)
    varying = [float((am[:, i] != am[0, i]).float().mean()) for i in range(1, 10)]
    print("attention argmax varies with input, per query position:",
          [f"{v:.2f}" for v in varying])
    ent = -(att.clamp_min(1e-9).log() * att).sum(-1)
    print(f"mean attention entropy (pos>=1): {float(ent[:, 1:].mean()):.3f} nats; "
          f"attn std across inputs: {float(att[:, 1:].std(0).max()):.3f}")

    y = data.targets(d)
    def em(logits):
        return float((logits[:, 1:, :].argmax(-1) == y).all(-1).float().mean())
    frozen = att.mean(0, keepdim=True)
    lg_f, _ = manual_forward(model, x, "frozen", frozen)
    lg_u, _ = manual_forward(model, x, "uniform")
    print(f"exact-match  normal={em(lg):.4f}  attn frozen at batch mean={em(lg_f):.4f}  "
          f"attn uniform-causal={em(lg_u):.4f}")

    # corrupting the model output must change the answers
    class Corrupt(torch.nn.Module):
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, xx): return s.m(xx) + torch.randn_like(s.m(xx)) * 50.0
    torch.manual_seed(0)
    cm = Corrupt(model)
    pr = _pairs(u[:500])
    diff = sum(sub.add(cm, p, q) != sub.add(model, p, q) for p, q in pr)
    print(f"answers changed when model output is corrupted: {diff}/500")

    ok = au >= 0.99 and ah >= 0.99 and ap_ >= 0.99 and not bad
    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'}  params={n_par} "
          f"uniform={au:.6f} hard={ah:.6f} patterns={ap_:.6f}")


if __name__ == "__main__":
    main()
