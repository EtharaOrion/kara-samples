"""Independent audit of /workspace/submission.py.

Loads the graded file from disk by path (nothing else from this workspace is imported
into the accuracy path), and checks:

  1. what the file imports and how big it is        -- the screening surface
  2. the parameter count, straight off the module   -- not off its metadata
  3. every float the module holds is either a registered parameter or a declared constant
  4. `add()` on held-out 8-digit pairs, against Python's own `a + b`
  5. the same at scale, batched, plus edge cases and every carry-class pattern
  6. an ablation showing the attention map really depends on the input

    python verify.py --pairs 200000
"""
import argparse
import ast
import importlib.util
import os
import random
import sys

import torch

WORK = os.path.dirname(os.path.abspath(__file__))
SUB = os.path.join(WORK, "submission.py")

ALLOWED_IMPORTS = {"torch", "torch.nn"}
BANNED_NAMES = {"base64", "pickle", "marshal", "exec", "eval", "compile", "__import__",
                "open", "urllib", "requests", "subprocess", "os", "sys", "zlib", "bz2",
                "lzma", "codecs", "importlib"}


def load_submission(path=SUB):
    spec = importlib.util.spec_from_file_location("graded_submission", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["graded_submission"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- 1. file screen
def screen(path=SUB):
    src = open(path).read()
    tree = ast.parse(src)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
    # `nn.Module.eval()` is not the builtin, so attributes are screened against the
    # narrower list; a bare `eval`/`exec`/`open` name would be the real problem.
    attr_banned = BANNED_NAMES - {"eval", "compile", "open"}
    bad_names = sorted(
        {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id in BANNED_NAMES}
        | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
           and n.attr in attr_banned})
    n_lit = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Constant)
                and isinstance(n.value, float))
    return {
        "bytes": len(src),
        "imports": sorted(imports),
        "imports_ok": imports <= ALLOWED_IMPORTS,
        "banned_names_found": bad_names,
        "float_literals": n_lit,
        "defines": sorted(n.name for n in tree.body
                          if isinstance(n, (ast.FunctionDef, ast.ClassDef))),
    }


# ---------------------------------------------------------------- 2/3. parameters
def parameter_report(model):
    named = [(n, tuple(p.shape), p.numel()) for n, p in model.named_parameters()]
    bufs = [(n, [round(float(x), 8) for x in b.reshape(-1).tolist()])
            for n, b in model.named_buffers()]
    # any float tensor hanging off the module that is neither a parameter nor a buffer
    known = {id(t) for t in list(model.parameters()) + list(model.buffers())}
    stray = [k for k, v in vars(model).items()
             if isinstance(v, torch.Tensor) and id(v) not in known]
    return {
        "n_parameters": sum(numel for _, _, numel in named),
        "parameters": named,
        "buffers": bufs,
        "stray_tensors": stray,
    }


# ---------------------------------------------------------------- 4/5. accuracy
def holdout_bucket_py(a, b, n=8):
    """Same split rule as data.py, in plain Python, so the auditor does not depend on it."""
    m30 = (1 << 30) - 1
    h = 0
    for k in range(n):
        h = (h * 1000003 + ((a // 10 ** k) % 10) * 10 + ((b // 10 ** k) % 10)) & m30
    h = ((h ^ (h >> 13)) * 1000003) & m30
    h ^= (h >> 7)
    return h % 16


def random_pair(rng):
    return rng.randint(10_000_000, 99_999_999), rng.randint(10_000_000, 99_999_999)


def check_add(mod, model, pairs):
    wrong = []
    for a, b in pairs:
        got = mod.add(model, a, b)
        if got != a + b:
            wrong.append((a, b, got, a + b))
            if len(wrong) > 8:
                break
    return wrong


def batched_predict(mod, model, A, B, n=8):
    """Same tokenisation and same argmax decode as `add`, run on many pairs at once."""
    dev = next(model.parameters()).device
    tok = torch.zeros(len(A), n + 2, 2, dtype=torch.long, device=dev)
    for k in range(n):
        tok[:, k + 1, 0] = torch.tensor([(a // 10 ** k) % 10 for a in A], device=dev)
        tok[:, k + 1, 1] = torch.tensor([(b // 10 ** k) % 10 for b in B], device=dev)
    with torch.no_grad():
        d = model(tok).argmax(-1)[:, 1:]
    pw = torch.tensor([10 ** k for k in range(n + 1)], dtype=torch.long, device=dev)
    return (d * pw).sum(-1)


def big_check(mod, model, rng, total, chunk=8192, holdout_only=True):
    seen = wrong = 0
    examples = []
    while seen < total:
        A, B = [], []
        while len(A) < chunk:
            a, b = random_pair(rng)
            if holdout_only and holdout_bucket_py(a, b) != 0:
                continue
            A.append(a)
            B.append(b)
        got = batched_predict(mod, model, A, B).tolist()
        for a, b, gv in zip(A, B, got):
            if gv != a + b:
                wrong += 1
                if len(examples) < 8:
                    examples.append((a, b, gv, a + b))
        seen += len(A)
    return seen, wrong, examples


def carry_pattern_check(mod, model, rng):
    """All 3^8 assignments of {absorb, transparent, generate} to the eight places -- the
    complete space of carry behaviours an 8-digit addition can have."""
    pats, A, B = [], [], []
    for code in range(3 ** 8):
        c, da, db = code, [], []
        for k in range(8):
            cls = c % 3
            c //= 3
            lo, hi = [(0, 8), (9, 9), (10, 18)][cls]
            s = rng.randint(lo, hi)
            x = rng.randint(max(0, s - 9), min(9, s))
            da.append(x)
            db.append(s - x)
        if da[7] == 0:
            da[7] = max(1, da[7])
        A.append(sum(d * 10 ** k for k, d in enumerate(da)))
        B.append(sum(d * 10 ** k for k, d in enumerate(db)))
        pats.append(code)
    bad = []
    for i in range(0, len(A), 4096):
        got = batched_predict(mod, model, A[i:i + 4096], B[i:i + 4096]).tolist()
        for a, b, gv in zip(A[i:i + 4096], B[i:i + 4096], got):
            if gv != a + b:
                bad.append((a, b, gv, a + b))
    return len(A), bad


def edge_cases(mod, model):
    cases = [(10_000_000, 10_000_000), (99_999_999, 99_999_999), (99_999_999, 10_000_001),
             (10_000_000, 99_999_999), (19_999_999, 10_000_001), (55_555_555, 44_444_445),
             (12_345_678, 87_654_321), (11_111_111, 88_888_889), (98_765_432, 12_345_678),
             (50_000_000, 50_000_000), (99_999_998, 10_000_002), (45_454_545, 54_545_455)]
    return check_add(mod, model, cases), len(cases)


# ---------------------------------------------------------------- 6. attention ablation
def ablate(mod, model, A, B, n=8, mode="freeze_key"):
    """Re-run the block with the content-dependent part of the attention logits removed,
    leaving only the fixed relative-position term.  If accuracy survives that, the
    attention was a fixed pattern; if it collapses, the map really is input-driven."""
    dev = next(model.parameters()).device
    tok = torch.zeros(len(A), n + 2, 2, dtype=torch.long, device=dev)
    for k in range(n):
        tok[:, k + 1, 0] = torch.tensor([(a // 10 ** k) % 10 for a in A], device=dev)
        tok[:, k + 1, 1] = torch.tensor([(b // 10 ** k) % 10 for b in B], device=dev)

    with torch.no_grad():
        code = model.codes()
        x = code[tok[..., 0]] + code[tok[..., 1]]
        u = torch.clamp(model.bank_w * (x.unsqueeze(-1) - model.knee), 0.0, 1.0)
        key = model.key_w * (u[..., 1] - u[..., 0])
        val = u[..., 1]
        p = x.shape[-1]
        idx = torch.arange(p, device=dev)
        rel = model.lam * (idx[:, None] - idx[None, :]).to(x.dtype)
        if mode == "freeze_key":
            logit = key.mean() * torch.ones_like(key).unsqueeze(-2) + rel
        elif mode == "mean_key":
            logit = key.mean(0, keepdim=True).unsqueeze(-2).expand(len(A), -1, -1) + rel
        else:
            raise ValueError(mode)
        after = idx[:, None] > idx[None, :]
        aoa = idx[:, None] >= idx[None, :]
        strict = after | ((idx[:, None] == 0) & (idx[None, :] == 0))
        neg = torch.tensor(-1e30, device=dev)
        cin = (torch.softmax(torch.where(strict, logit, neg), -1) * val.unsqueeze(-2)).sum(-1)
        cout = (torch.softmax(torch.where(aoa, logit, neg), -1) * val.unsqueeze(-2)).sum(-1)
        y = x + model.carry_w * cin + model.fold * cout
        d = (-(y.unsqueeze(-1) - code) ** 2).argmax(-1)[:, 1:]
    pw = torch.tensor([10 ** k for k in range(n + 1)], device=dev)
    got = (d * pw).sum(-1).tolist()
    ok = sum(1 for a, b, gv in zip(A, B, got) if gv == a + b)
    return ok / len(A)


def carry_heavy_pairs(rng, m):
    """Pairs whose carries actually have to travel: many carry-transparent places."""
    A, B = [], []
    for _ in range(m):
        da, db = [], []
        for k in range(8):
            if rng.random() < 0.55:
                x = rng.randint(0, 9)
                da.append(x)
                db.append(9 - x)
            else:
                s = rng.randint(0, 18)
                x = rng.randint(max(0, s - 9), min(9, s))
                da.append(x)
                db.append(s - x)
        da[7] = max(da[7], 1)
        db[7] = max(db[7], 1)
        A.append(sum(d * 10 ** k for k, d in enumerate(da)))
        B.append(sum(d * 10 ** k for k, d in enumerate(db)))
    return A, B


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=200_000)
    ap.add_argument("--add_calls", type=int, default=5000)
    ap.add_argument("--path", default=SUB)
    args = ap.parse_args()
    rng = random.Random(20260907)

    print("=" * 72)
    print("file screen")
    print("=" * 72)
    s = screen(args.path)
    for k, v in s.items():
        print(f"  {k}: {v}")

    mod = load_submission(args.path)
    model, meta = mod.build_model()
    print("=" * 72)
    print("parameters (read off the module, not the metadata)")
    print("=" * 72)
    rep = parameter_report(model)
    for k, v in rep.items():
        print(f"  {k}: {v}")
    print(f"  metadata claims: {meta.get('parameters')}")

    print("=" * 72)
    print("accuracy")
    print("=" * 72)
    pairs = []
    while len(pairs) < args.add_calls:
        a, b = random_pair(rng)
        if holdout_bucket_py(a, b) == 0:
            pairs.append((a, b))
    wrong = check_add(mod, model, pairs)
    print(f"  add() on {len(pairs)} held-out pairs: {len(pairs) - len(wrong)} correct, "
          f"{len(wrong)} wrong {wrong[:3]}")

    # the batched path must agree with add() itself
    A = [a for a, _ in pairs[:2048]]
    B = [b for _, b in pairs[:2048]]
    bp = batched_predict(mod, model, A, B).tolist()
    dis = sum(1 for i, (a, b) in enumerate(pairs[:2048]) if bp[i] != mod.add(model, a, b))
    print(f"  batched decode disagrees with add() on {dis}/2048 pairs")

    seen, w, ex = big_check(mod, model, rng, args.pairs, holdout_only=True)
    print(f"  held-out only: {seen - w}/{seen} exact  ({(seen - w) / seen:.6f}) {ex[:3]}")
    seen2, w2, ex2 = big_check(mod, model, rng, args.pairs, holdout_only=False)
    print(f"  all pairs    : {seen2 - w2}/{seen2} exact  ({(seen2 - w2) / seen2:.6f}) {ex2[:3]}")

    ew, en = edge_cases(mod, model)
    print(f"  edge cases   : {en - len(ew)}/{en} exact {ew[:3]}")

    npat, bad = carry_pattern_check(mod, model, rng)
    print(f"  carry-class patterns (3^8): {npat - len(bad)}/{npat} exact {bad[:3]}")

    print("=" * 72)
    print("is the attention doing real work?")
    print("=" * 72)
    for name, (A2, B2) in [("carry-heavy", carry_heavy_pairs(rng, 4096)),
                           ("uniform", ([random_pair(rng)[0] for _ in range(4096)],
                                        [random_pair(rng)[1] for _ in range(4096)]))]:
        full = sum(1 for a, b, gv in zip(A2, B2, batched_predict(mod, model, A2, B2).tolist())
                   if gv == a + b) / len(A2)
        fz = ablate(mod, model, A2, B2, mode="freeze_key")
        mk = ablate(mod, model, A2, B2, mode="mean_key")
        print(f"  {name:12s}: intact {full:.4f} | constant key {fz:.4f} | "
              f"batch-mean key {mk:.4f}")

    ok = (len(wrong) == 0 and w == 0 and len(bad) == 0 and len(ew) == 0
          and s["imports_ok"] and not s["banned_names_found"])
    print("=" * 72)
    print("VERDICT:", "pass" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
