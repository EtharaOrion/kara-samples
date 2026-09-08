"""Audit a submission.py: accuracy, carry-structure coverage, and whether the
attention is actually load-bearing.
"""
import argparse
import importlib.util
import ast
import itertools
import sys

import torch

ALLOWED_IMPORTS = {"torch", "torch.nn", "torch.nn.functional"}


def load(path):
    spec = importlib.util.spec_from_file_location("sub", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sub"] = mod
    spec.loader.exec_module(mod)
    return mod


def screen(path):
    tree = ast.parse(open(path).read())
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
    bad = imports - ALLOWED_IMPORTS
    return sorted(imports), sorted(bad)


def digits(n, k=8):
    return [(n // 10 ** i) % 10 for i in range(k)]


def to_int(d):
    return sum(int(v) * 10 ** i for i, v in enumerate(d))


def rand_pairs(n, device, gen):
    a = torch.randint(0, 10, (n, 8), device=device, generator=gen)
    b = torch.randint(0, 10, (n, 8), device=device, generator=gen)
    a[:, 7] = torch.randint(1, 10, (n,), device=device, generator=gen)
    b[:, 7] = torch.randint(1, 10, (n,), device=device, generator=gen)
    return a, b


def enriched_pairs(n, device, gen, p=0.5):
    a, b = rand_pairs(n, device, gen)
    at = torch.randint(0, 10, (n, 8), device=device, generator=gen)
    at[:, 7] = torch.randint(1, 9, (n,), device=device, generator=gen)
    m = torch.rand((n, 8), device=device, generator=gen) < p
    a = torch.where(m, at, a)
    b = torch.where(m, 9 - at, b)
    return a, b


def pattern_pairs(patterns, device, gen, reps=8):
    """patterns: (M,8) in {0:kill, 1:transparent, 2:generate}."""
    M = patterns.shape[0]
    pat = patterns.repeat_interleave(reps, 0)
    n = pat.shape[0]
    a = torch.zeros((n, 8), dtype=torch.long, device=device)
    b = torch.zeros((n, 8), dtype=torch.long, device=device)
    for place in range(8):
        lo = 1 if place == 7 else 0
        p = pat[:, place]
        # kill: a+b < 9 ; transparent: a+b == 9 ; generate: a+b > 9
        av = torch.randint(lo, 10, (n,), device=device, generator=gen)
        bv = torch.randint(lo, 10, (n,), device=device, generator=gen)
        # kill
        ak = torch.randint(lo, 9, (n,), device=device, generator=gen)
        hi = (9 - ak).clamp(min=lo + 1)
        bk = lo + (torch.randint(0, 10, (n,), device=device, generator=gen)
                   % (hi - lo).clamp(min=1))
        ak = torch.where(ak + bk >= 9, torch.full_like(ak, lo), ak)
        bk = torch.where(ak + bk >= 9, torch.full_like(bk, lo), bk)
        # transparent
        at = torch.randint(lo, 9 - lo + 1, (n,), device=device, generator=gen)
        bt = 9 - at
        # generate
        ag = torch.randint(max(lo, 1), 10, (n,), device=device, generator=gen)
        bg = (10 - ag) + (torch.randint(0, 10, (n,), device=device,
                                        generator=gen) % ag)
        av = torch.where(p == 0, ak, torch.where(p == 1, at, ag))
        bv = torch.where(p == 0, bk, torch.where(p == 1, bt, bg))
        a[:, place] = av
        b[:, place] = bv
    return a, b


def true_digits(a, b):
    s = a + b
    carry = torch.zeros_like(s[:, :1])
    out = []
    for i in range(8):
        t = s[:, i:i + 1] + carry
        out.append(t % 10)
        carry = t // 10
    out.append(carry)
    return torch.cat(out, 1)


@torch.no_grad()
def batched_acc(model, a, b, chunk=200_000, attn_override=None):
    ok = 0
    for i in range(0, a.shape[0], chunk):
        aa, bb = a[i:i + chunk], b[i:i + chunk]
        if attn_override is None:
            logits = model(aa, bb)
        else:
            logits = model(aa, bb, attn_override=attn_override
                           .expand(aa.shape[0], -1, -1))
        pred = logits[:, 1:, :].argmax(-1)
        ok += int((pred == true_digits(aa, bb)).all(-1).sum())
    return ok / a.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="/workspace/submission.py")
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--api_n", type=int, default=3000)
    args = ap.parse_args()

    imports, bad = screen(args.path)
    print("imports:", imports, "| disallowed:", bad or "none")

    mod = load(args.path)
    model, meta = mod.build_model()
    n_params = sum(p.numel() for p in model.parameters())
    n_buf = sum(b.numel() for b in model.buffers())
    print(f"parameters: {n_params}  (buffers: {n_buf})  meta says "
          f"{meta.get('n_parameters')}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    gen = torch.Generator(device=device).manual_seed(20260903)

    a, b = rand_pairs(args.n, device, gen)
    acc_u = batched_acc(model, a, b)
    ae, be = enriched_pairs(args.n // 2, device, gen, p=0.5)
    acc_e = batched_acc(model, ae, be)
    ah, bh = enriched_pairs(args.n // 4, device, gen, p=0.85)
    acc_h = batched_acc(model, ah, bh)
    print(f"uniform held-out acc      : {acc_u:.6f}  (n={args.n})")
    print(f"transparent-enriched p=.50: {acc_e:.6f}")
    print(f"transparent-enriched p=.85: {acc_h:.6f}")

    pats = torch.tensor(list(itertools.product([0, 1, 2], repeat=8)),
                        device=device)
    pa, pb = pattern_pairs(pats, device, gen, reps=args.reps)
    logits = None
    with torch.no_grad():
        preds = []
        for i in range(0, pa.shape[0], 200_000):
            preds.append(model(pa[i:i + 200_000], pb[i:i + 200_000])
                         [:, 1:, :].argmax(-1))
        pred = torch.cat(preds)
    okp = (pred == true_digits(pa, pb)).all(-1)
    acc_p = float(okp.float().mean())
    per_pat = okp.view(pats.shape[0], args.reps).all(1)
    print(f"all 3^8 carry patterns    : {acc_p:.6f}  "
          f"({int((~per_pat).sum())} of {pats.shape[0]} patterns imperfect)")
    if int((~per_pat).sum()):
        bad_idx = (~per_pat).nonzero()[:10, 0].tolist()
        for i in bad_idx:
            print("   failing pattern:", pats[i].tolist())

    # ---- attention ablation -------------------------------------------
    with torch.no_grad():
        sub_a, sub_b = a[:20000], b[:20000]
        _, attn = model(sub_a, sub_b, return_attn=True)
        mean_attn = attn.mean(0, keepdim=True)
        am = attn.argmax(-1)
        var = (am != am[0:1]).any(0).float().mean().item()
        print(f"attention argmax varies on {var * 100:.1f}% of query positions")
        for src in [(1, "pos1"), (5, "pos5"), (9, "pos9")]:
            i, nm = src
            uniq = torch.unique(am[:, i]).tolist()
            print(f"   {nm} argmax targets seen: {uniq}")
    fro = batched_acc(model, a[:200000], b[:200000], attn_override=mean_attn)
    fro_h = batched_acc(model, ah[:200000], bh[:200000], attn_override=mean_attn)
    print(f"FROZEN-attention acc (uniform)  : {fro:.4f}")
    print(f"FROZEN-attention acc (p=.85)    : {fro_h:.4f}")

    # ---- public API path ----------------------------------------------
    cpu_model, _ = mod.build_model()
    ok = 0
    aa, bb = a[:args.api_n].cpu(), b[:args.api_n].cpu()
    for i in range(args.api_n):
        A, B = to_int(aa[i]), to_int(bb[i])
        ok += int(mod.add(cpu_model, A, B) == A + B)
    print(f"add() API accuracy (cpu)  : {ok / args.api_n:.4f}  (n={args.api_n})")

    edges = [(99999999, 99999999), (10000000, 10000000), (99999999, 10000001),
             (19999999, 10000001), (12345678, 87654321), (99999999, 10000000),
             (55555555, 44444445), (10000001, 89999999), (11111111, 88888889),
             (99999998, 10000002)]
    bad_edges = [(x, y, mod.add(cpu_model, x, y)) for x, y in edges
                 if mod.add(cpu_model, x, y) != x + y]
    print("edge cases:", "all correct" if not bad_edges else bad_edges)


if __name__ == "__main__":
    main()
