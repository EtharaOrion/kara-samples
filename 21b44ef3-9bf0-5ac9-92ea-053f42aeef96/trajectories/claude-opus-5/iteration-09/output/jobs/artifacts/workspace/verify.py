"""Audit a built submission.py: accuracy, carry structure, edge cases,
attention ablations, numeric agreement."""
import argparse
import ast
import importlib.util
import itertools
import sys
import torch

from data import pad_and_label, Sampler


def load(path):
    spec = importlib.util.spec_from_file_location("sub", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def batch_acc(model, a, b, tgt, dev="cpu"):
    ap, bp, t = pad_and_label(a, b)
    pred = model(ap.to(dev), bp.to(dev))[0, :, 1:, :].argmax(-1).cpu()
    return (pred == t).all(1)


def imports_of(path):
    tree = ast.parse(open(path).read())
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            mods.add(n.module.split(".")[0])
    return mods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="/workspace/submission.py")
    ap.add_argument("--n", type=int, default=1 << 20)
    ap.add_argument("--dev", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.dev

    print("imports:", sorted(imports_of(args.sub)))
    sub = load(args.sub)
    model, meta = sub.build_model()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"parameters {n_par}   metadata says {meta.get('parameters')}")
    model = model.to(dev)

    # --- headline accuracy on unseen uniform 8-digit pairs -------------------
    g = torch.Generator(device="cpu").manual_seed(12345)
    ok = tot = 0
    B = 1 << 14
    for _ in range(max(1, args.n // B)):
        a = torch.randint(0, 10, (B, 8), generator=g)
        b = torch.randint(0, 10, (B, 8), generator=g)
        a[:, 7].clamp_(min=1)
        b[:, 7].clamp_(min=1)
        r = batch_acc(model, a, b, None, dev)
        ok += r.sum().item()
        tot += B
    print(f"uniform 8-digit exact match: {ok}/{tot} = {ok/tot:.6f}")

    # --- the same, restricted to the 1-in-16 of pair space never trained on ---
    smp = Sampler(dev)
    hok = htot = 0
    for _ in range(max(1, args.n // B // 4)):
        a, b, _t = smp.batch(B, n=8, split="val")
        r = batch_acc(model, a[:, 1:-1].cpu(), b[:, 1:-1].cpu(), None, dev)
        hok += r.sum().item()
        htot += a.shape[0]
    print(f"held-out (split=val) 8-digit exact match: {hok}/{htot} = {hok/htot:.6f}")

    # --- every carry structure: each place absorb / transparent / generate ---
    tabs = [[(x, y) for x in range(10) for y in range(10) if x + y <= 8],
            [(x, y) for x in range(10) for y in range(10) if x + y == 9],
            [(x, y) for x in range(10) for y in range(10) if x + y >= 10]]
    pats = list(itertools.product(range(3), repeat=8))
    A = torch.zeros(len(pats), 8, dtype=torch.long)
    Bd = torch.zeros(len(pats), 8, dtype=torch.long)
    gg = torch.Generator().manual_seed(7)
    bad = 0
    for rep in range(4):
        for i, p in enumerate(pats):
            for j, c in enumerate(p):
                t = tabs[c]
                k = int(torch.randint(0, len(t), (1,), generator=gg))
                A[i, j], Bd[i, j] = t[k]
        A[:, 7].clamp_(min=1)
        Bd[:, 7].clamp_(min=1)
        r = batch_acc(model, A, Bd, None, dev)
        bad += (~r).sum().item()
    print(f"all {len(pats)} carry patterns x4 draws: {4*len(pats)-bad}/{4*len(pats)} correct")

    # --- edge cases ----------------------------------------------------------
    edges = [(10000000, 10000000), (99999999, 99999999), (10000000, 99999999),
             (19999999, 10000001), (99999999, 10000001), (55555555, 44444445),
             (12345678, 87654321), (50000000, 50000000), (99999998, 10000001),
             (11111111, 88888889)]
    ebad = [e for e in edges if sub.add(model, *e) != e[0] + e[1]]
    print(f"edge cases: {len(edges)-len(ebad)}/{len(edges)} correct", ebad if ebad else "")

    # --- add() agrees with the batched path ---------------------------------
    a = torch.randint(0, 10, (256, 8), generator=g)
    b = torch.randint(0, 10, (256, 8), generator=g)
    a[:, 7].clamp_(min=1)
    b[:, 7].clamp_(min=1)
    ints = [(sum(int(a[i, j]) * 10 ** j for j in range(8)),
             sum(int(b[i, j]) * 10 ** j for j in range(8))) for i in range(256)]
    n_ok = sum(sub.add(model, x, y) == x + y for x, y in ints)
    print(f"add() on 256 unseen pairs: {n_ok}/256")

    # --- is the attention doing work? ---------------------------------------
    a = torch.randint(0, 10, (4096, 8), generator=g)
    b = torch.randint(0, 10, (4096, 8), generator=g)
    a[:, 7].clamp_(min=1)
    b[:, 7].clamp_(min=1)
    base = batch_acc(model, a, b, None, dev).float().mean().item()
    ap_, bp_, _ = pad_and_label(a, b)
    with torch.no_grad():
        _, parts = model(ap_.to(dev), bp_.to(dev), return_parts=True)
    at = parts["attn"][0]
    pat = at.argmax(-1)
    uniq = len({tuple(r.tolist()) for r in pat})
    print(f"attention argmax patterns over 4096 inputs: {uniq} distinct  "
          f"(base accuracy {base:.4f})")

    orig = torch.softmax
    for mode in ("mean", "prev"):
        def patched(sc, dim=-1, _m=mode):
            a_ = orig(sc, dim=dim)
            if _m == "mean":
                return a_.mean(dim=1, keepdim=True).expand_as(a_)
            P = a_.shape[-1]
            idx = torch.arange(P, device=a_.device)
            oh = ((idx[None, :] == (idx[:, None] - 1)) |
                  ((idx[:, None] == 0) & (idx[None, :] == 0))).to(a_.dtype)
            return oh.expand_as(a_)
        torch.softmax = patched
        try:
            abl = batch_acc(model, a, b, None, dev).float().mean().item()
        finally:
            torch.softmax = orig
        print(f"  ablation attn={mode}: accuracy {abl:.4f}")

    # --- numeric robustness --------------------------------------------------
    m64 = sub.build_model()[0].double().to(dev)
    p32 = model(ap_.to(dev), bp_.to(dev))[0, :, 1:, :].argmax(-1).cpu()
    p64 = m64(ap_.to(dev), bp_.to(dev))[0, :, 1:, :].argmax(-1).cpu()
    mcpu = sub.build_model()[0]
    pcpu = mcpu(ap_, bp_)[0, :, 1:, :].argmax(-1)
    print(f"float32 vs float64 argmax agreement: {(p32==p64).all().item()}   "
          f"cpu vs {dev}: {(p32==pcpu).all().item()}")
    with torch.no_grad():
        lg = model(ap_.to(dev), bp_.to(dev))[0, :, 1:, :].float()
    top2 = lg.topk(2, -1).values
    print(f"min logit margin over 4096x9 decisions: {(top2[...,0]-top2[...,1]).min().item():.4g}")


if __name__ == "__main__":
    main()
