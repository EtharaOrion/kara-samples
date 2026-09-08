"""Audit a submission file: accuracy, coverage, robustness, attention ablation."""
import argparse
import importlib.util
import math
import sys

import torch

import lab

DEV = lab.DEV


def load(path):
    spec = importlib.util.spec_from_file_location("_sub", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def decode(model, ad, bd):
    """Batched answer digits from one forward pass (same path as add())."""
    with torch.no_grad():
        return model(ad, bd).argmax(-1)[:, 1:]


def exact(model, ad, bd, tgt):
    return (decode(model, ad, bd) == tgt[:, 1:]).all(-1)


# ---------------------------------------------------------------- carry patterns
def pattern_pairs(reps=1, seed=0, dev=DEV):
    """One operand pair per (absorb / transparent / generate) pattern, 3^8 of them."""
    g = torch.Generator(device=dev).manual_seed(seed)
    n = 8
    pat = torch.arange(3 ** n, device=dev)
    dig = torch.stack([(pat // 3 ** i) % 3 for i in range(n)], 1)      # (6561, n)
    dig = dig.repeat(reps, 1)
    N = dig.shape[0]
    lo = torch.where(dig == 0, 0, torch.where(dig == 1, 9, 10))
    hi = torch.where(dig == 0, 8, torch.where(dig == 1, 9, 18))
    msb = torch.zeros_like(lo, dtype=torch.bool)
    msb[:, n - 1] = True
    lo = torch.where(msb & (dig == 0), torch.full_like(lo, 2), lo)
    u = torch.rand(N, n, generator=g, device=dev)
    s = torch.minimum(lo + (u * (hi - lo + 1).float()).long(), hi)
    amin = torch.where(msb, torch.ones_like(s), torch.zeros_like(s))
    a_lo = torch.maximum(amin, s - 9)
    a_hi = torch.minimum(torch.full_like(s, 9), s - amin)
    u2 = torch.rand(N, n, generator=g, device=dev)
    a = a_lo + (u2 * (a_hi - a_lo + 1).float()).long()
    a = torch.minimum(a, a_hi)
    b = s - a
    assert int((a < 0).sum()) == 0 and int((b < 0).sum()) == 0
    assert int((a > 9).sum()) == 0 and int((b > 9).sum()) == 0
    assert int((a[:, n - 1] < 1).sum()) == 0 and int((b[:, n - 1] < 1).sum()) == 0
    cls = torch.where(s <= 8, 0, torch.where(s == 9, 1, 2))
    assert bool((cls == dig).all()), "pattern construction mismatch"
    tgt = digits_sum(a, b)
    return a, b, tgt


def digits_sum(a, b):
    n = a.shape[1]
    s = a + b
    outs, carry = [], torch.zeros(a.shape[0], dtype=torch.long, device=a.device)
    for i in range(n):
        v = s[:, i] + carry
        outs.append(v % 10)
        carry = v // 10
    outs.append(carry)
    return torch.stack([torch.zeros_like(carry)] + outs, 1)


# ---------------------------------------------------------------- mirrored forward
def parts(model, ad, bd, freeze=None):
    """Mirror of Adder.forward, optionally with the attention maps frozen.

    ``freeze`` replaces both attention matrices by a fixed pattern (their mean
    over the batch), which removes every input dependence of the attention
    while leaving the rest of the model untouched.
    """
    cfg = model.cfg
    B, n = ad.shape
    pad = torch.zeros(B, 1, dtype=ad.dtype, device=ad.device)
    ai = torch.cat([pad, ad, pad], 1)
    bi = torch.cat([pad, bd, pad], 1)
    code = torch.cat([model.code_fix, model.code], 0)
    x = code[ai] + code[bi]
    u = torch.clamp(x @ model.bw.t() + model.bb, 0.0, 1.0)
    key = u @ model.kw
    val = u @ model.vw
    p = n + 2
    idx = torch.arange(p, device=ad.device)
    dist = idx[:, None] - idx[None, :]
    score = key[:, None, :] + model.lam * dist
    blk = torch.finfo(score.dtype).min
    eye0 = (idx[:, None] == 0) & (idx[None, :] == 0)
    A = torch.softmax(torch.where((dist >= 1) | eye0, score, blk), -1)
    Bm = torch.softmax(torch.where(dist >= 0, score, blk), -1)
    if freeze is not None:
        A = freeze[0].expand_as(A)
        Bm = freeze[1].expand_as(Bm)
    oa = (A @ val[..., None]).squeeze(-1)
    ob = (Bm @ val[..., None]).squeeze(-1)
    r = x + oa[..., None] * model.e1 + ob[..., None] * model.e2
    if cfg["rb"]:
        r = r + model.rb
    logits = -model.ls * ((r[:, :, None, :] - code) ** 2).sum(-1)
    return dict(x=x, u=u, key=key, val=val, A=A, B=Bm, r=r, logits=logits, code=code)


def margin(model, ad, bd):
    """Smallest distance from the residual to a read-out decision boundary,
    in units of the smallest gap between prototypes."""
    with torch.no_grad():
        d = parts(model, ad, bd)
        r, code = d["r"].double(), d["code"].double()
        pred = d["logits"].argmax(-1)                       # (B, P)
        pc = code[pred]                                     # (B, P, C)
        diff = pc[:, :, None, :] - code[None, None, :, :]   # (B, P, 10, C)
        mid = (pc[:, :, None, :] + code[None, None, :, :]) / 2
        num = ((r[:, :, None, :] - mid) * diff).sum(-1).abs()
        den = diff.norm(dim=-1)
        dd = torch.where(den > 1e-12, num / den.clamp(min=1e-12), torch.full_like(num, 1e30))
        gaps = (code[:, None, :] - code[None, :, :]).norm(dim=-1)
        gaps = gaps[gaps > 1e-12].min()
        return float(dd.min() / gaps), float(gaps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="/workspace/submission.py")
    ap.add_argument("--n_uniform", type=int, default=1 << 20)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    mod = load(a.sub)
    model, meta = mod.build_model()
    model = model.to(DEV)
    npar = sum(p.numel() for p in model.parameters())
    print(f"file            : {a.sub}")
    print(f"parameters      : {npar}   (metadata says {meta.get('n_params')})")
    print(f"buffers         : {[ (k, tuple(v.shape)) for k,v in model.named_buffers() ]}")

    # ---- uniform held-out accuracy -----------------------------------------
    g = torch.Generator(device=DEV).manual_seed(20260905)
    tot = ok = 0
    ho_tot = ho_ok = 0
    B = 1 << 16
    while tot < a.n_uniform:
        ad, bd, tgt, bk = lab.uniform_pairs(B, 8, g)
        e = exact(model, ad, bd, tgt)
        ok += int(e.sum()); tot += B
        sel = bk == 0
        ho_ok += int(e[sel].sum()); ho_tot += int(sel.sum())
    print(f"uniform 8-digit : {ok}/{tot} = {ok/tot:.6f}")
    print(f"  held-out only : {ho_ok}/{ho_tot} = {ho_ok/max(ho_tot,1):.6f}")

    # ---- structured / carry chains -----------------------------------------
    tot2 = ok2 = 0
    for _ in range(8):
        ad, bd, tgt, bk = lab.sample(B, 8, g)
        ok2 += int(exact(model, ad, bd, tgt).sum()); tot2 += B
    print(f"carry-enriched  : {ok2}/{tot2} = {ok2/tot2:.6f}")

    # ---- every carry pattern ------------------------------------------------
    ad, bd, tgt = pattern_pairs(reps=a.reps, seed=5)
    e = exact(model, ad, bd, tgt)
    print(f"3^8 patterns    : {int(e.sum())}/{e.numel()} = {float(e.float().mean()):.6f}")
    if int(e.sum()) < e.numel():
        bad = torch.nonzero(~e)[:5, 0]
        for i in bad.tolist():
            print("   fail:", ad[i].tolist(), bd[i].tolist())

    # ---- edges --------------------------------------------------------------
    edges = [(10000000, 10000000), (99999999, 99999999), (10000000, 99999999),
             (19999999, 10000001), (12345678, 87654321), (55555555, 44444445),
             (99999999, 10000001), (50000000, 50000000), (98765432, 12345678),
             (11111111, 88888889), (10000001, 89999999), (99999998, 99999999)]
    bad = [(x, y, mod.add(model, x, y)) for x, y in edges if mod.add(model, x, y) != x + y]
    print(f"edge cases      : {len(edges)-len(bad)}/{len(edges)}" + (f"  FAIL {bad}" if bad else ""))

    # ---- add() agrees with the batched decode -------------------------------
    ad, bd, tgt, _ = lab.uniform_pairs(512, 8, g)
    pw = 10 ** torch.arange(8, device=DEV, dtype=torch.long)
    av, bv = (ad * pw).sum(1), (bd * pw).sum(1)
    okadd = sum(1 for i in range(512) if mod.add(model, int(av[i]), int(bv[i])) == int(av[i]) + int(bv[i]))
    print(f"add() on 512    : {okadd}/512")

    if a.quick:
        return

    # ---- decision margin ----------------------------------------------------
    ad, bd, tgt, _ = lab.sample(1 << 15, 8, g)
    mg, gap = margin(model, ad, bd)
    print(f"margin          : {mg:.5f} prototype gaps (gap={gap:.5f})")

    # ---- float64 / device agreement ----------------------------------------
    m64 = mod.build_model()[0].to(DEV).double()
    with torch.no_grad():
        p32 = model(ad, bd).argmax(-1)
        p64 = m64(ad, bd).argmax(-1)
    mcpu = mod.build_model()[0]
    with torch.no_grad():
        pcpu = mcpu(ad.cpu(), bd.cpu()).argmax(-1)
    print(f"float64 agree   : {float((p32==p64).all())}   cpu agree: {float((p32.cpu()==pcpu).all())}")

    # ---- attention ablation -------------------------------------------------
    with torch.no_grad():
        d = parts(model, ad, bd)
        base = (d["logits"].argmax(-1)[:, 1:] == tgt[:, 1:]).all(-1).float().mean()
        fz = (d["A"].mean(0, keepdim=True), d["B"].mean(0, keepdim=True))
        d2 = parts(model, ad, bd, freeze=fz)
        frozen = (d2["logits"].argmax(-1)[:, 1:] == tgt[:, 1:]).all(-1).float().mean()
        pats = d["A"].argmax(-1).unique(dim=0).shape[0]
    print(f"attention       : mirror acc {float(base):.6f} -> frozen-map acc {float(frozen):.6f}"
          f"   ({pats} distinct strict-head argmax patterns over {ad.shape[0]} inputs)")

    # ---- length generalisation ---------------------------------------------
    for n in (3, 5, 12, 16):
        acc, cnt = lab.model_acc(model.to(DEV), n=n, batches=2, B=8192, held_out=None)
        print(f"  n={n:<3d} places : {acc:.6f}")


if __name__ == "__main__":
    main()
