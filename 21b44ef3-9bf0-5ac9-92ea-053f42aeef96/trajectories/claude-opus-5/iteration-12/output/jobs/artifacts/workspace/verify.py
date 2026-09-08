"""Audit the graded file: accuracy, a whole-domain certificate, and ablations.

Nothing here is imported by submission.py.  This is the independent check that
what got shipped really does add, really uses its attention, and really is 12
parameters.
"""

import argparse
import importlib.util
import itertools
import torch

import data


def load(path):
    spec = importlib.util.spec_from_file_location("sub", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- accuracy --
@torch.no_grad()
def batch_exact(model, a, b, places, device):
    da, db, tgt, bucket = data.from_ints(a, b, places, device)
    pred = model(da, db).argmax(-1)
    ok = (pred[:, 1:] == tgt[:, 1:]).all(1)
    return ok, bucket


@torch.no_grad()
def random_accuracy(model, total, device, seed=0, chunk=131072):
    g = torch.Generator(device=device).manual_seed(seed)
    lo, hi = 10_000_000, 100_000_000
    nok = nheld = nheld_ok = 0
    done = 0
    while done < total:
        m = min(chunk, total - done)
        a = torch.randint(lo, hi, (m,), device=device, generator=g)
        b = torch.randint(lo, hi, (m,), device=device, generator=g)
        ok, bucket = batch_exact(model, a, b, 8, device)
        nok += int(ok.sum())
        held = bucket == 0
        nheld += int(held.sum())
        nheld_ok += int((ok & held).sum())
        done += m
    return nok / total, nheld_ok / max(nheld, 1), nheld


@torch.no_grad()
def structured_accuracy(model, device, seed=0):
    """Every one of the 3^8 carry-structure patterns, with random digits."""
    g = torch.Generator(device=device).manual_seed(seed)
    pats = torch.tensor(list(itertools.product([0, 1, 2], repeat=8)), device=device)
    n = pats.shape[0]
    a = torch.randint(0, 10, (n, 8), device=device, generator=g)
    b = torch.randint(0, 10, (n, 8), device=device, generator=g)
    # 0 = absorb (a+b<=8), 1 = transparent (a+b==9), 2 = generate (a+b>=10)
    a = torch.where(pats == 0, a % 9, a)
    b = torch.where(pats == 0, torch.remainder(b, 9 - a + 1), b)
    a = torch.where(pats == 1, a, a)
    b = torch.where(pats == 1, 9 - a, b)
    a2 = torch.clamp(a, 1, 9)
    b2 = 10 + torch.remainder(b, 9) - a2
    b2 = torch.clamp(b2, 0, 9)
    a = torch.where(pats == 2, a2, a)
    b = torch.where(pats == 2, b2, b)
    a[:, 7] = torch.clamp(a[:, 7], 1, 9)
    b[:, 7] = torch.clamp(b[:, 7], 1, 9)
    pw = 10 ** torch.arange(8, device=device, dtype=torch.long)
    ai, bi = (a * pw).sum(1), (b * pw).sum(1)
    ok, _ = batch_exact(model, ai, bi, 8, device)
    return float(ok.float().mean()), n


# ------------------------------------------------------------- certificate --
@torch.no_grad()
def certify(model, n=8, device="cpu"):
    """Exhaustive proof of exactness over all 10^(2n) inputs of width n.

    The token at a position depends only on that place's digit pair, and both
    heads are causal, so the residual written at answer position i is fixed by
    (a) the sequence of *key/value classes* at places <= i and (b) the digit
    pair at place i.  There are only a handful of classes, so enumerating
    class patterns x digit pairs covers the whole input domain exactly.
    """
    dt = torch.float64
    code = model.code().to(device, dt)
    bw, bb = model.bank_w.to(device, dt), model.bank_bias.to(device, dt)
    kw, vw = model.key_w.to(device, dt), model.val_w.to(device, dt)
    lam = model.dist_bias.to(device, dt)
    cw, fold = model.carry_w.to(device, dt), model.fold.to(device, dt)

    dig = torch.arange(10, device=device)
    x = code[dig][:, None] + code[dig][None, :]                      # (10,10)
    u = torch.clamp(x[..., None] * bw + bb, 0.0, 1.0)
    saturated = bool(((u == 0) | (u == 1)).all())
    k = (u * kw).sum(-1).reshape(-1)                                 # (100,)
    v = (u * vw).sum(-1).reshape(-1)
    xf = x.reshape(-1)

    kv = torch.stack([k, v], -1)
    uniq, cls = torch.unique(kv, dim=0, return_inverse=True)         # class per pair
    ncls = uniq.shape[0]

    s = dig[:, None] + dig[None, :]
    is_gen = (s >= 10).reshape(-1)
    is_tr = (s == 9).reshape(-1)
    # a class must determine the arithmetic role, else the enumeration below
    # would not be exhaustive
    role_ok = True
    roles = []
    for c in range(ncls):
        m = cls == c
        g, t = is_gen[m], is_tr[m]
        role_ok &= bool((g == g[0]).all() and (t == t[0]).all())
        roles.append((bool(g[0]), bool(t[0])))

    pats = torch.tensor(list(itertools.product(range(ncls), repeat=n)), device=device)
    npat, P = pats.shape[0], n + 2
    pad_cls = int(cls[0])                                            # pair (0,0)
    full = torch.cat([torch.full((npat, 1), pad_cls, device=device), pats,
                      torch.full((npat, 1), pad_cls, device=device)], 1)
    kseq = uniq[:, 0][full]                                          # (npat,P)
    vseq = uniq[:, 1][full]

    pos = torch.arange(P, device=device)
    delta = pos[:, None] - pos[None, :]
    logit = kseq[:, None, :] + lam * delta
    strict = logit.masked_fill(delta <= 0, -1e9).softmax(-1)
    incl = logit.masked_fill(delta < 0, -1e9).softmax(-1)
    h1 = (strict * vseq[:, None, :]).sum(-1)
    h2 = (incl * vseq[:, None, :]).sum(-1)
    shift = cw * h1 + fold * h2                                      # (npat,P)

    # carry into each place, from the arithmetic roles of the pattern classes
    gen_of = torch.tensor([r[0] for r in roles], device=device)
    tr_of = torch.tensor([r[1] for r in roles], device=device)
    carry = torch.zeros(npat, n + 1, dtype=torch.long, device=device)
    for j in range(n):
        g = gen_of[pats[:, j]].long()
        t = tr_of[pats[:, j]].long()
        carry[:, j + 1] = g + t * carry[:, j]

    worst = float("inf")
    nchecked = 0
    wrong = 0
    for i in range(1, P):
        j = i - 1                                                    # place index
        cin = carry[:, j]
        if j < n:
            members = [torch.nonzero(cls == c).reshape(-1) for c in range(ncls)]
            for c in range(ncls):
                idx = members[c]
                sel = pats[:, j] == c
                if not bool(sel.any()) or idx.numel() == 0:
                    continue
                y = xf[idx][None, :] + shift[sel, i][:, None]        # (npat_c, npairs)
                d = (y[..., None] - code).abs()                      # distance to prototypes
                nearest = d.argmin(-1)
                truth = (s.reshape(-1)[idx][None, :] + cin[sel][:, None]) % 10
                wrong += int((nearest != truth).sum())
                dd = d.clone()
                dd.scatter_(-1, truth.unsqueeze(-1), float("inf"))
                margin = dd.min(-1).values - d.gather(-1, truth.unsqueeze(-1)).squeeze(-1)
                worst = min(worst, float(margin.min()))
                nchecked += margin.numel()
        else:                                                        # carry-out place, pair (0,0)
            y = xf[0] + shift[:, i]
            d = (y[:, None] - code).abs()
            nearest = d.argmin(-1)
            truth = cin % 10
            wrong += int((nearest != truth).sum())
            dd = d.clone()
            dd.scatter_(-1, truth.unsqueeze(-1), float("inf"))
            margin = dd.min(-1).values - d.gather(-1, truth.unsqueeze(-1)).squeeze(-1)
            worst = min(worst, float(margin.min()))
            nchecked += margin.numel()

    gaps = (code.sort().values.diff()).min()
    return {"saturated": saturated, "classes": ncls, "role_ok": bool(role_ok),
            "patterns": npat, "checks": nchecked, "wrong": wrong,
            "min_margin": worst, "min_code_gap": float(gaps),
            "roles": roles, "keys": uniq[:, 0].tolist(), "values": uniq[:, 1].tolist()}


# ---------------------------------------------------------------- ablation --
@torch.no_grad()
def attention_ablation(model, device, samples=8192, seed=3):
    """Replace the input-dependent attention map with its batch mean.

    If attention were a fixed pattern in disguise this would change nothing.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    a = torch.randint(10_000_000, 100_000_000, (samples,), device=device, generator=g)
    b = torch.randint(10_000_000, 100_000_000, (samples,), device=device, generator=g)
    da, db, tgt, _ = data.from_ints(a, b, 8, device)
    code = model.code()
    x = code[da] + code[db]
    u = torch.clamp(x.unsqueeze(-1) * model.bank_w + model.bank_bias, 0.0, 1.0)
    k = (u * model.key_w).sum(-1)
    v = (u * model.val_w).sum(-1)
    P = x.shape[-1]
    pos = torch.arange(P, device=device)
    delta = pos[:, None] - pos[None, :]
    logit = k.unsqueeze(-2) + model.dist_bias * delta
    strict = logit.masked_fill(delta <= 0, -1e9).softmax(-1)
    incl = logit.masked_fill(delta < 0, -1e9).softmax(-1)
    npat = len(torch.unique(strict.argmax(-1), dim=0))
    fs, fi = strict.mean(0, keepdim=True), incl.mean(0, keepdim=True)
    out = {}
    for name, (sa, ia) in {"live": (strict, incl), "frozen": (fs, fi)}.items():
        y = x + model.carry_w * (sa * v.unsqueeze(-2)).sum(-1) + model.fold * (ia * v.unsqueeze(-2)).sum(-1)
        pred = (-(y.unsqueeze(-1) - code) ** 2).argmax(-1)
        out[name] = float((pred[:, 1:] == tgt[:, 1:]).all(1).float().mean())
    out["distinct_attention_patterns"] = npat
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="submission.py")
    ap.add_argument("--samples", type=int, default=1_048_576)
    ap.add_argument("--skip_cert", action="store_true")
    args = ap.parse_args()

    mod = load(args.file)
    model, meta = mod.build_model()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)

    nparam = sum(p.numel() for p in model.parameters())
    print(f"parameters: {nparam}")
    for n_, p in model.named_parameters():
        print(f"    param  {n_:<12} {tuple(p.shape)} {p.detach().reshape(-1).tolist()}")
    for n_, b in model.named_buffers():
        print(f"    buffer {n_:<12} {tuple(b.shape)} {b.reshape(-1).tolist()}")
    print(f"metadata parameters field: {meta.get('parameters')}")

    acc, held, nheld = random_accuracy(model, args.samples, dev)
    print(f"uniform 8-digit exact match: {acc:.6f} on {args.samples}")
    print(f"held-out split (bucket 0)  : {held:.6f} on {nheld}")

    sacc, sn = structured_accuracy(model, dev)
    print(f"all {sn} carry patterns      : {sacc:.6f}")

    edges = [(99999999, 99999999), (10000000, 10000000), (99999999, 10000001),
             (11111111, 88888889), (19999999, 10000001), (12345678, 87654322),
             (50000000, 50000000), (99999999, 99999998), (10000001, 89999999)]
    bad = [(a, b) for a, b in edges if mod.add(model, a, b) != a + b]
    print(f"edge cases via add(): {len(edges)-len(bad)}/{len(edges)} ok {bad}")

    g = torch.Generator().manual_seed(7)
    ra = torch.randint(10_000_000, 100_000_000, (400,), generator=g).tolist()
    rb = torch.randint(10_000_000, 100_000_000, (400,), generator=g).tolist()
    api_bad = sum(1 for a, b in zip(ra, rb) if mod.add(model, a, b) != a + b)
    print(f"add() interface on 400 random pairs: {400-api_bad}/400 ok")

    abl = attention_ablation(model, dev)
    print(f"attention ablation: live {abl['live']:.4f} -> frozen {abl['frozen']:.4f} "
          f"({abl['distinct_attention_patterns']} distinct attention maps in 8192 inputs)")

    if not args.skip_cert:
        cert = certify(model, 8, dev)
        print(f"certificate: classes={cert['classes']} roles={cert['roles']} "
              f"saturated={cert['saturated']} role_ok={cert['role_ok']}")
        print(f"  keys={[round(v,4) for v in cert['keys']]} values={[round(v,4) for v in cert['values']]}")
        print(f"  patterns={cert['patterns']} checks={cert['checks']} wrong={cert['wrong']} "
              f"min_margin={cert['min_margin']:.6g} (code gap {cert['min_code_gap']:.6g})")

    # float64 / device agreement
    m64 = mod.build_model()[0].double()
    a = torch.randint(10_000_000, 100_000_000, (20000,), generator=g)
    b = torch.randint(10_000_000, 100_000_000, (20000,), generator=g)
    da, db, tgt, _ = data.from_ints(a, b, 8, "cpu")
    p32 = mod.build_model()[0](da, db).argmax(-1)
    p64 = m64(da, db).argmax(-1)
    print(f"cpu float32 vs float64 argmax agreement: {float((p32==p64).all(1).float().mean()):.6f}")
    print(f"cpu float32 exact match: {float((p32[:,1:]==tgt[:,1:]).all(1).float().mean()):.6f}")


if __name__ == "__main__":
    main()
