"""Independent verification of the shipped /workspace/submission.py.

Loads the graded file by path (nothing else from this repo is imported into it) and
runs six checks:

  1. shape       parameter/buffer inventory; every learned float is an nn.Parameter
                 and every buffer is a hand-set constant, not a trained value
  2. held-out    >= 1e6 unseen operand pairs, exact match, plus edge cases
  3. domain      the whole-domain certificate from certify.py re-run on the shipped
                 literals (all 3^8 class patterns, bank saturation, read-out margin)
  4. numerics    float32 vs float64, CPU vs CUDA, batched vs single-example
  5. width       the same weights run at other digit widths (5, 8, 11, 16)
  6. attention   ablations showing the self-attention does real, input-dependent work

The held-out split is the same hash bucket that training never sampled from
(data._hash_bucket(a, b) == 0), so nothing here was seen during training.
"""

import argparse, importlib.util, random, sys
import torch

import data


def load(path):
    spec = importlib.util.spec_from_file_location("submission", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def decode(mod, model, A, B, n):
    """Batched equivalent of mod.add: argmax of one forward pass, recombined."""
    pred = model(A, B).argmax(-1)                                  # [N,P]
    w = (10 ** torch.arange(n + 1, device=A.device, dtype=torch.int64))
    return (pred[:, 1:n + 2] * w).sum(1)


def tokenize(a, b, n, dev):
    i = torch.arange(n, device=dev)
    da = (a[:, None] // 10 ** i) % 10
    db = (b[:, None] // 10 ** i) % 10
    z = torch.zeros(a.shape[0], 1, dtype=torch.long, device=dev)
    return torch.cat([z, da, z], 1), torch.cat([z, db, z], 1)


# ---------------------------------------------------------------- 1. shape

def check_shape(mod, model, meta):
    print("1. shape")
    tot = sum(p.numel() for p in model.parameters())
    for n_, p in model.named_parameters():
        print(f"     param  {n_:<10} {str(tuple(p.shape)):<8} {p.numel():>3}  "
              f"requires_grad={p.requires_grad}")
    for n_, b in model.named_buffers():
        vals = b.reshape(-1).tolist()
        print(f"     buffer {n_:<10} {str(tuple(b.shape)):<8} {b.numel():>3}  {vals}")
    assert isinstance(model, torch.nn.Module), "not an nn.Module"
    assert meta["n_parameters"] == tot, "metadata disagrees with the real count"
    # every buffer must be a hand-set constant: a short round number, not a trained float
    allowed = {0.0, 1.0, -1.0, 8.0, -8.0, 400.0}
    for n_, b in model.named_buffers():
        for v in b.reshape(-1).tolist():
            assert v in allowed, f"buffer {n_} holds {v!r}, which is not a fixed constant"
    print(f"   -> {tot} parameters, {sum(b.numel() for b in model.buffers())} constant "
          f"buffers; metadata agrees. OK\n")
    return tot


# ---------------------------------------------------------------- 2. held-out

def check_heldout(mod, model, dev, total=1 << 20, bs=8192, seed=1234):
    print("2. held-out accuracy (unseen pairs: hash bucket 0, never sampled in training)")
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    LO, HI = 10_000_000, 99_999_999
    seen = wrong = 0
    worst = []
    while seen < total:
        a = torch.randint(LO, HI + 1, (bs * 24,), generator=gen, device=dev)
        b = torch.randint(LO, HI + 1, (bs * 24,), generator=gen, device=dev)
        A, B = tokenize(a, b, 8, dev)
        keep = data._hash_bucket(A[:, 1:9], B[:, 1:9]) == 0   # the held-out bucket
        a, b, A, B = a[keep][:bs], b[keep][:bs], A[keep][:bs], B[keep][:bs]
        if a.numel() == 0:
            continue
        got = decode(mod, model, A, B, 8)
        bad = got != (a + b)
        wrong += int(bad.sum()); seen += a.numel()
        if bad.any() and len(worst) < 5:
            k = int(bad.nonzero()[0])
            worst.append((int(a[k]), int(b[k]), int(got[k]), int(a[k] + b[k])))
    acc = (seen - wrong) / seen
    print(f"     {seen - wrong}/{seen} exact = {acc:.6f}")
    for w in worst:
        print(f"     miss: {w[0]} + {w[1]} -> {w[2]} (want {w[3]})")

    edge = [(10_000_000, 10_000_000), (99_999_999, 99_999_999),
            (10_000_000, 99_999_999), (99_999_999, 10_000_000),
            (19_999_999, 10_000_001), (99_999_999, 10_000_001),
            (12_345_678, 87_654_321), (55_555_555, 44_444_445),
            (11_111_111, 88_888_889), (50_000_000, 50_000_000),
            (99_999_998, 10_000_002), (45_454_545, 54_545_455)]
    ebad = [(a, b, mod.add(model, a, b)) for a, b in edge if mod.add(model, a, b) != a + b]
    print(f"     edge cases: {len(edge) - len(ebad)}/{len(edge)}"
          f"{'' if not ebad else '  FAILURES: ' + str(ebad)}")
    print(f"   -> {'OK' if acc >= 0.99 and not ebad else 'FAIL'}\n")
    return acc


# ---------------------------------------------------------------- 3. whole domain

def check_domain(model, dev):
    print("3. whole-domain certificate (re-run on the shipped literals)")
    import certify as C
    code = torch.cat([model.code_zero, model.code_free]).to(torch.float64).to(dev)
    p = {"code": code[None], "rb": torch.zeros(1, dtype=torch.float64, device=dev),
         "bw": model.bw.to(torch.float64)[None].to(dev),
         "bb": model.bb.to(torch.float64)[None].to(dev),
         "kw": model.kw.to(torch.float64)[None].to(dev),
         "vw": model.vw.to(torch.float64)[None].to(dev),
         "vb": torch.zeros(1, dtype=torch.float64, device=dev),
         "q": torch.ones(1, dtype=torch.float64, device=dev),
         "lam": model.lam.to(torch.float64)[None].to(dev),
         "e": torch.stack([model.carry_w, model.fold]).to(torch.float64)[None].to(dev),
         "ls": torch.ones(1, dtype=torch.float64, device=dev)}
    ok, info = C.certify(p, n=8, verbose=True)
    print(f"   -> {'OK' if ok else 'FAIL'}\n")
    return ok


# ---------------------------------------------------------------- 4. numerics

def check_numerics(mod, model, dev, n_cases=4096, seed=7):
    print("4. numerics")
    g = torch.Generator(device=dev); g.manual_seed(seed)
    a = torch.randint(10_000_000, 100_000_000, (n_cases,), generator=g, device=dev)
    b = torch.randint(10_000_000, 100_000_000, (n_cases,), generator=g, device=dev)
    A, B = tokenize(a, b, 8, dev)
    base = decode(mod, model, A, B, 8)

    m64 = mod.build_model()[0].to(dev).to(torch.float64)
    got64 = decode(mod, m64, A, B, 8)
    print(f"     float32 vs float64 : {int((got64 == base).sum())}/{n_cases} identical")

    mcpu = mod.build_model()[0]
    gotcpu = decode(mod, mcpu, A.cpu(), B.cpu(), 8)
    print(f"     cuda vs cpu        : {int((gotcpu == base.cpu()).sum())}/{n_cases} identical")

    single = torch.tensor([mod.add(model, int(x), int(y)) for x, y in
                           zip(a[:256].tolist(), b[:256].tolist())], device=dev)
    print(f"     batched vs single  : {int((single == base[:256]).sum())}/256 identical")
    ok = bool((got64 == base).all() and (gotcpu == base.cpu()).all()
              and (single == base[:256]).all())
    print(f"   -> {'OK' if ok else 'FAIL'}\n")
    return ok


# ---------------------------------------------------------------- 5. width

def check_width(mod, model, dev, widths=(5, 8, 11, 16), n_cases=4096, seed=11):
    print("5. width generalisation (identical weights, no position-dependent parameters)")
    g = torch.Generator(device=dev); g.manual_seed(seed)
    ok = True
    for n in widths:
        hi = 10 ** n - 1
        a = torch.randint(10 ** (n - 1), hi + 1, (n_cases,), generator=g, device=dev)
        b = torch.randint(10 ** (n - 1), hi + 1, (n_cases,), generator=g, device=dev)
        A, B = tokenize(a, b, n, dev)
        acc = float((decode(mod, model, A, B, n) == a + b).double().mean())
        star = " (trained width)" if n == 8 else ""
        print(f"     {n:>2} digits: {acc:.6f}{star}")
        ok &= acc >= 0.99
    print(f"   -> {'OK' if ok else 'FAIL'}\n")
    return ok


# ---------------------------------------------------------------- 6. attention

def check_attention(mod, model, dev, n_cases=8192, seed=23):
    """The attention map must depend on the input, and the answer must depend on it.

    Run on two input sets. `uniform` is the graded distribution. `carry-heavy` mixes in
    transparent places (a_i + b_i == 9), which is where the choice of attended position
    actually varies: a transparent place must be skipped, so the query has to look past
    it to the nearest earlier place that decides the carry. A head whose pattern were
    fixed could not do that, and the recency-only ablation below is exactly that head.
    """
    print("6. attention ablations")
    code = torch.cat([model.code_zero, model.code_free])
    P = 10
    idx = torch.arange(P, device=dev)
    dist = (idx[:, None] - idx[None, :]).float()
    strict = (dist >= 1.0).clone(); strict[0, 0] = True
    mask = torch.stack([strict, dist >= 0.0], 0)

    def run(A, B, att_fn):
        u = code[A] + code[B]
        gg = torch.clamp(u.unsqueeze(-1) * model.bw + model.bb, 0.0, 1.0)
        k = (gg * model.kw).sum(-1)
        v = (gg * model.vw).sum(-1)
        lg = k[:, None, None, :] + model.lam[None, :, None, None] * dist[None, None]
        lg = lg.masked_fill(~mask[None], torch.finfo(lg.dtype).min / 4)
        att = att_fn(torch.softmax(lg, dim=-1))
        c = (att * v[:, None, None, :]).sum(-1)
        z = u + model.carry_w * c[:, 0, :] + model.fold * c[:, 1, :]
        pred = (-(z.unsqueeze(-1) - code) ** 2).argmax(-1)
        w = 10 ** torch.arange(9, device=dev, dtype=torch.long)
        return (pred[:, 1:10] * w).sum(1)

    def att_map(A, B):
        u = code[A] + code[B]
        gg = torch.clamp(u.unsqueeze(-1) * model.bw + model.bb, 0.0, 1.0)
        k = (gg * model.kw).sum(-1)
        lg = k[:, None, None, :] + model.lam[None, :, None, None] * dist[None, None]
        lg = lg.masked_fill(~mask[None], torch.finfo(lg.dtype).min / 4)
        return torch.softmax(lg, dim=-1)

    g = torch.Generator(device=dev); g.manual_seed(seed)
    sets = {}
    a = torch.randint(10_000_000, 100_000_000, (n_cases,), generator=g, device=dev)
    b = torch.randint(10_000_000, 100_000_000, (n_cases,), generator=g, device=dev)
    sets["uniform"] = tokenize(a, b, 8, dev) + (a + b,)
    da, db = data._trans(n_cases, 8, dev, g, p=0.55)
    da, db = data._force_msb(da, db, g)
    pw = 10 ** torch.arange(8, device=dev, dtype=torch.long)
    sets["carry-heavy"] = data.tokens(da, db) + ((da * pw).sum(1) + (db * pw).sum(1),)

    ok = True
    for tag, (A, B, truth) in sets.items():
        att = att_map(A, B)
        arg = att.argmax(-1)                                # [N,2,P] attended position
        distinct = len(set(map(tuple, arg[:, 0, 1:9].tolist())))
        skip = float((arg[:, 0, 1:9] != (idx[None, 1:9] - 1)).double().mean())

        live = float((run(A, B, lambda t: t) == truth).double().mean())
        # (a) freeze the map at its batch average -> a fixed, input-independent pattern
        mean_att = att.mean(0, keepdim=True)
        frozen = float((run(A, B, lambda t: mean_att.expand_as(t)) == truth).double().mean())
        # (b) delete the content term: pure recency, i.e. exactly the fixed pattern that a
        #     head "dressed up as attention" would compute
        rec = torch.softmax(
            (model.lam[None, :, None, None] * dist[None, None]).expand(A.shape[0], -1, -1, -1)
            .masked_fill(~mask[None], torch.finfo(torch.float32).min / 4), dim=-1)
        keyless = float((run(A, B, lambda t: rec) == truth).double().mean())
        # (c) give each input another input's map
        perm = torch.randperm(A.shape[0], generator=g, device=dev)
        swapped = float((run(A, B, lambda t: att[perm]) == truth).double().mean())

        print(f"     [{tag}]  {distinct} distinct head-0 maps over {A.shape[0]} inputs; "
              f"{skip:.1%} of queries look past their immediate predecessor")
        print(f"       intact {live:.4f} | frozen at batch mean {frozen:.4f} | "
              f"content term deleted {keyless:.4f} | map swapped {swapped:.4f}")
        ok &= (live >= 0.99 and distinct > 20 and max(frozen, keyless, swapped) < 0.9)

    print(f"   -> attention is load-bearing and input-dependent: {'OK' if ok else 'FAIL'}\n")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="/workspace/submission.py")
    ap.add_argument("--total", type=int, default=1 << 20)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mod = load(args.path)
    model, meta = mod.build_model()
    model = model.to(dev)
    print(f"verifying {args.path} on {dev}\nmetadata: {meta}\n")

    results = {
        "shape": check_shape(mod, model, meta) == 12,
        "heldout": check_heldout(mod, model, dev, total=args.total) >= 0.99,
        "domain": check_domain(model, dev),
        "numerics": check_numerics(mod, model, dev),
        "width": check_width(mod, model, dev),
        "attention": check_attention(mod, model, dev),
    }
    print("summary:", {k: ("pass" if v else "FAIL") for k, v in results.items()})
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
