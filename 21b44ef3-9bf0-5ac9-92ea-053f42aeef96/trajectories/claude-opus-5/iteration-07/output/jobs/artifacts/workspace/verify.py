"""Audit a candidate submission.py: accuracy, margins, and attention ablations."""
import argparse, importlib.util, os, sys
import torch

import data
import probe

torch.backends.cuda.matmul.allow_tf32 = False


def load(path):
    spec = importlib.util.spec_from_file_location("_sub", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def relength(mod, model, P, dev):
    """Same trained weights, run over a different number of positions."""
    m = mod.DigitPairAdder(dict(model.cfg, P=int(P)))
    with torch.no_grad():
        for (k, p), (_, q) in zip(m.named_parameters(), model.named_parameters()):
            p.copy_(q.detach().cpu())
    return m.to(dev).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="submission.py")
    ap.add_argument("--n", type=int, default=1000000)
    ap.add_argument("--dev", default="cuda")
    args = ap.parse_args()
    dev = args.dev if torch.cuda.is_available() else "cpu"

    mod = load(args.sub)
    model, meta = mod.build_model()
    npar = sum(p.numel() for p in model.parameters())
    nbuf = sum(b.numel() for b in model.buffers())
    print(f"submission: {args.sub}")
    print(f"parameters : {npar}   (metadata claims {meta.get('n_parameters')})")
    print(f"buffers    : {nbuf} elements (fixed structure, not learned)")
    print("buffer values:",
          {n: b.flatten().tolist()[:6] for n, b in model.named_buffers()})
    model = model.to(dev).eval()

    g = torch.Generator(device=dev).manual_seed(20260904)
    rows = []
    tok, tgt = data.sample(args.n, dev, g, mix=(1.0, 0.0, 0.0), split="eval")
    rows.append(("held-out uniform (bucket 0)", tok.shape[0],
                 probe.batched_acc(model, tok, tgt)))
    tok2, tgt2 = data.sample(args.n // 2, dev, g, mix=(0., 1., 0.), split="eval")
    rows.append(("held-out transparent-enriched", tok2.shape[0],
                 probe.batched_acc(model, tok2, tgt2)))
    tok3, tgt3 = data.sample(args.n // 2, dev, g, mix=(0., 0., 1.), split="eval")
    rows.append(("held-out maximal carry chains", tok3.shape[0],
                 probe.batched_acc(model, tok3, tgt3)))
    tok4, tgt4 = data.all_carry_patterns(dev)
    rows.append(("all 6561 carry patterns", tok4.shape[0],
                 probe.batched_acc(model, tok4, tgt4)))
    tok5, tgt5 = data.edge_cases(dev)
    rows.append(("edge cases", tok5.shape[0], probe.batched_acc(model, tok5, tgt5)))
    tokA, tgtA = data.sample(args.n, dev, g, split="all")
    rows.append(("mixed, whole space", tokA.shape[0],
                 probe.batched_acc(model, tokA, tgtA)))
    print("\naccuracy")
    for name, n, a in rows:
        print(f"  {name:32s} n={n:8d}  {a:.6f}")

    # --- add() path agrees with the batched forward ------------------------
    cm = model.to("cpu")
    sub = tok4[:400].cpu()
    aa = data.to_int(sub[:, 1:9, 0]).tolist()
    bb = data.to_int(sub[:, 1:9, 1]).tolist()
    bad = [(x, y) for x, y in zip(aa, bb) if mod.add(cm, x, y) != x + y]
    print(f"\nadd() on 400 carry patterns: {len(bad)} wrong  {bad[:3]}")
    tokE = tok5.cpu()
    ae = data.to_int(tokE[:, 1:9, 0]).tolist()
    be = data.to_int(tokE[:, 1:9, 1]).tolist()
    badE = [(x, y, mod.add(cm, x, y)) for x, y in zip(ae, be)
            if mod.add(cm, x, y) != x + y]
    print(f"add() on edge cases: {len(badE)} wrong  {badE[:3]}")
    model = model.to(dev)

    # --- decision margin ---------------------------------------------------
    with torch.no_grad():
        t2 = model(tok4)[:, 1:].topk(2, dim=-1).values
        m1 = float((t2[..., 0] - t2[..., 1]).min())
        t2 = model(tok[:200000])[:, 1:].topk(2, dim=-1).values
        m2 = float((t2[..., 0] - t2[..., 1]).min())
    print(f"\nmin top-1/top-2 logit gap: carry patterns {m1:.5f}  uniform {m2:.5f}")

    # --- float64 / device agreement ----------------------------------------
    md = mod.build_model()[0].double().to(dev).eval()
    with torch.no_grad():
        p32 = model(tok4).argmax(-1)
        p64 = md(tok4).argmax(-1)
    print(f"float32 vs float64 argmax agreement (carry patterns): "
          f"{float((p32 == p64).float().mean()):.6f}")
    mc = mod.build_model()[0].eval()
    with torch.no_grad():
        pc = mc(tok4.cpu()).argmax(-1)
    print(f"cuda vs cpu argmax agreement: "
          f"{float((p32.cpu() == pc).float().mean()):.6f}")

    # --- is the attention doing real work? ---------------------------------
    print(f"\nattention  (mirror error {probe.check_mirror(model, tok4[:512]):.2e})")
    A = probe.attn_maps(model, tok4)
    pat = A.argmax(-1)[:, 1:]
    print(f"  {len(set(map(tuple, pat[:4096].tolist())))} distinct argmax "
          f"patterns over 4096 inputs")
    print(f"  per-position argmax variation: "
          f"{[int(len(set(pat[:, i].tolist()))) for i in range(pat.shape[1])]}")
    fixed = A.mean(0, keepdim=True)
    print(f"  frozen at its batch mean : carry patterns "
          f"{probe.frozen_acc(model, tok4, tgt4, fixed):.4f}, uniform "
          f"{probe.frozen_acc(model, tok[:200000], tgt[:200000], fixed):.4f}")
    onehot = torch.zeros_like(A).scatter_(-1, A.argmax(-1).unsqueeze(-1), 1.0)
    print(f"  hardened to its own argmax: "
          f"{probe.frozen_acc(model, tok4, tgt4, onehot):.4f}")

    # --- length generalisation ---------------------------------------------
    # A fixed distance-decay pattern cannot move a carry across a long run of
    # transparent places, so its chain accuracy falls off with length; a real
    # position-selecting attention is length-invariant.
    print("\nlength generalisation (same weights, more positions)")
    gl = torch.Generator(device=dev).manual_seed(31337)
    for npl in (8, 10, 12, 14, 16, 20):
        ml = relength(mod, mod.build_model()[0], npl + 2, dev)
        tc, gc = data.sample(50000, dev, gl, mix=(0., 0., 1.), split="all",
                             nplace=npl)
        tu, gu = data.sample(50000, dev, gl, mix=(1., 0., 0.), split="all",
                             nplace=npl)
        print(f"  {npl:3d} places   chains {probe.batched_acc(ml, tc, gc):.4f}"
              f"   uniform {probe.batched_acc(ml, tu, gu):.4f}")


if __name__ == "__main__":
    main()
