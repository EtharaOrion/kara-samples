"""Structured stress tests for a trained adder.

Beyond random held-out pairs this enumerates the structure the task actually
depends on:

  * every one of the 3^8 carry-structure patterns (each place generates a
    carry / is carry-transparent / does neither), with concrete digits sampled
    inside each pattern;
  * every (place, digit-pair, carry-in) cell of the per-place output table;
  * a large uniform held-out sample.
"""
import argparse
import itertools
import os

import torch

import data

HERE = os.path.dirname(os.path.abspath(__file__))

G, P, N = 0, 1, 2


def pair_tables(device):
    """tables[msb][type] -> (m, 2) tensor of allowed digit pairs."""
    tabs = {}
    for msb in (0, 1):
        lo = 1 if msb else 0
        for t in (G, P, N):
            pairs = []
            for x in range(lo, 10):
                for y in range(lo, 10):
                    s = x + y
                    if (t == G and s >= 10) or (t == P and s == 9) or (t == N and s <= 8):
                        pairs.append((x, y))
            tabs[(msb, t)] = torch.tensor(pairs, dtype=torch.long, device=device)
    return tabs


def sample_typed(types, k, device, gen, tabs):
    """types: (npat, 8) long in {G,P,N}. Returns a, b of shape (npat*k, 8)."""
    npat = types.shape[0]
    a = torch.empty((npat, k, 8), dtype=torch.long, device=device)
    b = torch.empty((npat, k, 8), dtype=torch.long, device=device)
    for place in range(8):
        msb = 1 if place == 7 else 0
        for t in (G, P, N):
            sel = types[:, place] == t
            if not sel.any():
                continue
            tab = tabs[(msb, t)]
            n = int(sel.sum()) * k
            idx = torch.randint(0, tab.shape[0], (n,), device=device, generator=gen)
            pick = tab[idx]
            a[sel, :, place] = pick[:, 0].view(-1, k)
            b[sel, :, place] = pick[:, 1].view(-1, k)
    return a.reshape(-1, 8), b.reshape(-1, 8)


@torch.no_grad()
def predict(model, a, b, chunk=131072):
    outs = []
    for i in range(0, a.shape[0], chunk):
        tok = data.to_tokens(a[i:i + chunk], b[i:i + chunk])
        outs.append(model(tok)[:, 1:, :].argmax(-1))
    return torch.cat(outs, 0)


def run(model, device="cuda", k_pattern=32, n_uniform=4_000_000, n_table=400,
        seed=555, verbose=True):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    tabs = pair_tables(device)
    report = {}

    # ---- every carry-structure pattern ---------------------------------------
    types = torch.tensor(list(itertools.product([G, P, N], repeat=8)),
                         dtype=torch.long, device=device)
    worst = []
    tot_ok = tot_n = 0
    for i in range(0, types.shape[0], 512):
        tt = types[i:i + 512]
        a, b = sample_typed(tt, k_pattern, device, gen, tabs)
        tgt = data.target_digits(a, b)
        ok = (predict(model, a, b) == tgt).all(-1).view(tt.shape[0], k_pattern)
        tot_ok += int(ok.sum())
        tot_n += ok.numel()
        rate = ok.float().mean(1)
        bad = (rate < 1.0).nonzero().flatten()
        for j in bad.tolist():
            worst.append((float(rate[j]), tt[j].tolist()))
    worst.sort()
    report["patterns"] = {"acc": tot_ok / tot_n, "n": tot_n,
                          "failing_patterns": len(worst), "total_patterns": types.shape[0],
                          "worst": worst[:12]}

    # ---- per-place (digit pair, carry-in) table ------------------------------
    cells_bad = []
    cells_missing = 0
    for place in range(8):
        msb = 1 if place == 7 else 0
        lo = 1 if msb else 0
        for x in range(lo, 10):
            for y in range(lo, 10):
                m = n_table * 6
                a = torch.randint(0, 10, (m, 8), device=device, generator=gen)
                b = torch.randint(0, 10, (m, 8), device=device, generator=gen)
                a[:, 7] = torch.randint(1, 10, (m,), device=device, generator=gen)
                b[:, 7] = torch.randint(1, 10, (m,), device=device, generator=gen)
                # bias lower places toward transparency so carry-in=1 is common
                tr = torch.rand((m, 8), device=device, generator=gen) < 0.45
                b = torch.where(tr, 9 - a, b)
                a[:, 7] = torch.where(tr[:, 7], torch.clamp(a[:, 7], 1, 8), a[:, 7])
                b[:, 7] = torch.where(tr[:, 7], 9 - a[:, 7], b[:, 7])
                a[:, place] = x
                b[:, place] = y
                tgt = data.target_digits(a, b)
                # carry into `place`: digit = (x + y + c) % 10 with c in {0,1}
                cin = ((tgt[:, place] - (x + y)) % 10).long()
                pred = predict(model, a, b)
                for c in (0, 1):
                    m2 = cin == c
                    if int(m2.sum()) < 20:
                        cells_missing += 1
                        continue
                    acc = (pred[m2] == tgt[m2]).all(-1).float().mean().item()
                    if acc < 1.0:
                        cells_bad.append((acc, place, x, y, c, int(m2.sum())))
    cells_bad.sort()
    report["table"] = {"failing_cells": len(cells_bad), "unsampled_cells": cells_missing,
                       "worst": cells_bad[:12]}

    # ---- large uniform held-out ---------------------------------------------
    done = 0
    ok = 0
    errs = []
    while done < n_uniform:
        m = min(1 << 20, n_uniform - done) * 100
        a, b = data.sample_transparency(min(1 << 21, m), 0.0, device, gen)
        sel = data.is_heldout(data.digits_to_value(a), data.digits_to_value(b))
        a, b = a[sel], b[sel]
        if a.shape[0] == 0:
            continue
        tgt = data.target_digits(a, b)
        good = (predict(model, a, b) == tgt).all(-1)
        ok += int(good.sum())
        done += a.shape[0]
        if (~good).any() and len(errs) < 12:
            bad = (~good).nonzero().flatten()[:12 - len(errs)]
            for j in bad.tolist():
                av = int(data.digits_to_value(a[j:j + 1])[0])
                bv = int(data.digits_to_value(b[j:j + 1])[0])
                errs.append((av, bv, av + bv))
    report["uniform"] = {"acc": ok / done, "n": done, "errors": errs}

    if verbose:
        r = report
        print("carry-structure patterns : acc %.6f over %d samples; "
              "%d/%d patterns not perfect" % (r["patterns"]["acc"], r["patterns"]["n"],
                                              r["patterns"]["failing_patterns"],
                                              r["patterns"]["total_patterns"]))
        if r["patterns"]["worst"]:
            print("   worst patterns (rate, types G=0/P=1/N=2, LSB first):")
            for rate, t in r["patterns"]["worst"][:6]:
                print("     %.3f %s" % (rate, t))
        print("per-place digit table    : %d failing cells (%d under-sampled)"
              % (r["table"]["failing_cells"], r["table"]["unsampled_cells"]))
        for c in r["table"]["worst"][:6]:
            print("     acc %.3f place %d (%d,%d) carry_in=%d n=%d" % c)
        print("uniform held-out         : acc %.6f over %d pairs" % (r["uniform"]["acc"],
                                                                     r["uniform"]["n"]))
        for e in r["uniform"]["errors"][:8]:
            print("     %d + %d = %d" % e)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--k_pattern", type=int, default=32)
    ap.add_argument("--n_uniform", type=int, default=4000000)
    ap.add_argument("--n_table", type=int, default=400)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    from finetune import load_model
    model, cfg, ck = load_model(a.ckpt, a.device)
    model.eval()
    print("params", sum(p.numel() for p in model.parameters()), cfg)
    run(model, a.device, a.k_pattern, a.n_uniform, a.n_table)
