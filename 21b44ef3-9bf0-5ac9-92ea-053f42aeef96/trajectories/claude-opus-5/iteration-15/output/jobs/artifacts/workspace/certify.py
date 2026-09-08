"""Whole-domain certificate for a shipped submission.

Sampling can only ever say "no counterexample was seen".  This says more,
because the block factors the input in a way that can be enumerated:

  1. the bank sees only  x = code[a] + code[b], so 100 digit pairs cover it;
     each pair must drive both units hard into 0 or 1 (checked with slack), and
     the resulting class must agree with arithmetic: absorbing for a+b <= 8,
     transparent for a+b == 9, generating for a+b >= 10;

  2. the keys and values depend on the input only through those classes, so the
     attention is fully covered by the 3^n class patterns (6561 for n = 8);
     each is evaluated in float64 and compared against the exact carry
     recurrence, giving a bound on how far the attention output can drift;

  3. the read-out then depends only on (a, b, carry-in) at each place, so 100 x
     2 combinations per place cover it; the worst margin over all of them is
     compared against the drift from step 2.

If the worst read-out margin beats the largest possible attention drift, the
model is exact on every 8-digit input -- all 8.1e15 of them.
"""
import argparse, importlib.util, json, math, sys
import torch

A, T, G = 0, 1, 2          # absorbing, transparent, generating


def load(path):
    spec = importlib.util.spec_from_file_location("subm", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def weights(model):
    d = torch.float64
    return dict(code=torch.cat([model.code0, model.code]).to(d),
                knee=model.knee.to(d), fold=model.fold.to(d),
                bank_w=model.bank_w.to(d), key_w=model.key_w.to(d),
                val_w=model.val_w.to(d), carry_w=model.carry_w.to(d),
                lam=model.lam.to(d))


def step1_bank(w):
    """Every reachable digit pair must saturate both bank units, into the class
    arithmetic demands."""
    code = w["code"]
    a = torch.arange(10).repeat_interleave(10)
    b = torch.arange(10).repeat(10)
    x = code[a] + code[b]                                   # [100]
    z = w["bank_w"] * x[:, None] + w["knee"]          # [100,2]
    g = torch.clamp(z, 0.0, 1.0)
    slack = float(torch.maximum(-z, z - 1.0).min())
    binary = bool(((g == 0) | (g == 1)).all())
    cls = (g[:, 0] + g[:, 1]).long()                        # 0,1,2
    want = torch.where(a + b <= 8, torch.full_like(a, A),
                       torch.where(a + b == 9, torch.full_like(a, T),
                                   torch.full_like(a, G)))
    return dict(saturated=binary, slack=slack,
                classes_match_arithmetic=bool((cls == want).all()),
                worst_pair=[int(a[int(torch.maximum(-z, z - 1.0).min(1).values.argmin())]),
                            int(b[int(torch.maximum(-z, z - 1.0).min(1).values.argmin())])])


def _patterns(n):
    """All 3^n class patterns over n places, as an [3^n, n] tensor."""
    idx = torch.arange(3 ** n)
    out = torch.stack([(idx // 3 ** i) % 3 for i in range(n)], 1)
    return out


def step2_attention(w, n):
    """Exact attention over every class pattern vs the carry recurrence."""
    pat = _patterns(n)                                       # [M,n]
    M = pat.shape[0]
    pad = torch.full((M, 1), A)
    cls = torch.cat([pad, pat, pad], 1)                      # [M,P]
    P = cls.shape[1]
    d = torch.float64
    onehot = torch.zeros(M, P, 2, dtype=d)
    onehot[..., 0] = (cls >= T).to(d)                        # g0 = 1[s >= 9]
    onehot[..., 1] = (cls >= G).to(d)                        # g1 = 1[s >= 10]
    k = (onehot * w["key_w"]).sum(-1)                        # [M,P]
    v = (onehot * w["val_w"]).sum(-1)

    p = torch.arange(P)
    dist = (p[:, None] - p[None, :]).to(d)
    logit = k[:, None, :] + w["lam"] * dist
    floor = torch.full_like(logit, -1e300)
    a_in = torch.softmax(torch.where(dist > 0, logit, floor), -1)
    a_out = torch.softmax(torch.where(dist >= 0, logit, floor), -1)
    c_in = (a_in * v[:, None, :]).sum(-1)                    # [M,P]
    c_out = (a_out * v[:, None, :]).sum(-1)

    # the exact carry recurrence for the same class patterns
    ideal_in = torch.zeros(M, P, dtype=d)
    ideal_out = torch.zeros(M, P, dtype=d)
    carry = torch.zeros(M, dtype=d)
    for i in range(P):
        ideal_in[:, i] = carry
        c = cls[:, i]
        carry = torch.where(c == G, torch.ones_like(carry),
                            torch.where(c == T, carry, torch.zeros_like(carry)))
        ideal_out[:, i] = carry
    # position 0 has nothing to attend to in the strict head; it is never read
    dev_in = (c_in - ideal_in)[:, 1:].abs().max()
    dev_out = (c_out - ideal_out)[:, 1:].abs().max()
    return dict(patterns=M, max_carry_in_drift=float(dev_in),
                max_carry_out_drift=float(dev_out),
                drift=float(torch.maximum(dev_in, dev_out)))


def step3_readout(w, drift):
    """Worst read-out margin over every (place digits, carry-in), and whether
    it survives the attention drift."""
    code = w["code"]
    fold = w["fold"][0]
    carry = w["carry_w"][0]
    a = torch.arange(10).repeat_interleave(10)
    b = torch.arange(10).repeat(10)
    rows = []
    for cin in (0, 1):
        s = a + b + cin
        cout = (s >= 10).to(torch.float64)
        y = code[a] + code[b] + carry * cin + fold * cout
        tgt = s % 10
        d2 = (y[:, None] - code[None, :]) ** 2
        good = d2.gather(1, tgt[:, None]).squeeze(1)
        other = d2.scatter(1, tgt[:, None], float("inf")).min(1).values
        rows.append(other - good)
    # the trailing pad position: x = 2*code[0], no carry out, carry-in 0 or 1
    for cin in (0, 1):
        y = 2 * code[0] + carry * cin
        tgt = cin
        d2 = (y - code) ** 2
        good = d2[tgt]
        other = torch.cat([d2[:tgt], d2[tgt + 1:]]).min()
        rows.append((other - good).reshape(1))
    m = torch.cat(rows)
    worst = float(m.min())
    # a drift of eps moves y by at most (|carry| + |fold|) * eps, and moves the
    # margin by at most 2 * max|code_i - code_j| * that
    spanned = float((code.max() - code.min()) * 2)
    shift = float(abs(carry) + abs(fold)) * drift
    return dict(worst_margin=worst, code_gap=float(code[1] - code[0]),
                max_margin_loss=spanned * shift,
                proved_exact=worst > spanned * shift,
                safety_factor=worst / max(spanned * shift, 1e-300))


def certify(path, n=8, verbose=True):
    mod = load(path)
    model, meta = mod.build_model()
    w = weights(model)
    r1 = step1_bank(w)
    r2 = step2_attention(w, n)
    r3 = step3_readout(w, r2["drift"])
    npar = sum(p.numel() for p in model.parameters())
    ok = (r1["saturated"] and r1["classes_match_arithmetic"] and r3["proved_exact"])
    out = dict(path=path, n=n, parameters=npar, bank=r1, attention=r2,
               readout=r3, certified_exact=ok)
    if verbose:
        print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="/workspace/submission.py")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    r = certify(args.path, args.n)
    sys.exit(0 if r["certified_exact"] else 1)
