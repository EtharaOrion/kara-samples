"""Whole-domain certificate for the shipped 12-parameter model.

Sampling can only ever say "no counterexample found here".  This proves
correctness on EVERY 8-digit input by decomposing the forward pass:

  A. Bank saturation.  For all 100 digit pairs (and the (0,0) pads) check that
     each clamp pre-activation lies strictly outside [0, 1].  If so, both gate
     values are exactly 0 or 1 for every possible input, so each place falls
     into one of three classes, and the key and value are exactly constant
     within a class.  We then CHECK (not assume) that the classes coincide
     with a+b <= 8 / a+b == 9 / a+b >= 10.

  B. Attention.  With keys and values exactly determined by the class, the
     attention pattern for n=8 depends only on the class pattern, of which
     there are 3^8 = 6561.  Enumerate all of them in float64 and measure how
     far the two heads' outputs deviate from the exact integer carry-in and
     carry-out.

  C. Read-out.  For every digit pair and every carry-in/carry-out combination
     that can actually arise, check the correct prototype is the nearest one
     and record how far the residual could move before that changes.

The model is exact on all 8-digit inputs if the read-out safety radius exceeds
the attention deviation amplified by the write-back weights.
"""
import argparse
import itertools

import torch

import build

torch.set_default_dtype(torch.float64)


def certify(mod, n=8, verbose=True):
    model, _ = mod.build_model()
    sd = {k: v.detach().double() for k, v in model.state_dict().items()}
    code = torch.cat([sd["code0"], sd["code_free"]])         # (10,)
    knee, bank_w = sd["knee"], sd["bank_w"]
    key_w, val_w, val_b = sd["key_w"], sd["val_w"], sd["val_b"]
    carry_w, fold, lam = sd["carry_w"][0], sd["fold"][0], sd["lam"][0]
    rep = {}

    # ---------------------------------------------------------- A. saturation
    da = torch.arange(10).repeat_interleave(10)
    db = torch.arange(10).repeat(10)
    x = code[da] + code[db]                                  # (100,)
    x_all = torch.cat([x, torch.zeros(1)])                   # + the (0,0) pad
    e = bank_w * (x_all[:, None] - knee)                     # (101,2)
    slack = torch.maximum(-e, e - 1.0)
    rep["bank_min_slack"] = float(slack.min())
    ok_sat = bool((slack > 0).all())

    g = (e > 0.5).long()                                     # exact gate values
    s = (da + db)
    want = torch.zeros(101, 2, dtype=torch.long)
    want[:100, 0] = (s >= 9).long()
    want[:100, 1] = (s >= 10).long()
    ok_cls = bool((g == want).all())
    rep["class_map_matches_a+b"] = ok_cls

    # ------------------------------------------------------------ B. attention
    P = n + 2
    idx = torch.arange(P)
    dist = (idx[:, None] - idx[None, :]).double()
    strict = dist > 0
    strict[0, 0] = True
    incl = dist >= 0
    NEG = -1e300

    pats = torch.tensor(list(itertools.product([0, 1, 2], repeat=n)))  # (3^n,n)
    M = pats.shape[0]
    cls = torch.zeros(M, P, dtype=torch.long)                # pads are absorb
    cls[:, 1:n + 1] = pats
    gg = torch.stack([(cls >= 1).double(), (cls >= 2).double()], -1)   # (M,P,2)
    k = gg @ key_w.double()
    v = gg @ val_w.double() + val_b.double()

    logit = k[:, None, :] + lam * dist
    aA = torch.softmax(logit.masked_fill(~strict, NEG), -1)
    aB = torch.softmax(logit.masked_fill(~incl, NEG), -1)
    cA = (aA * v[:, None, :]).sum(-1)                        # (M,P)
    cB = (aB * v[:, None, :]).sum(-1)

    # exact carries implied by the class pattern
    cin = torch.zeros(M, P, dtype=torch.float64)
    cout = torch.zeros(M, P, dtype=torch.float64)
    cur = torch.zeros(M, dtype=torch.float64)
    for p in range(P):
        cin[:, p] = cur
        c = cls[:, p]
        cur = torch.where(c == 2, torch.ones_like(cur),
                          torch.where(c == 1, cur, torch.zeros_like(cur)))
        cout[:, p] = cur
    dev = max(float((cA[:, 1:] - cin[:, 1:]).abs().max()),
              float((cB[:, 1:] - cout[:, 1:]).abs().max()))
    rep["patterns_checked"] = M
    rep["attention_deviation"] = dev
    drift = dev * (abs(float(carry_w)) + abs(float(fold)))
    rep["residual_drift_bound"] = drift

    # ------------------------------------------------------------- C. read-out
    worst = float("inf")
    wrong = 0
    checked = 0
    for ai in range(10):
        for bi in range(10):
            ss = ai + bi
            for ci in (0, 1):
                co = 1 if ss >= 10 else (ci if ss == 9 else 0)
                y = code[ai] + code[bi] + carry_w * ci + fold * co
                t = (ss + ci) % 10
                d = (y - code).abs()
                if int(d.argmin()) != t:
                    wrong += 1
                r = (d[torch.arange(10) != t].min() - d[t]) / 2.0
                worst = min(worst, float(r))
                checked += 1
    rep["readout_cases"] = checked
    rep["readout_wrong"] = wrong
    rep["readout_safety_radius"] = worst

    ok = ok_sat and ok_cls and wrong == 0 and worst > drift
    rep["margin_ratio"] = worst / drift if drift > 0 else float("inf")
    rep["CERTIFIED"] = ok
    if verbose:
        for kk, vv in rep.items():
            print(f"  {kk:26s} {vv}")
    return ok, rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="/workspace/submission.py")
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    mod = build.load_fresh(a.path, "cert_mod")
    ok, _ = certify(mod, a.n)
    print("CERTIFIED" if ok else "NOT CERTIFIED")
