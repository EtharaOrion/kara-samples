"""Checkpoint surgery for the shrinking cascade.

Two kinds of move:

* exact gauge changes -- rotating / rescaling the residual space, which leave
  the function computed by the block unchanged (verified numerically) but move
  the weights somewhere a smaller parameterisation can hold them;
* a projection onto a smaller config, which is only an initialisation -- the
  rung is then retrained and has to reach the accuracy bar on its own.
"""

import argparse, json

import torch

from model_src import DigitPairAdder, default_cfg

MAT_IN = ("f1_in", "f2_in", "q_w", "k_w", "v_w")     # rows are read directions
MAT_OUT = ("f1_out", "f2_out", "o_w")                # rows are write directions


def full_code(sd, cfg):
    nfix = cfg["code_fix"]
    fixed = torch.zeros(nfix, cfg["C"])
    if nfix >= 2:
        fixed[1, 0] = 1.0
    return torch.cat([fixed, sd["code_free"]], 0)


def _mat(sd, k, D):
    return k in sd and sd[k].dim() >= 2 and sd[k].shape[-1] == D


# ------------------------------------------------------------ exact gauges --

def apply_orth(sd, cfg, R):
    """R: (D, D) orthogonal, identity outside the code subspace."""
    D, C = cfg["D"], cfg["C"]
    out = dict(sd)
    out["code_free"] = sd["code_free"] @ R[:C, :C].T
    for k in MAT_IN + MAT_OUT:
        if _mat(sd, k, D):
            out[k] = sd[k] @ R.T
    return out


def scale_all(sd, cfg, u):
    """x -> u x on every residual axis."""
    assert u > 0
    out = {k: v.clone() for k, v in sd.items()}
    out["code_free"] = sd["code_free"] * u
    for tag in ("f1", "f2"):
        if f"{tag}_b" not in sd:
            continue
        if cfg[f"{tag}_id_in"]:
            out[f"{tag}_b"] = sd[f"{tag}_b"] * u          # hidden units scale by u
        else:
            out[f"{tag}_in"] = sd[f"{tag}_in"] / u         # hidden units unchanged
            out[f"{tag}_out"] = sd[f"{tag}_out"] * u
    if cfg["kv_id"]:
        out["q_b"] = sd["q_b"] / u                         # keys scale by u
    else:
        out["k_w"] = sd["k_w"] / u
        out["v_w"] = sd["v_w"] / u
        out["o_w"] = sd["o_w"] * u                         # values unchanged
    if cfg["q_proj"]:
        out["q_w"] = sd["q_w"] / u
    if "ls" in sd:
        out["ls"] = sd["ls"] - 2 * torch.log(torch.tensor(float(u)))
    return out


def scale_kv(sd, cfg, t):
    """Scale the key/value axis by t: the FFN writes t times as much and the
    query and write weights take up the slack.  Exact only when that axis is
    written by the FFN and read by nothing but the attention -- so it must be
    axis D-1, outside the answer subspace, with keys/values pinned to it and no
    query projection."""
    assert t > 0 and cfg["kv_id"] and cfg["f1_out"] == cfg["D"] - 1
    assert not cfg["q_proj"] and cfg["C"] <= cfg["D"] - 1 and not cfg["U2"]
    out = {k: v.clone() for k, v in sd.items()}
    out["f1_out"] = sd["f1_out"] * t
    out["q_b"] = sd["q_b"] / t
    out["o_w"] = sd["o_w"] / t
    return out


def unit_gauge(sd, cfg):
    """relu(w x + b) = w relu(x + b/w) for w > 0: fold each hidden unit's input
    weight into its bias and output weight, so the input read can be pinned to
    axis 0 with unit weight.  Exact per unit when that unit reads only axis 0
    with a positive weight; the report says how far from that the bank is."""
    out = {k: v.clone() for k, v in sd.items()}
    w = sd["f1_in"]
    w0 = w[:, 0].clone()
    off = w[:, 1:].abs().max().item() if w.shape[1] > 1 else 0.0
    neg = int((w0 <= 0).sum())
    w0 = torch.where(w0.abs() < 1e-6, torch.full_like(w0, 1e-6), w0)
    out["f1_b"] = sd["f1_b"] / w0
    out["f1_out"] = sd["f1_out"] * (w0[:, None] if sd["f1_out"].dim() == 2 else w0)
    return out, f"unit_gauge(neg {neg}/{w.shape[0]}, off-axis {off:.3f})"


def gauge_fix(sd, cfg):
    """Rotate so the prototypes lie along axis 0, ascending.  Only a symmetry
    while nothing pinned reads the answer subspace: axis 0 is pinned open by
    f1_id_in, axis D-1 by kv_id, so require both to be clear of it."""
    D, C = cfg["D"], cfg["C"]
    assert not cfg["f1_id_in"], "axis 0 is pinned; rotating it is not a symmetry"
    assert not cfg["kv_id"] or C <= D - 1, "rotation would move the key axis"
    if C == 1:
        code = full_code(sd, cfg)
        if (code[1, 0] - code[0, 0]) < 0:
            R = torch.eye(D)
            R[0, 0] = -1.0
            sd = apply_orth(sd, cfg, R)
        return sd
    code = full_code(sd, cfg)
    _, _, v = torch.linalg.svd(code, full_matrices=True)
    R = torch.eye(D)
    R[:C, :C] = v
    sd = apply_orth(sd, cfg, R)
    code = full_code(sd, cfg)
    if (code[1, 0] - code[0, 0]) < 0:
        R = torch.eye(D)
        R[0, 0] = -1.0
        sd = apply_orth(sd, cfg, R)
    return sd


def translate(sd, cfg, tvec):
    """code -> code - tvec.  NOT a symmetry: a token embeds as code[a]+code[b],
    so the residual moves by 2t while the readout prototypes move by t, and the
    leftover t has nowhere to go.  The FFN, the queries and the softmax are
    carried exactly; the readout is left off by t for retraining to clean up.
    Only worth doing when t is already small, which it is once the net has
    learnt code[0] ~ 0 (the FFN's absorb value is exactly zero)."""
    out = {k: v.clone() for k, v in sd.items()}
    C, D = cfg["C"], cfg["D"]
    shift = torch.zeros(D)
    shift[:C] = tvec
    out["code_free"] = sd["code_free"] - tvec
    if cfg["f1_id_in"]:
        out["f1_b"] = sd["f1_b"] + 2 * shift[0]
    else:
        out["f1_b"] = sd["f1_b"] + 2 * (sd["f1_in"] @ shift)
    if cfg["q_proj"] and sd["q_b"].shape[0] == sd["q_w"].shape[0]:
        out["q_b"] = sd["q_b"] + 2 * (sd["q_w"] @ shift)
    return out


# ------------------------------------------------------------- equivalence --

@torch.no_grad()
def check_equiv(sd_a, cfg_a, sd_b, cfg_b, n=4096, seed=0):
    import data
    g = torch.Generator().manual_seed(seed)
    da, db, _ = data.batch(n, "cpu", g, train=False,
                           mix=((0.5, None, None), (0.5, 0.05, 0.9)))
    outs = []
    for sd, cfg in ((sd_a, cfg_a), (sd_b, cfg_b)):
        m = DigitPairAdder(cfg).double()
        with torch.no_grad():
            for k, p in m.named_parameters():
                p.copy_(sd[k].double().reshape(p.shape))
        outs.append(m(da, db))
    d = (outs[0] - outs[1]).abs().max().item()
    same = (outs[0].argmax(-1) == outs[1].argmax(-1)).float().mean().item()
    return d, same


# ------------------------------------------------------------------ adapt ---

def adapt(ck, overrides, verbose=True):
    cfg = default_cfg(**ck["cfg"])
    sd = {k: v.clone().float() for k, v in ck["sd"].items()}
    new = default_cfg(**{**cfg, **overrides})
    D = cfg["D"]
    notes = []

    if new["C"] < cfg["C"] or new["code_fix"] > cfg["code_fix"] or new["f1_id_in"]:
        sd = gauge_fix(sd, cfg)
        notes.append("rotate")
    if new["code_fix"] >= 1 > cfg["code_fix"]:
        t = full_code(sd, cfg)[0].clone()
        rng = full_code(sd, cfg).std().item() + 1e-9
        sd = translate(sd, cfg, t)
        notes.append(f"translate(|t|/sd {t.norm().item()/rng:.3f})")
    if new["code_fix"] >= 2 > cfg["code_fix"]:
        u = full_code(sd, cfg)[1, 0].item()
        if u > 0:
            sd = scale_all(sd, cfg, 1.0 / u)
            notes.append("scale")
        else:
            notes.append("scale skipped (code[1] <= 0)")
    if new["f1_id_in"] and not cfg["f1_id_in"]:
        sd, note = unit_gauge(sd, cfg)
        notes.append(note)
    if new["wo_fix"] > cfg["wo_fix"] and cfg["wo_out"] is not None and cfg["kv_id"]:
        t = sd["o_w"][0, 0].item()
        if t > 0:
            sd = scale_kv(sd, cfg, t)
            notes.append("scale_kv")
        else:
            notes.append("scale_kv skipped (head 0 writes negative)")
    if new["kv_id"] and not cfg["kv_id"]:
        sd["q_b"] = sd["q_b"] * sd["k_w"][:, :, D - 1].mean()
        sd["o_w"] = sd["o_w"] * sd["v_w"][:, :, D - 1].mean()
        notes.append("kv_scale")

    ref = DigitPairAdder(new)
    want = dict(ref.named_parameters())
    out = {}
    code = full_code(sd, cfg)[:, :new["C"]]
    for name, p in want.items():
        if name == "code_free":
            out[name] = code[new["code_fix"]:].clone()
            continue
        if name == "q_b":
            q = sd["q_b"]
            out[name] = (q.mean(0, keepdim=True) if p.shape[0] == 1 and q.shape[0] > 1
                         else q[:p.shape[0]]).reshape(p.shape).clone()
            continue
        if name == "o_w":
            o = sd["o_w"]
            if o.dim() == 3 and p.dim() == 2:
                o = o[..., new["wo_out"]]
            out[name] = o[o.shape[0] - p.shape[0]:].reshape(p.shape).clone()
            continue
        if name in ("f1_out", "f2_out"):
            o = sd[name]
            if o.dim() == 2 and p.dim() == 1:
                o = o[:, new[name]]
            out[name] = o[:p.shape[0]].reshape(p.shape).clone()
            continue
        src = sd.get(name)
        if src is None:
            out[name] = p.detach().clone()
            continue
        sl = tuple(slice(0, min(a, b)) for a, b in zip(p.shape, src.shape))
        t = p.detach().clone()
        t[sl] = src[sl]
        out[name] = t
    n = sum(v.numel() for v in out.values())
    if verbose:
        print(f"adapt {ck.get('nparam')} -> {n} params  {overrides}  [{', '.join(notes)}]")
    return {"cfg": new, "sd": out, "acc": -1.0, "nparam": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ck")
    ap.add_argument("out")
    ap.add_argument("--set", default="{}")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ck = torch.load(a.ck, map_location="cpu", weights_only=False)
    new = adapt(ck, json.loads(a.set))
    if a.check:
        d, same = check_equiv(ck["sd"], default_cfg(**ck["cfg"]), new["sd"], new["cfg"])
        print(f"  equivalence: max |dlogit| {d:.3e}, argmax agreement {same:.6f}")
    torch.save(new, a.out)


if __name__ == "__main__":
    main()
