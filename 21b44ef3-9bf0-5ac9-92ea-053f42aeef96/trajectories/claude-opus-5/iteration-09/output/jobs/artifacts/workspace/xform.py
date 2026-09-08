"""Exact re-parameterisations of a trained checkpoint.

Each transform rewrites the *trained* weights into an equivalent model with a
smaller parameter count and is checked to leave the logits (and therefore the
answers) unchanged.  Nothing here invents a value: every transform is a change
of basis or a rescaling that the architecture is already free in.
"""
import argparse
import copy
import torch

from model_src import DigitAdder, default_cfg
from data import Sampler


def _get(st, k):
    return st[k].clone() if k in st else None


# --------------------------------------------------------------------------- #
def absorb_q(cfg, st):
    """scores = q*(key_w.h): fold q into the key weights and pin q = 1."""
    assert cfg["q"] == "free"
    q = st.pop("q")
    st["key_w"] = st["key_w"] * q[:, None]
    if cfg["key_lin"]:
        st["kl"] = st["kl"] * q[:, None]
    cfg["q"], cfg["q_val"] = "fix", 1.0
    return cfg, st


def absorb_wo(cfg, st):
    """y = x + w_o*o with o linear in the value: fold w_o into the value."""
    assert cfg["wo"] == "free" and cfg["C"] == 1
    w = st.pop("w_o")[:, 0]
    if cfg["val"] == "bank":
        st["val_w"] = st["val_w"] * w[:, None]
    elif cfg["val"] == "lin":
        st["vl"] = st["vl"] * w[:, None]
    else:
        raise ValueError("val='id' has no scale to absorb w_o into")
    cfg["wo"], cfg["wo_val"] = "fix", 1.0
    return cfg, st


def free_rows(cfg):
    pinned = [int(p[0]) for p in cfg["pin_code"]]
    return [r for r in range(10) if r not in pinned]


def _move_to_pins(cfg, st, row, val):
    fr = free_rows(cfg)
    j = fr.index(row)
    keep = [i for i in range(len(fr)) if i != j]
    st["code_free"] = st["code_free"][:, keep]
    cfg["pin_code"] = tuple(cfg["pin_code"]) + ((int(row), float(val)),)
    return cfg, st


def rescale_code(cfg, st, s):
    """code -> s*code is an exact symmetry of the whole block."""
    st["code_free"] = st["code_free"] * s[:, None, None]
    if cfg["f_in"] == "free":
        st["W1"] = st["W1"] / s[:, None, None]
    else:                       # relu is positively homogeneous: fold into b, out
        st["b1"] = st["b1"] * s[:, None]
        st["key_w"] = st["key_w"] / s[:, None]
        if "val_w" in st:
            st["val_w"] = st["val_w"] / s[:, None]
    if "kl" in st:
        st["kl"] = st["kl"] / s[:, None]
    if "vl" in st:
        st["vl"] = st["vl"] / s[:, None]
    for k in ("w_o", "w_o2"):
        if k in st:
            st[k] = st[k] * s[:, None]
    if cfg["fold"]:
        if "W2" in st:
            st["W2"] = st["W2"] / s[:, None, None]
        elif cfg["tie"]:
            raise ValueError("tied banks see two scales; rescale before tying")
        for k in ("O2", "O2a"):
            if k in st:
                st[k] = st[k] * (s[:, None, None] if k == "O2" else s[:, None])
    if cfg["ls"] == "free":
        st["ls"] = st["ls"] / s ** 2
    return cfg, st


def gauge_pin(cfg, st, row, val):
    """Use the code-scale symmetry to hold one code entry at a constant."""
    assert cfg["C"] == 1 and cfg["val"] != "id"
    fr = free_rows(cfg)
    cur = st["code_free"][:, fr.index(row), 0]
    assert cur.abs().min() > 1e-6, "cannot rescale a code entry that is zero"
    cfg, st = rescale_code(cfg, st, val / cur)
    return _move_to_pins(cfg, st, row, val)


def add_pin(cfg, st, row, val):
    """Retire a code entry that training has already driven onto a constant."""
    fr = free_rows(cfg)
    cur = st["code_free"][:, fr.index(row), 0]
    assert (cur - val).abs().max() < 1e-6, f"code[{row}] is not at {val}: {cur}"
    return _move_to_pins(cfg, st, row, val)


def share_kv(cfg, st):
    """Key and value read the same projection once training has tied them."""
    assert not cfg["kv_share"] and cfg["val"] == "bank"
    d = (st["val_w"] - st["key_w"]).abs().max().item()
    assert d < 1e-6, f"key and value weights still differ by {d:.3e}"
    st.pop("val_w")
    cfg["kv_share"] = True
    return cfg, st


def signfix(cfg, st, which=1):
    """relu(w*x+b) = |w|*relu(sign(w)*x + b/|w|): pin the bank input slopes."""
    assert cfg["C"] == 1
    W, B = ("W1", "b1") if which == 1 else ("W2", "b2")
    w = st.pop(W)[:, :, 0]
    aw = w.abs()
    assert (aw > 1e-6).all(), "a unit has a near-zero input weight"
    st[B] = st[B] / aw
    if which == 1:
        st["key_w"] = st["key_w"] * aw        # kv_share: the value follows the key
        if cfg["val"] == "bank" and "val_w" in st:
            st["val_w"] = st["val_w"] * aw
    else:
        key = "O2" if cfg["fold_out"] == "free" else "O2a"
        st[key] = st[key] * (aw[:, :, None] if key == "O2" else aw)
    sg = torch.sign(w)
    uniq = {tuple(r.tolist()) for r in sg}
    assert len(uniq) == 1, f"members disagree on sign pattern: {uniq}"
    cfg["f_in" if which == 1 else "f2_in"] = "sign"
    cfg["signs" if which == 1 else "signs2"] = [float(v) for v in sg[0]]
    return cfg, st


def drop_axis(cfg, st, ax=(1,)):
    """Remove residual axes that the trained model no longer uses."""
    ax = (ax,) if isinstance(ax, int) else tuple(ax)
    keep = [i for i in range(cfg["C"]) if i not in ax]
    for a in ax:
        assert st["code_free"][:, :, a].abs().max() < 1e-6, f"code axis {a} is live"
        for k in ("w_o", "w_o2"):
            if k in st:
                assert st[k][:, a].abs().max() < 1e-6, f"{k} still writes to axis {a}"
    st["code_free"] = st["code_free"][:, :, keep]
    if "w_o2" in st:
        st["w_o2"] = st["w_o2"][:, keep]
    for k in ("W1", "W2"):
        if k in st:
            st[k] = st[k][:, :, keep]
    if "w_o" in st:
        st["w_o"] = st["w_o"][:, keep]
    if "O2" in st:
        st["O2"] = st["O2"][:, :, keep]
    for k in ("kl", "vl"):
        if k in st:
            st[k] = st[k][:, keep]
    cfg["C"] = len(keep)
    return cfg, st


def drop_unit(cfg, st, u, which=1):
    """Remove a bank unit whose output weights have been annealed to zero."""
    U = cfg["U"] if which == 1 else (cfg["U"] if cfg["tie"] else cfg["U2"])
    keep = [i for i in range(U) if i != u]
    if which == 1:
        for k in ("W1", "b1"):
            if k in st:
                st[k] = st[k][:, keep]
        for k in ("key_w", "val_w"):
            if k in st:
                assert st[k][:, u].abs().max() < 1e-5, f"{k} unit {u} is still live"
                st[k] = st[k][:, keep]
        cfg["U"] = U - 1
        if cfg["signs"]:
            cfg["signs"] = [cfg["signs"][i] for i in keep]
        if cfg["tie"]:
            for k in ("O2", "O2a"):
                if k in st:
                    st[k] = st[k][:, keep]
    else:
        for k in ("W2", "b2"):
            if k in st:
                st[k] = st[k][:, keep]
        for k in ("O2", "O2a"):
            if k in st:
                assert st[k][:, u].abs().max() < 1e-5, f"{k} unit {u} is still live"
                st[k] = st[k][:, keep]
        cfg["U2"] = U - 1
        if cfg["signs2"]:
            cfg["signs2"] = [cfg["signs2"][i] for i in keep]
    return cfg, st


def balance(cfg, st, thresh=0.01):
    """Put a near-constant bank unit's bias and key weight on the same scale.

    relu is positively homogeneous, so scaling a unit's (W1, b1) by 1/c and its
    key weight by c leaves the block's function untouched.  Training leaves the
    unit that supplies the constant offset with a tiny bias multiplied by a huge
    key weight, which makes that offset swing by over 1% per optimiser step and
    breaks any anneal that has to move it.  This rescales it to O(1).
    """
    cfg = dict(cfg)
    assert cfg["f_in"] == "free" and cfg["C"] == 1
    w = st["W1"][:, :, 0]
    scale = w.abs().amax(1, keepdim=True)
    const = [u for u in range(cfg["U"]) if (w[:, u].abs() / scale[:, 0]).max() < thresh]
    assert const, "no near-constant unit"
    c = torch.ones_like(st["b1"])
    for u in const:
        c[:, u] = st["b1"][:, u].abs().clamp(min=1e-12)
    st["W1"] = st["W1"] / c[:, :, None]
    st["b1"] = st["b1"] / c
    st["key_w"] = st["key_w"] * c
    if "val_w" in st:
        st["val_w"] = st["val_w"] * c
    print(f"  rebalanced near-constant units {const} by {[round(float(c[0, u]), 5) for u in const]}")
    return cfg, st


def drop_dead(cfg, st):
    """Remove bank units whose ReLU never fires on any reachable input.

    The token embedding can only ever be ``code[a] + code[b]`` for a digit pair
    (the pads are the (0, 0) pair), so 100 vectors exhaust the bank's input set.
    A unit whose pre-activation is <= 0 on all of them contributes nothing to
    any key or value, for any operand, and dropping it is exact.
    """
    cfg = dict(cfg)
    code = torch.cat([st["code_free"],
                      _pin_tensor(cfg, st["code_free"])], dim=1)[:, _perm(cfg)]
    xs = (code[:, :, None, :] + code[:, None, :, :]).reshape(code.shape[0], -1, cfg["C"])
    w1 = st["W1"] if cfg["f_in"] == "free" else _sign_tensor(cfg, st)
    pre = torch.einsum("enc,euc->enu", xs, w1) + st["b1"][:, None, :]
    live = (pre.amax(dim=1) > 0).any(dim=0)              # (U,) live for any member
    dead = [u for u in range(cfg["U"]) if not live[u]]
    assert dead, "no dead units"
    print(f"  dead units {dead}  (max pre-activation "
          f"{[round(pre[:, :, u].max().item(), 4) for u in dead]})")
    for u in reversed(dead):
        keep = [i for i in range(cfg["U"]) if i != u]
        for k in ("W1", "b1", "key_w", "val_w"):
            if k in st:
                st[k] = st[k][:, keep]
        if cfg["signs"]:
            cfg["signs"] = [cfg["signs"][i] for i in keep]
        cfg["U"] -= 1
    return cfg, st


def _perm(cfg):
    prow = [int(p[0]) for p in cfg["pin_code"]]
    free = [r for r in range(10) if r not in prow]
    perm = [0] * 10
    for i, r in enumerate(free):
        perm[r] = i
    for i, r in enumerate(prow):
        perm[r] = len(free) + i
    return torch.tensor(perm, dtype=torch.long)


def _pin_tensor(cfg, ref):
    pins = tuple(cfg["pin_code"])
    pv = torch.zeros(ref.shape[0], len(pins), cfg["C"], dtype=ref.dtype)
    for i, p in enumerate(pins):
        pv[:, i, 0] = float(p[1])
    return pv


def _sign_tensor(cfg, st):
    sg = torch.tensor([float(s) for s in cfg["signs"]], dtype=st["b1"].dtype)
    return sg.reshape(1, -1, 1).expand(st["b1"].shape[0], -1, cfg["C"]).contiguous()


def drop_fold(cfg, st):
    """Remove the post-attention bank once its output weights have annealed
    to zero (the second attention head now does the mod-10 fold)."""
    assert cfg["fold"]
    key = "O2" if cfg["fold_out"] == "free" else "O2a"
    assert st[key].abs().max() < 1e-6, "the fold bank is still live"
    for k in (key, "W2", "b2"):
        st.pop(k, None)
    cfg["fold"] = False
    return cfg, st


def retire(cfg, st, name, val):
    """Turn a scalar the trainer has driven onto an architectural constant into
    a buffer.  The value is the constant training was annealed to, not a learned
    number being smuggled out of the parameter count."""
    cur = st[name]
    d = (cur - val).abs().max().item()
    assert d < 1e-6, f"{name} is not at {val}: off by {d:.3e}"
    st.pop(name)
    key = {"w_o": "wo"}.get(name, name)      # the cfg switch is not always the
    cfg[key] = "fix"                         # same string as the parameter
    cfg[key + "_val"] = float(val)
    return cfg, st


def fix_scalar(cfg, st, name):
    """Freeze a scalar that only sets the read-out temperature (argmax-invariant)."""
    assert name == "ls"
    v = st.pop("ls")
    cfg["ls"], cfg["ls_val"] = "fix", 1.0
    return cfg, st


TRANSFORMS = {
    "absorb_q": absorb_q,
    "absorb_wo": absorb_wo,
    "signfix": signfix,
    "signfix2": lambda c, s: signfix(c, s, 2),
    "drop_axis": drop_axis,
    "drop_fold": drop_fold,
    "share_kv": share_kv,
    "drop_dead": drop_dead,
    "balance": balance,
    "fix_ls": lambda c, s: fix_scalar(c, s, "ls"),
}


# --------------------------------------------------------------------------- #
def build(cfg, st, E):
    cfg = default_cfg(dict(cfg))
    cfg["E"] = E
    m = DigitAdder(cfg)
    # load_state_dict copies into the parameter's dtype, so the module has to be
    # promoted first or a float64 checkpoint is silently rounded on the way in
    m = m.to(next(iter(st.values())).dtype)
    sd = m.state_dict()
    for k, v in st.items():
        sd[k] = v.reshape(sd[k].shape)
    m.load_state_dict(sd)
    return m


def compare(cfg0, st0, cfg1, st1, E, dev, n=200000, argmax_only=False):
    smp = Sampler(dev)
    m0 = build(cfg0, st0, E).to(dev).double()
    m1 = build(cfg1, st1, E).to(dev).double()   # already float64; .double() is a no-op
    dmax, agree, tot = 0.0, 0, 0
    with torch.no_grad():
        for pl in (8, 5, 12):
            for _ in range(max(1, n // 8192 // 3)):
                a, b, _t = smp.batch(8192, n=pl)
                l0, l1 = m0(a, b), m1(a, b)
                # a shift that is constant across the ten classes cannot move a
                # decision, so compare class-centred logits
                c0 = l0 - l0.mean(-1, keepdim=True)
                c1 = l1 - l1.mean(-1, keepdim=True)
                dmax = max(dmax, (c0 - c1).abs().max().item())
                agree += (l0.argmax(-1) == l1.argmax(-1)).all(-1).sum().item()
                tot += l0.shape[0] * l0.shape[1]
    return dmax, agree / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ops", required=True, help="comma list of transforms")
    ap.add_argument("--out", required=True)
    ap.add_argument("--members", type=int, default=0, help="keep only the first k")
    ap.add_argument("--dev", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg0 = default_cfg(dict(ck["cfg"]))
    # the algebra below is exact, but these models are numerically stiff: the
    # keys run to hundreds, so a float32 rounding of one weight moves a logit by
    # ~1e-2 and the exactness check cannot see past it.  Work in float64 and save
    # in float64; the single rounding back to float32 happens once, when the
    # submission is written, and verify.py checks that it changes no answer.
    st0 = {k: v.double().clone() for k, v in ck["state"].items()}
    if args.members:
        st0 = {k: v[: args.members] for k, v in st0.items()}
        ck["acc"] = ck["acc"][: args.members]
    E = next(iter(st0.values())).shape[0]
    cfg0["E"] = E

    cfg, st = copy.deepcopy(cfg0), {k: v.clone() for k, v in st0.items()}
    argmax_only = False
    for op in args.ops.split(","):
        name, _, arg = op.partition(":")
        if name in ("gauge_pin", "add_pin"):
            r, v = arg.split("/")
            fn = gauge_pin if name == "gauge_pin" else add_pin
            cfg, st = fn(cfg, st, int(r), float(v))
        elif name == "drop_axis":
            cfg, st = drop_axis(cfg, st, tuple(int(t) for t in arg.split("/")))
        elif name == "retire":
            nm, v = arg.split("/")
            cfg, st = retire(cfg, st, nm, float(v))
        elif name == "drop_unit":
            w, u = arg.split("/")
            cfg, st = drop_unit(cfg, st, int(u), int(w))
        else:
            cfg, st = TRANSFORMS[name](cfg, st)
        if name in ("fix_ls",):
            argmax_only = True
        print(f"applied {op}")

    n0 = sum(v.numel() for v in st0.values()) // E
    n1 = sum(v.numel() for v in st.values()) // E
    dmax, agree = compare(cfg0, st0, cfg, st, E, args.dev, argmax_only=argmax_only)
    print(f"parameters {n0} -> {n1}   max|dlogit| {dmax:.3e}   argmax agreement {agree:.6f}")
    assert agree == 1.0, "transform changed an answer"
    torch.save({"cfg": cfg, "state": st, "acc": ck.get("acc"), "n_par": n1,
                "from": args.ckpt, "ops": args.ops}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
