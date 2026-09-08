"""Inspect a trained member: the learned code, the key/value functions of the
digit sum, the attention it puts on a carry chain, and where it fails."""
import argparse, torch, lib
from check import load

DEV = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--member", type=int, default=0)
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location=DEV, weights_only=False)
    cfg = lib.default_cfg(**ck["cfg"])
    p = {n: t[args.member:args.member + 1].to(DEV) for n, t in ck["params"].items()}
    print("cfg:", cfg)
    print("scores:", [round(s, 4) for s in ck["scores"]])

    code = lib.full_code(p, cfg)[0]
    print("code (digit -> vector):")
    for d in range(10):
        print(f"   {d}: " + " ".join(f"{v:8.3f}" for v in code[d].tolist()))

    # key / value as a function of the local digit sum s = a+b
    pairs = [(a, s - a) for s in range(19) for a in range(10) if 0 <= s - a <= 9]
    a = torch.tensor([[x for x, _ in pairs]], device=DEV)
    b = torch.tensor([[y for _, y in pairs]], device=DEV)
    x = code[a[0]] + code[b[0]]
    h = lib._bank(p, cfg, x[None, None], 1)[0, 0]
    if cfg["kv"] == "share":
        o = p["o_free"][0]
        if cfg["o_pin"]:
            o = torch.cat([torch.ones(1, device=DEV), o])
        z = h @ o
        k, v = p["b_q"][0] * z, z[:, None] * p["w_v"][0]
    else:
        k, v = h @ p["k_o"][0], h @ p["v_o"][0]
    print("\n  s    key      value")
    for s in range(19):
        idx = [i for i, (u, w) in enumerate(pairs) if u + w == s]
        kk = k[idx]
        vv = v[idx]
        print(f" {s:2d}  {kk.mean():9.3f}+-{kk.std():.3f}  " +
              " ".join(f"{c:8.3f}" for c in vv.mean(0).tolist()))
    lam = p["lam"][0].item() if cfg["lam_learn"] else cfg["lam"]
    print("lam:", round(float(lam), 4))

    # attention on a maximal carry chain: 1 + 99999999 style
    m = load(cfg, {n: t[0] for n, t in p.items()}, DEV)
    da = torch.tensor([[0, 5, 4, 4, 4, 2, 4, 4, 4, 0]], device=DEV)
    db = torch.tensor([[0, 5, 5, 5, 5, 3, 5, 5, 5, 0]], device=DEV)
    with torch.no_grad():
        lg, w, kk_ = lib.fwd(p, cfg, da, db, want_attn=True)
    print("\nchain a=", da.tolist()[0], "b=", db.tolist()[0])
    print("digit sums:", (da + db).tolist()[0])
    print("attention argmax per position:", w[0, 0].argmax(-1).tolist())
    print("pred:", lg.argmax(-1)[0, 0].tolist())

    # failure profile
    g = torch.Generator(device=DEV).manual_seed(3)
    a, b = lib.sample_digits(20000, 8, DEV, g)
    y = lib.targets(a, b)
    pa, pb = lib.pad_places(a, b)
    with torch.no_grad():
        pred = lib.fwd(p, cfg, pa, pb).argmax(-1)[0]
    perpos = (pred[:, 1:] == y[:, 1:]).float().mean(0)
    print("\nper-position accuracy:", [round(float(v), 4) for v in perpos])
    s = a + b
    ntr = (s == 9).sum(1)
    for t in range(0, 9):
        m_ = ntr == t
        if m_.sum() > 20:
            acc = (pred[m_][:, 1:] == y[m_][:, 1:]).all(1).float().mean()
            print(f"  {t} transparent places: n={int(m_.sum()):6d} exact={float(acc):.4f}")


if __name__ == "__main__":
    main()
