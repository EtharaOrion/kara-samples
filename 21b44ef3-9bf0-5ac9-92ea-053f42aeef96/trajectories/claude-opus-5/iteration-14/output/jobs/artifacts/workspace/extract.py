"""Read the shipped-form quantities off a trained stage-B member.

The shipped form has increasing clamp units and a fixed sign convention, but a
trained member can express the same bank with either slope sign and either unit
order.  Rather than pattern-match on weights, this measures the member's own
behaviour: for each bank unit it evaluates u as a function of the place sum s and
records where that unit crosses 1/2, in units of the member's own code step.  Those
crossings, and the code, are what stage C starts from -- both are measured from the
trained model, not chosen.
"""
import torch, arch


@torch.no_grad()
def read_off(P, cfg, device="cuda"):
    """P: ensemble dict.  -> dict of (E,...) tensors describing each member."""
    E = P["code"].shape[0]
    code = arch.full_code(P, cfg)[:, :, 0].double()                   # (E,10)
    d = torch.arange(10, device=code.device, dtype=torch.float64)
    sig = (code * d).sum(-1) / (d * d).sum()                          # (E,) code step
    s = torch.arange(19, device=code.device, dtype=torch.float64)     # place sums
    x = sig[:, None] * s[None, :]                                     # (E,19) ideal residual
    U = cfg["U"]
    z = x[:, :, None] * P["Bw"][:, 0, :].double()[:, None, :] + P["bb"].double()[:, None, :]
    u = z.clamp(0, 1)                                                 # (E,19,U)
    rising = (u[:, -1, :] - u[:, 0, :]) >= 0                          # (E,U)
    ur = torch.where(rising[:, None, :], u, 1.0 - u)                  # as an increasing unit
    # first s where the increasing form is at least 1/2, linearly interpolated
    ge = ur >= 0.5
    idx = torch.where(ge.any(1), ge.double().argmax(1), torch.full_like(s[:1].long(), 18))
    idx = idx.clamp(min=1)
    lo = torch.gather(ur, 1, (idx - 1)[:, None, :]).squeeze(1)
    hi = torch.gather(ur, 1, idx[:, None, :]).squeeze(1)
    frac = ((0.5 - lo) / (hi - lo).where((hi - lo).abs() > 1e-12,
                                         torch.ones_like(hi))).clamp(0, 1)
    cross = idx.double() - 1.0 + frac                                 # (E,U) knee in s units
    cross = torch.where(ge.any(1), cross, torch.full_like(cross, 99.0))
    knee, order = torch.sort(cross, dim=1)
    return dict(code_unit=(code[:, 1:] / sig[:, None])[:, :, None].float(),   # (E,9,1) step 1
                knee=knee.float(), sigma=sig.float(), rising=rising, cross=cross.float())


if __name__ == "__main__":
    import sys
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    st = torch.load(sys.argv[1], map_location=dev)
    P = {k: v.to(dev) for k, v in st["p"].items()}
    r = read_off(P, st["cfg"], dev)
    k = r["knee"]
    good = ((k[:, 0] > 8) & (k[:, 0] < 9) & (k[:, 1] > 9) & (k[:, 1] < 10))
    print(f"{P['code'].shape[0]} members; {int(good.sum())} have one crossing in (8,9) "
          f"and one in (9,10) -- the topology the shipped form needs")
    for i in range(min(12, k.shape[0])):
        print(f"  #{i:2d} sigma {float(r['sigma'][i]):+.4f} crossings {k[i].tolist()} "
              f"{'OK' if bool(good[i]) else ''}")
