"""Rank stage-3 members by the quantity the whole-domain certificate actually
turns on: the worst read-out margin over all 100 digit pairs x both carry-in
values, computed in closed form in float64, subject to the clamp bank being
saturated and splitting the pairs by carry class correctly.

Ranking on sampled held-out margin is weaker -- a rare pair can be the tight
one and never appear in the sample.
"""

import argparse
import torch


def score_members(ck, slope=8.0):
    E = ck["fold"].shape[0]
    code = torch.cat([torch.zeros(E, 1), ck["code_free"]], 1).double()   # (E,10)
    carry_w, knee, fold = ck["carry_w"].double(), ck["knee"].double(), ck["fold"].double()

    d = torch.arange(10)
    A, B = torch.meshgrid(d, d, indexing="ij")
    a, b = A.reshape(-1), B.reshape(-1)
    s = (a + b)
    true_cls = torch.where(s < 9, 0, torch.where(s == 9, 1, 2))

    z = code[:, a] + code[:, b]                                          # (E,100)
    e = slope * (z.unsqueeze(-1) - knee[:, None, :])                     # (E,100,2)
    slack = torch.maximum(-e, e - 1.0).amin(dim=(1, 2))                  # (E,)
    u = torch.clamp(e, 0.0, 1.0)
    cls_ok = ((u[..., 0] + u[..., 1]).long() == true_cls[None]).all(1) & \
             ((u == 0) | (u == 1)).all(-1).all(-1)

    cin = torch.tensor([0.0, 1.0]).double()
    ss = s[None, :, None].double() + cin[None, None, :]                  # (1,100,2)
    co = (ss >= 10).double()
    res = z.unsqueeze(-1) + carry_w[:, None, None] * cin[None, None, :] \
        + fold[:, None, None] * co                                       # (E,100,2)
    tgt = (ss % 10).long().expand(res.shape)
    dist = (res.unsqueeze(-1) - code[:, None, None, :]).abs()            # (E,100,2,10)
    dt = dist.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    dw = dist.scatter(-1, tgt.unsqueeze(-1), float("inf")).amin(-1)
    m_min = (dw - dt).amin(dim=(1, 2))
    return m_min, slack, cls_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default="ckpt/stage3.pt")
    args = ap.parse_args()
    ck = torch.load(args.src)
    m_min, slack, cls_ok = score_members(ck)
    ok = cls_ok & (slack > 0)
    print(f"{int(ok.sum())} of {len(m_min)} members have a saturated bank that "
          f"splits the pairs by carry class")
    rank = torch.where(ok, m_min, torch.full_like(m_min, -1e9))
    order = torch.argsort(rank, descending=True)
    for i in order[:5].tolist():
        print(f"  member {i:4d}: worst margin {float(m_min[i]):.6f}  "
              f"bank slack {float(slack[i]):.4f}")
    best = int(order[0])
    print(f"\nbest member index: {best}")
    return best


if __name__ == "__main__":
    main()
