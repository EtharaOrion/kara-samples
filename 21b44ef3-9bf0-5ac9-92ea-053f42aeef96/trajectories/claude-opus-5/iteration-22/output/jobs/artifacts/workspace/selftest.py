"""Check that the ensemble mirror in lab.py computes exactly the same function
as the nn.Module in model_src.py, and that the data generator is sane."""

import torch
import lab
import model_src


def main():
    torch.manual_seed(0)
    E = 5
    p = lab.init_params(E, seed=3)
    p["code_free"][:, 0] = 1.0                       # shipped gauge: code[1] = 1

    da, db, tgt = lab.sample(8, 64, split="train")
    ens = lab.fwd(p, da, db, ship_consts=True)

    worst = 0.0
    for e in range(E):
        m = model_src.DigitPairAdder()
        with torch.no_grad():
            m.code_free.copy_(p["code_free"][e, 1:].cpu())
            m.carry_w.copy_(p["carry_w"][e].cpu())
            m.knee.copy_(p["knee"][e].cpu())
            m.fold.copy_(p["fold"][e].cpu())
            ref = m(da.cpu(), db.cpu())
        worst = max(worst, (ref - ens[e].cpu()).abs().max().item())
    print(f"mirror vs module: max abs logit difference {worst:.3e}")
    assert worst < 1e-4, worst

    # data sanity: answers really are the sums
    a, b, t = lab.sample(8, 4096, split="train")
    powers = 10 ** torch.arange(8, device=a.device)
    av = (a * powers).sum(-1)
    bv = (b * powers).sum(-1)
    tv = (t * (10 ** torch.arange(9, device=a.device))).sum(-1)
    assert bool((av + bv == tv).all())
    print("answer() matches integer addition on 4096 samples")

    tr_a, tr_b, _ = lab.sample(8, 20000, split="train")
    ho_a, ho_b, _ = lab.sample(8, 20000, split="heldout")
    assert int((lab.hash_bucket(tr_a, tr_b) == 0).sum()) == 0
    assert int((lab.hash_bucket(ho_a, ho_b) != 0).sum()) == 0
    tr_keys = set(((tr_a * powers).sum(-1) * 100000000 + (tr_b * powers).sum(-1)).tolist())
    ho_keys = set(((ho_a * powers).sum(-1) * 100000000 + (ho_b * powers).sum(-1)).tolist())
    print(f"train/held-out key overlap: {len(tr_keys & ho_keys)} (of "
          f"{len(tr_keys)} / {len(ho_keys)} distinct)")
    assert len(tr_keys & ho_keys) == 0

    cls = lab.CLASS_OF[(a[:, 0] * 10 + b[:, 0])]
    print("place-0 carry-class mix (absorb/transparent/generate):",
          [round(float((cls == k).float().mean()), 3) for k in range(3)])
    print("selftest OK")


if __name__ == "__main__":
    main()
