"""Assert the E-batched training forward and the shipped nn.Module agree."""
import torch, lib
from model_src import DigitPairAdder


def member(ck, i=0):
    return {n: t[i].clone() for n, t in ck["params"].items()}


def load(cfg, pm, device="cpu"):
    m = DigitPairAdder(cfg).to(device)
    sd = m.state_dict()
    for n, t in pm.items():
        sd[n] = t.to(device).to(sd[n].dtype).reshape(sd[n].shape)
    m.load_state_dict(sd)
    return m.eval()


def main():
    torch.manual_seed(0)
    cfgs = [
        dict(),
        dict(C=1, U=3, U2=2),
        dict(C=2, U=5, U2=5, kv='split', logit_scale=True),
        dict(C=2, U=6, U2=6, kv="split"),
        dict(C=1, U=4, U2=3, kv="split", f1_in="sign", f1_sign=-1.0),
        dict(C=1, U=3, U2=2, o_pin=True, pin_row=1, lam_learn=False),
        dict(C=1, U=3, U2=3, tie=True, f1_in="sign", f1_sign=1.0,
             o_pin=True, pin_row=3, lam_learn=False),
        dict(C=2, U=4, U2=3, f1_in="sign", f1_sign=1.0,
             f2_in="sign", f2_sign=-1.0, p_rows=[0, 2]),
        dict(C=1, U=3, U2=2, f1_in="sign", f1_sign=1.0, f2_in="sign", f2_sign=-1.0,
             o_pin=True, pin_row=0, y_bias=True),
        dict(C=1, U=3, U2=2, o_pin=True, pin_row=0, y_bias=True, logit_scale=True),
        dict(C=1, U=3, U2=2, kv="one", f1_in="sign", f1_sign=-1.0,
             f2_in="sign", f2_sign=1.0, o_pin=True, pin_row=0, lam_learn=False),
        dict(C=1, U=3, U2=3, kv="one", tie=True, f1_in="sign", f1_sign=-1.0,
             o_pin=True, pin_row=0, lam_learn=False, p_rows=[0, 2]),
        dict(C=2, U=4, U2=4, kv="one", tie=True),
    ]
    for over in cfgs:
        cfg = lib.default_cfg(**over)
        p = lib.init_params(cfg, 3, "cpu", seed=1)
        m = load(cfg, {n: t[1] for n, t in p.items()})
        a = torch.randint(0, 10, (7, 10))
        b = torch.randint(0, 10, (7, 10))
        with torch.no_grad():
            l1 = lib.fwd(p, cfg, a, b)[1]
            l2 = m(a, b)
        d = ((l1 - l2).abs().max() / l1.abs().max()).item()
        agree = (l1.argmax(-1) == l2.argmax(-1)).float().mean().item()
        print(f"n_params={lib.n_params(cfg):3d}  rel_maxdiff={d:.2e}  argmax_agree={agree}  {over}")
        assert d < 1e-5 and agree == 1.0, over
    print("OK")


if __name__ == "__main__":
    main()
