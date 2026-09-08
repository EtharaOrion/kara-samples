"""Print what a checkpoint's block actually computes."""
import argparse

import torch

import cut
import lab
import verify


def show(cfg, sd, tag=""):
    m = cut.model_of(cfg, sd, dev="cpu")
    code = torch.cat([m.code_fix, m.code], 0).detach()
    C, U = cfg["C"], cfg["U"]
    print(f"--- {tag}  cfg C={C} U={U} code_fix={cfg['code_fix']} rb={cfg['rb']} "
          f"params={sum(p.numel() for p in m.parameters())}")
    torch.set_printoptions(precision=4, sci_mode=False, linewidth=160)
    print("code      :", code.t().tolist() if C > 1 else code.view(-1).tolist())
    d = torch.arange(10.0)
    A = torch.stack([torch.ones(10), d], 1)
    sol = torch.linalg.lstsq(A, code).solution
    res = (A @ sol - code).abs().max()
    print(f"code fit  : intercept {sol[0].tolist()}  slope {sol[1].tolist()}  max resid {float(res):.3e}")
    bw = m.bw.detach().view(U, C)
    print("bank bw   :", bw.tolist(), " bb:", m.bb.detach().tolist())
    if C == 1:
        print("bank knees:", (-m.bb.detach() / bw.view(-1)).tolist(), " slopes:", bw.view(-1).tolist())
    print("kw        :", m.kw.detach().view(-1).tolist())
    print("vw        :", m.vw.detach().view(-1).tolist())
    print("e1        :", m.e1.detach().view(-1).tolist(), " e2:", m.e2.detach().view(-1).tolist())
    if cfg["rb"]:
        print("rb        :", m.rb.detach().view(-1).tolist())
    print(f"lam       : {float(m.lam):.4f}   ls: {float(m.ls):.4f}")

    # key / value as a function of the digit-pair sum
    aa, bb_ = torch.meshgrid(torch.arange(10), torch.arange(10), indexing="ij")
    x = code[aa.reshape(-1)] + code[bb_.reshape(-1)]
    u = torch.clamp(x @ bw.t() + m.bb.detach(), 0, 1)
    key, val = u @ m.kw.detach(), u @ m.vw.detach()
    s = (aa + bb_).reshape(-1)
    print("  s : key(min..max)            val(min..max)")
    for sv in range(19):
        k, v = key[s == sv], val[s == sv]
        print(f" {sv:2d} : {float(k.min()):10.4f} {float(k.max()):10.4f}   "
              f"{float(v.min()):8.4f} {float(v.max()):8.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    a = ap.parse_args()
    cfg, sd, note = lab.load_ckpt(a.ckpt)
    show(cfg, sd, tag=f"{a.ckpt} [{note}]")
    m = cut.model_of(cfg, sd, dev=lab.DEV)
    acc_u, _ = lab.model_acc(m, n=8, batches=4, B=16384, mix=(1.0, 0.0, 0.0))
    acc_s, _ = lab.model_acc(m, n=8, batches=4, B=16384)
    print(f"held-out acc: uniform {acc_u:.6f}  structured {acc_s:.6f}")


if __name__ == "__main__":
    main()
