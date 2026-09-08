"""Sanity check that the 16-parameter architecture *can* express exact addition.

This hand-sets weights to prove the target architecture is expressive enough
before spending GPU time on it.  Nothing here is shipped: the submitted weights
come from training only.
"""
import torch

import data
from model_src import DigitPairAdder, default_cfg

CFG = default_cfg(D=2, C=1, U1=3, U2=0, H=2, dh=1, masks=("strict", "causal"),
                  f1_id_in=True, f1_out=1, q_proj=False, share_q=True, kv_id=True,
                  wo_out=0, wo_fix=1, lam=(-12.0, -12.0), code_fix=2,
                  logit_scale=False)

m = DigitPairAdder(CFG)
n = sum(p.numel() for p in m.parameters())
print("parameters:", n, {k: tuple(v.shape) for k, v in m.named_parameters().__iter__().__class__ and dict(m.named_parameters()).items()})

# knees at 8.5, 9.5, 9.75 -> key is 0 on a+b<=8, -M at a+b==9, +1 on a+b>=10
M = 200.0
o1 = -2 * M                      # k(9) = 0.5 * o1
o2 = 4 + 10 * M                  # k(10) = 1
o3 = -o1 - o2                    # flat above 9.75
with torch.no_grad():
    m.code_free.copy_(torch.arange(2, 10, dtype=torch.float32)[:, None])
    m.f1_b.copy_(torch.tensor([-8.5, -9.5, -9.75]))
    m.f1_out.copy_(torch.tensor([o1, o2, o3]))
    m.q_b.copy_(torch.tensor([[1.0]]))
    m.o_w.copy_(torch.tensor([[-10.0]]))

# key function it now implements, per place sum s
x = torch.arange(0, 19.0)[:, None]
k = (torch.relu(x + m.f1_b) * m.f1_out).sum(-1)
print("key(s) for s=0..18:", [round(v, 3) for v in k.tolist()])

dev = "cpu"
g = torch.Generator().manual_seed(0)
for name, mix in (("uniform", ((1.0, None, None),)), ("chains", ((1.0, 0.02, 0.96),))):
    da, db, y = data.batch(20000, dev, g, train=False, mix=mix)
    pred = m(da, db).argmax(-1)[:, 1:]
    print(f"{name}: exact-match {(pred == y).all(-1).float().mean().item():.6f}")
