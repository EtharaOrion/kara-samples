import sys, os, torch
sys.path.insert(0, os.path.dirname(__file__))
from core import Ens, sample_batch, stress_batch, targets, to_onehot, NPOS
import torch.nn.functional as F

ck = torch.load(sys.argv[1], map_location='cpu')
P = ck['params']
M = next(iter(P.values())).shape[0]
W = P['b1a'].shape[1]
m = Ens(M, fold_sign=ck.get('fold_sign'), drop=ck.get('drop') or (), width=W, untied=('V' in P))
with torch.no_grad():
    for k, v in m.named_parameters():
        v.copy_(P[k])
m = m.cuda()
g = torch.Generator(device='cuda').manual_seed(7)
ea, eb = sample_batch(4096, 'cuda', g); et = targets(ea, eb, 'cuda'); eoh = to_onehot(ea, eb, 'cuda')
sa, sb = stress_batch(4096, 'cuda', g); st = targets(sa, sb, 'cuda'); soh = to_onehot(sa, sb, 'cuda')
with torch.no_grad():
    du = (m(eoh).argmax(-1) == et).float().mean(dim=(1, 2))
    ds = (m(soh).argmax(-1) == st).float().mean(dim=(1, 2))
    eu = (m(eoh).argmax(-1) == et).all(-1).float().mean(-1)
k = int((du + ds).argmax()) if len(sys.argv) < 3 else int(sys.argv[2])
print(f'M={M} width={W} best cell {k}: digit uni {du[k]:.4f} str {ds[k]:.4f} exact {eu[k]:.4f}')
print('top5 cells', [(int(i), round(float(du[i]), 4), round(float(ds[i]), 4)) for i in (du + ds).argsort(descending=True)[:5]])

with torch.no_grad():
    U = m.U[k]
    print('U      ', [round(float(x), 3) for x in U])
    if 'V' in P: print('V      ', [round(float(x), 3) for x in m.V[k]]); print('V diffs', [round(float(m.V[k][i+1]-m.V[k][i]), 3) for i in range(9)])
    dU = U - U[0]
    print('U-U[0] ', [round(float(x), 3) for x in dU])
    print('U diffs', [round(float(U[i + 1] - U[i]), 3) for i in range(9)])
    print('slope', float(m.slope[k]), 'bq', float(m.bq[k]), 'wo', [round(float(x), 3) for x in m.wo[k]])
    print('W1a', [round(float(x), 3) for x in m.w1a()[k]], 'b1a', [round(float(x), 3) for x in m.b1a[k]])
    print('W2a0', [round(float(x), 3) for x in m.W2a0[k]], 'W2a1', [round(float(x), 3) for x in m.w2a1()[k]])
    # features as a function of digit sum s (only valid if U is affine)
    zs = torch.stack([U[a] + U[b] for a in range(10) for b in range(10)]).cuda()
    hs = F.relu(zs[:, None] * m.w1a()[k][None] + m.b1a[k][None])
    c1 = (hs * m.W2a0[k]).sum(-1); c2 = (hs * m.w2a1()[k]).sum(-1)
    print('\n s : z(mean,spread)   c1(mean)  c2(mean)')
    for s in range(19):
        idx = [a * 10 + b for a in range(10) for b in range(10) if a + b == s]
        zz = zs[idx]
        print(f'{s:2d} : {zz.mean():7.3f} {zz.max()-zz.min():6.3f}  {c1[idx].mean():8.3f}  {c2[idx].mean():8.3f}')
    # attention pattern on a stress example
    oh = soh[:1]
    z = torch.einsum('bsd,md->mbs', oh, m.U)[k:k + 1]
    h = F.relu(z.unsqueeze(-1) * m.w1a()[k].view(1, 1, 1, -1) + m.b1a[k].view(1, 1, 1, -1))
    cc1 = (h * m.W2a0[k]).sum(-1); cc2 = (h * m.w2a1()[k]).sum(-1)
    q = cc1 + m.bq[k]
    sc = q.unsqueeze(-1) * cc1.unsqueeze(-2) + m.slope[k] * m.rel
    A = torch.softmax(sc + m.mA, -1)[0, 0]
    print('\nexample a', sa[0].tolist(), '\n        b', sb[0].tolist())
    print('digit sums', [int(sa[0, i] + sb[0, i]) for i in range(NPOS)])
    print('argmax attn (head A) per slot:', A.argmax(-1).tolist(), ' maxw', [round(float(x), 2) for x in A.max(-1).values])
