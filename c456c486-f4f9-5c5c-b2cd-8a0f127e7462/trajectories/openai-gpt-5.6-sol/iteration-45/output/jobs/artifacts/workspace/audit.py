import copy, random, sys, torch
sys.path.insert(0,'/workspace')
import submission
m,md=submission.build_model(); print('metadata',md); print('params',sum(p.numel() for p in m.parameters()))
# Public inference path edge suite.
cases={(0,0),(1,0),(99999999999999,0),(99999999999999,1),(99999999999999,99999999999999)}
for k in range(14):
 p=10**k
 for length in range(1,15-k):
  run=(10**length-1)*p
  cases.add((run,p)); cases.add((max(0,run-p),p)); cases.add((5*p,5*p)); cases.add((9*p,9*p))
for a in [0,1,5,9,10,99,9999999999999,99999999999999]:
 for b in [0,1,5,9,10,10**13,99999999999999]: cases.add((a,b))
failed=[]
for a,b in cases:
 got=submission.add(m,a,b)
 if got!=a+b: failed.append((a,b,got,a+b))
print('systematic',len(cases),'failures',len(failed),failed[:20])
# Attention content dependence from first block's actual normalized Q/K projection.
def qk(a,b):
 ad=torch.tensor([[int(c) for c in f'{a:014d}'[::-1]]+[0]]); bd=torch.tensor([[int(c) for c in f'{b:014d}'[::-1]]+[0]])
 x=m.a_embed(ad)+m.b_embed(bd)+torch.nn.functional.pad(m.pos[:15],(0,7)); z=m.block0.norm(x); q,k,_=m.block0.qkv(z).chunk(3,-1); return q@k.transpose(-1,-2)
delta=(qk(12345678901234,98765432109876)-qk(11111111111111,22222222222222)).abs().max().item(); print('qk_delta',delta)
# Output depends on weights: zeroing head changes predictions/logits.
ad=torch.tensor([[4,3,2,1]+[0]*11]); bd=torch.tensor([[6,6,6,6]+[0]*11]); prev=torch.empty(1,0,dtype=torch.long); normal=m(ad,bd,prev)[:,-1].argmax().item(); saved=m.head.weight.detach().clone(); m.head.weight.data.zero_(); changed=m(ad,bd,prev)[:,-1].argmax().item(); print('head_perturbation',normal,changed,normal!=changed)
