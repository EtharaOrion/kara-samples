import importlib.util, random, torch, math
spec=importlib.util.spec_from_file_location('submission','/workspace/submission.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
m,meta=s.build_model(); print(meta); print('params',sum(p.numel() for p in m.parameters()))
cases=[(0,0),(1,0),(0,1),(99999999999999,0),(99999999999999,1),(99999999999999,99999999999999),(50000000000000,50000000000000),(99999999999990,9),(99999999999990,10)]
for k in range(14):
 p=10**k; cases += [(p-1,1),(p, p),(5*p,5*p),(9*p,9*p),(99999999999999,p)]
random.seed(500); cases += [(random.randrange(10**14),random.randrange(10**14)) for _ in range(2000)]
bad=[]
for a,b in cases:
 r=s.add(m,a,b)
 if r != a+b: bad.append((a,b,r,a+b))
print('api cases',len(cases),'errors',len(bad),bad[:5])
# Demonstrate content-dependent QK scores in the actual first attention block.
def inp(a,b):
 ad=torch.tensor([[ord(c)-48 for c in reversed(f'{a:014d}')]]); bd=torch.tensor([[ord(c)-48 for c in reversed(f'{b:014d}')]])
 p=torch.nn.functional.pad(m.pos,(0,8)); return m.ea(ad)+m.eb(bd)+p[:14]
x1=inp(12345678901234,11111111111111); x2=inp(98765432109876,22222222222222); block=m.blocks[0]
def scores(x):
 q,k,_=block.qkv(block.n1(x)).chunk(3,-1); q=q.view(1,14,2,5).transpose(1,2); k=k.view(1,14,2,5).transpose(1,2); return q@k.transpose(-2,-1)/math.sqrt(5)
print('qk delta',float((scores(x1)-scores(x2)).abs().max()))
# Verify trained head is consequential.
base=s.add(m,12345678901234,87654321098765); saved=m.head.weight.detach().clone(); m.head.weight.data.zero_(); changed=s.add(m,12345678901234,87654321098765); m.head.weight.data.copy_(saved)
print('parameter perturbation',base,changed,base!=changed)
