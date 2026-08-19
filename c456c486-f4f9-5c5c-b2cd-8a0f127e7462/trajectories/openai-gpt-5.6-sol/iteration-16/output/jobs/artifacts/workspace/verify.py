import random, sys, time, torch
sys.path.insert(0, '/workspace')
import submission
model, metadata = submission.build_model()
assert isinstance(model, torch.nn.Module)
assert sum(p.numel() for p in model.parameters()) == 7786
cases=[]
r=random.Random(1616)
for _ in range(2000): cases.append((r.randrange(10**14),r.randrange(10**14)))
M=10**14-1
cases += [(0,0),(M,0),(M,1),(M,M)]
for k in range(1,15):
 p=10**k
 cases += [(p-1,1),(p-1,p-1),(M-(p-1),p-1),(M,p-1)]
 for d in range(1,10):
  rep=d*(p-1)//9
  cases += [(rep,rep),(rep,min(M,p)),(M-rep,rep)]
start=time.time(); bad=[]
for a,b in cases:
 got=submission.add(model,a,b)
 if got != a+b: bad.append((a,b,got,a+b))
print('cases',len(cases),'bad',len(bad),'seconds',time.time()-start,'parameters',sum(p.numel() for p in model.parameters()),metadata)
print(bad[:10])
