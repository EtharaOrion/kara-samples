import random, sys, time, torch
sys.path.insert(0,'/workspace')
import submission
m,md=submission.build_model()
cases={(0,0),(1,0),(0,1),(99999999999999,1),(99999999999999,99999999999999),(50000000000000,50000000000000)}
for k in range(14):
 p=10**k
 for length in range(1,15-k):
  run=(10**length-1)*p
  cases.add((run,p));cases.add((p,run));cases.add((99999999999999-run,run));cases.add((run,99999999999999-run))
for d in range(10):
 r=int(str(d)*14)
 cases.add((r,1));cases.add((r,99999999999999-r));cases.add((r,r))
bad=[]
t=time.time()
for a,b in cases:
 y=submission.add(m,a,b)
 if y != a+b: bad.append((a,b,y,a+b))
print('params',sum(p.numel() for p in m.parameters()),'metadata',md,'cases',len(cases),'bad',len(bad),'sample',bad[:10],'seconds',time.time()-t)
# QK input dependence in encoder first head
s1=torch.tensor([[0]*28]);s2=torch.tensor([[i%10 for i in range(28)]])
x1=m.token(s1)+m.place.repeat(2,1)+m.kind.repeat_interleave(14,0)
x2=m.token(s2)+m.place.repeat(2,1)+m.kind.repeat_interleave(14,0)
a=m.encoder.attn
q1=a.q(m.encoder.n1(x1)).view(1,28,2,8).transpose(1,2);k1=a.k(m.encoder.n1(x1)).view(1,28,2,8).transpose(1,2)
q2=a.q(m.encoder.n1(x2)).view(1,28,2,8).transpose(1,2);k2=a.k(m.encoder.n1(x2)).view(1,28,2,8).transpose(1,2)
print('qk_delta',((q1@k1.transpose(-2,-1))-(q2@k2.transpose(-2,-1))).abs().max().item())
