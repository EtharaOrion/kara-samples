import sys,random,time,torch
sys.path.insert(0,'/workspace');import submission
m,_=submission.build_model(); random.seed(94821)
e=0;t=time.time()
for _ in range(20000):
 a=random.randrange(10**14);b=random.randrange(10**14);e+=submission.add(m,a,b)!=(a+b)
print('random',e,'/20000',time.time()-t)
edge=[]
for st in range(14):
 p=10**st
 for ln in range(1,15-st):
  run=(10**(st+ln))-p
  edge += [(run,p),(run-p,p),(99999999999999,p),(p*5,p*5),(p*9,p*9)]
e2=sum(submission.add(m,a,b)!=(a+b) for a,b in edge)
print('systematic',e2,'/',len(edge))
# self-attention score input dependence in encoder
x1=m.ea(torch.zeros(1,14,dtype=torch.long))+m.eb(torch.zeros(1,14,dtype=torch.long))+m.pad2(m.sp)[None]
x2=m.ea(torch.arange(14).remainder(10)[None])+m.eb(torch.arange(14).mul(3).remainder(10)[None])+m.pad2(m.sp)[None]
a=m.enc.a
q1=a.q(m.enc.n1(x1));k1=a.k(m.enc.n1(x1));q2=a.q(m.enc.n1(x2));k2=a.k(m.enc.n1(x2))
print('qk_delta',((q1@k1.transpose(-2,-1))-(q2@k2.transpose(-2,-1))).abs().max().item())
