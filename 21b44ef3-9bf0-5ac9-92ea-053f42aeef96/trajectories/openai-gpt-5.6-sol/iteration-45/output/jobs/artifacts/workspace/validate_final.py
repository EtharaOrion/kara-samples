import sys, random, torch
sys.path.insert(0,'/workspace')
import submission
m,md=submission.build_model(); m.cuda()
P=torch.tensor([10**i for i in range(9)],device='cuda')
def dig(x,n=8): return (x[:,None]//P[:n])%10
@torch.no_grad()
def run(a,b):
 ad,bd=dig(a),dig(b); x=torch.empty(len(a),17,dtype=torch.long,device='cuda')
 x[:,0:16:2]=ad;x[:,1:16:2]=bd;x[:,16]=10; out=[]
 for _ in range(9):
  d=m(x)[:,-1].argmax(-1);out.append(d);x=torch.cat((x,d[:,None]),1)
 return (torch.stack(out,1)*P).sum(1)
good=tot=0
for _ in range(10):
 a=torch.randint(10_000_000,100_000_000,(50000,),device='cuda');b=torch.randint(10_000_000,100_000_000,(50000,),device='cuda')
 good+=int((run(a,b)==a+b).sum());tot+=len(a)
print('random',good,tot)
# Broad cartesian boundaries.
vals={10_000_000,99_999_999,10_000_001,99_999_998,50_000_000}
for p in [10,100,1000,10000,100000,1000000,10000000]:
 for k in range(1,10):
  base=k*p
  for off in [-2,-1,0,1,2]:
   for lead in [10_000_000,20_000_000,50_000_000,90_000_000]:
    v=lead+base+off
    if 10_000_000<=v<100_000_000: vals.add(v)
vals=sorted(vals); pairs=[]
random.seed(7)
for _ in range(100000): pairs.append((random.choice(vals),random.choice(vals)))
for a in vals[:200]:
 for b in [10_000_000,10_999_999,49_999_999,50_000_001,89_999_999,99_999_999]: pairs.append((a,b))
good=0
for i in range(0,len(pairs),20000):
 q=pairs[i:i+20000];a=torch.tensor([x for x,y in q],device='cuda');b=torch.tensor([y for x,y in q],device='cuda');good+=int((run(a,b)==a+b).sum())
print('edge',good,len(pairs), 'vals',len(vals))
# API CPU
m.cpu(); random.seed(11); cases=[(random.randrange(10_000_000,100_000_000),random.randrange(10_000_000,100_000_000)) for _ in range(1000)]
print('api',sum(submission.add(m,a,b)==a+b for a,b in cases),len(cases))
# Ablations through projection and classifier.
baseline=[submission.add(m,a,b) for a,b in cases[:30]]
with torch.no_grad():
 saved=[p.weight.clone() for p in m.proj]
 for p in m.proj:p.weight.zero_()
att=[submission.add(m,a,b) for a,b in cases[:30]]
with torch.no_grad():
 for p,s in zip(m.proj,saved):p.weight.copy_(s)
 savedh=m.head.weight.clone();m.head.weight.zero_()
out=[submission.add(m,a,b) for a,b in cases[:30]]
with torch.no_grad():m.head.weight.copy_(savedh)
print('ablation_changed',sum(x!=y for x,y in zip(baseline,att)),sum(x!=y for x,y in zip(baseline,out)))
