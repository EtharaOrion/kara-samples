import importlib.util,random,time,torch
spec=importlib.util.spec_from_file_location('s','/workspace/submission.py'); s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
m,meta=s.build_model(); print('PARAMS',sum(p.numel() for p in m.parameters()),meta,flush=True); m=m.cuda()
@torch.no_grad()
def check(n,structured=False,bs=10000):
 good=tot=0
 while tot<n:
  z=min(bs,n-tot); a=torch.randint(10_000_000,100_000_000,(z,),device='cuda'); b=torch.randint(10_000_000,100_000_000,(z,),device='cuda')
  if structured:
   k=torch.randint(1,8,(z,),device='cuda'); p=10**k
   a=(a.div(p,rounding_mode='floor')*p+p-1).clamp(10_000_000,99_999_999)
   b=(100_000_000-a+torch.randint(-20,21,(z,),device='cuda')).clamp(10_000_000,99_999_999)
  aa=a.clone();bb=b.clone(); seq=torch.empty((z,17),dtype=torch.long,device='cuda')
  for j in range(8): seq[:,2*j]=aa%10;seq[:,2*j+1]=bb%10;aa//=10;bb//=10
  seq[:,16]=10; val=torch.zeros(z,dtype=torch.long,device='cuda'); mul=1
  for j in range(9):
   d=m(seq)[:,-1].argmax(-1); val+=d*mul; mul*=10;seq=torch.cat((seq,d[:,None]),1)
  good+=(val==a+b).sum().item();tot+=z
 return good,tot
print('RANDOM',check(500000),flush=True); print('STRUCTURED',check(500000,True),flush=True)
m=m.cpu(); random.seed(9); cases=[(random.randint(10_000_000,99_999_999),random.randint(10_000_000,99_999_999)) for _ in range(200)]
vals=[10_000_000,10_000_001,10_999_999,40_000_009,49_999_999,89_999_999,99_000_001,99_899_999,99_999_900,99_999_999]
cases += [(a,b) for a in vals for b in vals]
bad=[(a,b,s.add(m,a,b),a+b) for a,b in cases if s.add(m,a,b)!=a+b];print('DIRECT',len(cases)-len(bad),len(cases),bad[:5],flush=True)
# model dependence
normal=[s.add(m,*x) for x in cases[:20]]; o0=m.o0.detach().clone();o1=m.o1.detach().clone()
with torch.no_grad():m.o0.zero_();m.o1.zero_()
ab=[s.add(m,*x) for x in cases[:20]]
with torch.no_grad():m.o0.copy_(o0);m.o1.copy_(o1)
ow=m.outw.detach().clone();ob=m.outb.detach().clone()
with torch.no_grad():m.outw.zero_();m.outb.zero_()
co=[s.add(m,*x) for x in cases[:20]]
print('ABLATION',sum(x!=y for x,y in zip(normal,ab)),sum(x!=y for x,y in zip(normal,co)),flush=True)
