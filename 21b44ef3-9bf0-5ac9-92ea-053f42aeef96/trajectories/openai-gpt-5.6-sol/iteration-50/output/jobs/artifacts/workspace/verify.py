import importlib.util, random, sys, time
import torch
spec=importlib.util.spec_from_file_location('final_submission','/workspace/submission.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
m,meta=mod.build_model(); print('PARAMS',sum(p.numel() for p in m.parameters()),meta)
random.seed(83017)
# Direct autoregressive API checks on broad random and deterministic boundaries.
cases=[(random.randint(10_000_000,99_999_999),random.randint(10_000_000,99_999_999)) for _ in range(10000)]
vals=set([10_000_000,10_000_001,10_000_009,10_000_099,10_000_999,10_009_999,10_099_999,10_999_999,11_111_111,20_000_000,40_000_009,49_999_999,50_000_000,89_999_999,90_000_000,98_999_999,99_000_001,99_899_999,99_999_900,99_999_990,99_999_998,99_999_999])
for k in range(1,8):
 p=10**k
 for lead in range(1,10):
  for off in (-2,-1,0,1,2):
   vals.add(max(10_000_000,min(99_999_999,lead*10_000_000+p+off)))
cases += [(a,b) for a in vals for b in vals]
start=time.time(); bad=[]
for i,(a,b) in enumerate(cases):
 r=mod.add(m,a,b)
 if r!=a+b:
  bad.append((a,b,r,a+b))
  if len(bad)>=20: break
print('DIRECT',len(cases)-len(bad),'/',len(cases),'bad',bad,'seconds',time.time()-start)
# Input-dependent attention score check from explicitly reconstructed first layer.
with torch.no_grad():
 def seq(a,b):
  aa=[int(c) for c in str(a)[::-1]]; bb=[int(c) for c in str(b)[::-1]]; z=[]
  for x,y in zip(aa,bb): z.extend((x,y))
  return torch.tensor([z+[10]])
 tok,pos,k,v,f1,out=m._expand()
 scores=[]
 for t in (seq(12_345_678,87_654_321),seq(99_999_999,10_000_001)):
  x=torch.nn.functional.embedding(t,tok)+pos[:17]; z=torch.nn.functional.layer_norm(x,(20,))
  q=torch.cat((m.q0,torch.zeros(20,1)),1)
  qq=torch.nn.functional.linear(z,q,m.qb0).view(1,17,4,5).transpose(1,2)
  kk=torch.nn.functional.linear(z,k).view(1,1,17,5)
  scores.append(qq@kk.transpose(-1,-2))
 print('ATTENTION_SCORE_DELTA',float((scores[0]-scores[1]).abs().max()))
# Ablate attention output and classifier parameters in-place and demand changed answers.
sample=cases[:40]; normal=[mod.add(m,*x) for x in sample]
backup0=m.o0.detach().clone(); backup1=m.o1.detach().clone()
with torch.no_grad(): m.o0.zero_(); m.o1.zero_()
ablated=[mod.add(m,*x) for x in sample]
with torch.no_grad(): m.o0.copy_(backup0); m.o1.copy_(backup1)
bo=m.outw.detach().clone(); bb=m.outb.detach().clone()
with torch.no_grad(): m.outw.zero_(); m.outb.zero_()
corrupt=[mod.add(m,*x) for x in sample]
with torch.no_grad(): m.outw.copy_(bo); m.outb.copy_(bb)
print('ATTENTION_ABLATION_CHANGED',sum(x!=y for x,y in zip(normal,ablated)),'/40')
print('CLASSIFIER_CORRUPTION_CHANGED',sum(x!=y for x,y in zip(normal,corrupt)),'/40')
