import importlib.util, random, torch
spec=importlib.util.spec_from_file_location('submission','/workspace/submission.py'); sub=importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
m,meta=sub.build_model(); print('PARAMETERS',sum(p.numel() for p in m.parameters()),meta)
device='cuda'; m.to(device)
POW=torch.tensor([10**i for i in range(8)],device=device)

def eval_set(n,structured=False):
 good=total=0
 for start in range(0,n,10000):
  z=min(10000,n-start); a=torch.randint(10_000_000,100_000_000,(z,),device=device); b=torch.randint(10_000_000,100_000_000,(z,),device=device)
  if structured:
   run=torch.randint(1,8,(z,),device=device); p=10**run
   a=(a//p)*p+p-1; b=(b//p)*p
   a.clamp_(10_000_000,99_999_999);b.clamp_(10_000_000,99_999_999)
  da=(a[:,None]//POW)%10;db=(b[:,None]//POW)%10
  seq=torch.empty(z,17,dtype=torch.long,device=device);seq[:,:16:2]=da;seq[:,1:16:2]=db;seq[:,16]=10;out=[]
  with torch.no_grad():
   for j in range(9):d=m(seq)[:,-1].argmax(1);out.append(d);seq=torch.cat((seq,d[:,None]),1)
  pred=(torch.stack(out,1)*torch.tensor([10**i for i in range(9)],device=device)).sum(1);good+=int((pred==a+b).sum());total+=z
 return good,total
print('RANDOM',eval_set(500000)); print('STRUCTURED',eval_set(500000,True))
# CPU direct API and broad deterministic edges.
m.cpu(); vals={10_000_000,10_000_001,10_000_009,10_000_099,10_000_999,10_009_999,10_099_999,10_999_999,11_111_111,20_000_000,40_000_009,49_999_999,50_000_000,50_000_001,89_999_999,90_000_000,98_888_888,99_000_001,99_899_999,99_990_000,99_999_900,99_999_990,99_999_998,99_999_999}
for k in range(1,8):
 p=10**k
 for lead in range(1,10):
  for d in (-1,0,1):
   x=lead*10_000_000 + p+d
   if 10_000_000<=x<=99_999_999:vals.add(x)
pairs=list(__import__('itertools').product(sorted(vals),repeat=2));bad=[]
for a,b in pairs:
 r=sub.add(m,a,b)
 if r!=a+b:bad.append((a,b,r,a+b))
print('EDGES',len(pairs)-len(bad),len(pairs),'bad',bad[:10])
random.seed(7); api=[(random.randint(10_000_000,99_999_999),random.randint(10_000_000,99_999_999)) for _ in range(2000)]
print('API',sum(sub.add(m,a,b)==a+b for a,b in api),len(api))
# Ablations must materially change outputs.
sample=api[:40];base=[sub.add(m,a,b) for a,b in sample]
with torch.no_grad(): saved=[x.clone() for x in m.o]; [x.zero_() for x in m.o]
abl=[sub.add(m,a,b) for a,b in sample]
with torch.no_grad(): [x.copy_(y) for x,y in zip(m.o,saved)]; cs=m.classifier_free.clone();m.classifier_free.zero_()
out=[sub.add(m,a,b) for a,b in sample]
with torch.no_grad():m.classifier_free.copy_(cs)
print('ATTN_ABLATION_CHANGED',sum(x!=y for x,y in zip(base,abl)),'OUTPUT_ABLATION_CHANGED',sum(x!=y for x,y in zip(base,out)))
