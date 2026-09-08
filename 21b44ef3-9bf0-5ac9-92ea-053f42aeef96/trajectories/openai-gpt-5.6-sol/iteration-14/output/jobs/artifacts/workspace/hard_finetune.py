import torch
import torch.nn.functional as F
from train import AdditionTransformer, make_batch, digits, evaluate, eval_curated, export, LO, HI

torch.manual_seed(1415);torch.backends.cuda.matmul.allow_tf32=True
dev=torch.device('cuda');m=AdditionTransformer().to(dev);m.load_state_dict(torch.load('/workspace/final.pt',weights_only=True))
# Exhaustive difficult families: sparse prefixes/suffixes, carry stopping points, and complements.
pairs=set()
for lead1 in range(1,10):
 for lead2 in range(1,10):
  for p in range(1,8):
   s=10**p
   bases=[lead1*10_000_000, (lead1+1)*10_000_000 if lead1<9 else HI+1]
   for base in bases:
    for da in range(-3,4):
     a=base+da
     if not LO<=a<=HI:continue
     for db in range(-3,4):
      for b in (lead2*10_000_000+s-1+db, lead2*10_000_000-s+db, 100_000_000-a+db):
       if LO<=b<=HI:pairs.add((a,b));pairs.add((b,a))
# Densify canonical x0000001 + y9999999-like failures.
for x in range(1,10):
 for y in range(1,10):
  for da in range(0,101):
   for db in range(0,101):
    a=x*10_000_000+da;b=(y+1)*10_000_000-1-db
    if LO<=a<=HI and LO<=b<=HI:pairs.add((a,b));pairs.add((b,a))
pairs=list(pairs);A=torch.tensor([a for a,b in pairs],device=dev);B=torch.tensor([b for a,b in pairs],device=dev)
print('hard pool',len(pairs),'initial',evaluate(m,100000,structured=0),eval_curated(m)[0])
opt=torch.optim.AdamW(m.parameters(),lr=8e-6,weight_decay=0)
best=-1
for step in range(1,3001):
 if step==2001:
  for g in opt.param_groups:g['lr']=2e-6
 t,y,_,_=make_batch(2048,dev,.3)
 k=int(2048*.70);idx=torch.randint(len(A),(k,),device=dev);a=A[idx];b=B[idx]
 t[:k,0:16:2]=digits(a);t[:k,1:16:2]=digits(b);y[:k]=digits(a+b,9)
 opt.zero_grad(set_to_none=True);z=m(t)
 ce=F.cross_entropy(z.flatten(0,1),y.flatten(),reduction='none').view(-1,9)
 loss=(ce.mean(1)+.5*ce.max(1).values).mean();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
 if step%250==0:
  r=evaluate(m,20000,structured=0)[0];e=eval_curated(m)[0];print(step,loss.item(),r,e,flush=True)
  score=min(r,e)
  if score>best:best=score;torch.save(m.state_dict(),'/workspace/hard_best.pt')
m.load_state_dict(torch.load('/workspace/hard_best.pt',weights_only=True));torch.save(m.state_dict(),'/workspace/final.pt');export(m,'/workspace/submission.py')
print('FINAL',evaluate(m,200000,structured=0),evaluate(m,500000,structured=.7),eval_curated(m))
