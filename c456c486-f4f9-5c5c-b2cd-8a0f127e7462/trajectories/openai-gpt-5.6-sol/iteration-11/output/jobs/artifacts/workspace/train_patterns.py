import os,random,time,torch
import torch.nn.functional as F
from train import Adder,random_batch,carry_batch,DEVICE
from finetune import pattern_batch,eval_model

torch.manual_seed(20250816);random.seed(20250816)
model=Adder(10,32,6).to(DEVICE)
opt=torch.optim.AdamW(model.parameters(),lr=3e-3,betas=(.9,.98),weight_decay=.003)
batch=4096;steps=10000;best=(10**9,None)
print('parameters',sum(p.numel() for p in model.parameters()),flush=True)
for step in range(1,steps+1):
 l,r,t=random_batch(batch)
 nc=batch*12//100; np=batch*10//100
 cl,cr,ct=carry_batch(nc);pl,pr,pt=pattern_batch(np)
 l[:nc],r[:nc],t[:nc]=cl,cr,ct
 l[nc:nc+np],r[nc:nc+np],t[nc:nc+np]=pl,pr,pt
 loss=F.cross_entropy(model(l,r).flatten(0,1),t.flatten())
 opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
 progress=step/steps
 lr=3e-3 if progress<.45 else 1e-3 if progress<.7 else 3e-4 if progress<.88 else 1e-4 if progress<.96 else 3e-5
 for g in opt.param_groups:g['lr']=lr
 if step%500==0:
  metrics=eval_model(model,131072)
  score=metrics[0]+metrics[1]*2000+metrics[2]*4
  print(step,float(loss),lr,metrics,flush=True)
  if metrics[1]==0 and metrics[0]<=40 and score<best[0]:
   best=(score,metrics);torch.save({'model':model.state_dict(),'config':(10,32,6),'step':step,'metric':metrics},'/workspace/best_pattern.pt')
print('best',best,flush=True)
