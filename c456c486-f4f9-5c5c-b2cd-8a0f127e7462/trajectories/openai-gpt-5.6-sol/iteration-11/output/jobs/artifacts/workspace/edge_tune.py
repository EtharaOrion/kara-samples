import torch
import torch.nn.functional as F
from train import Adder,random_batch,carry_batch
from finetune import explicit_data,eval_model
ck=torch.load('/workspace/best_finetuned.pt',weights_only=True)
m=Adder(*ck['config']).cuda();m.load_state_dict(ck['model'])
opt=torch.optim.AdamW(m.parameters(),lr=3e-5,betas=(.9,.98),weight_decay=0)
el,er,et=explicit_data();batch=4096;best=None
for step in range(1,2001):
 l,r,t=random_batch(batch); n=512
 ids=torch.randint(len(el),(n,),device='cuda');l[:n],r[:n],t[:n]=el[ids],er[ids],et[ids]
 cl,cr,ct=carry_batch(512);l[n:2*n],r[n:2*n],t[n:2*n]=cl,cr,ct
 loss=F.cross_entropy(m(l,r).flatten(0,1),t.flatten())
 opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
 if step%100==0:
  met=eval_model(m,131072);print(step,float(loss),met,flush=True)
  if met[1]==0 and met[0]<50:
   best=met;torch.save({'model':m.state_dict(),'config':ck['config'],'step':step,'metric':met},'/workspace/final.pt')
print('best',best)
