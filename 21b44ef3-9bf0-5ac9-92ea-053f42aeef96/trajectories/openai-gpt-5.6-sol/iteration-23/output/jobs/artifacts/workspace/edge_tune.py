import time, torch
import torch.nn.functional as F
from submission import AdditionTransformer
from train import make_batch, digits

model=AdditionTransformer().cuda(); model.load_state_dict(torch.load('/workspace/trained.pt',weights_only=True)['model'])
vals=set()
for lead in range(1,10):
 for run in range(1,8):
  p=10**run
  for delta in [0,1,2,9,10,11,19,20,90,91,99,100,101,109,110,190,191,199,999,p-1,max(0,p-9),max(0,p-99)]:
   for base in [lead*10_000_000, (lead+1)*10_000_000-p]:
    v=base+delta
    if 10_000_000<=v<=99_999_999: vals.add(v)
vals=torch.tensor(sorted(vals),device='cuda'); print('pool',len(vals),flush=True)
opt=torch.optim.AdamW(model.parameters(),lr=8e-6,betas=(.9,.98),weight_decay=0.0001)
start=time.time()
for step in range(1,8001):
 x,y=make_batch(4096,.5)
 # Replace half with independently sampled Cartesian edge values.
 n=4096
 aa=vals[torch.randint(0,len(vals),(n,),device='cuda')]; bb=vals[torch.randint(0,len(vals),(n,),device='cuda')]
 da,db=digits(aa,8),digits(bb,8); yy=digits(aa+bb,9)
 xx=torch.empty(n,25,dtype=torch.long,device='cuda'); xx[:,:16:2]=da;xx[:,1:16:2]=db;xx[:,16]=10;xx[:,17:]=yy[:,:8]
 x=torch.cat([x,xx]);y=torch.cat([y,yy])
 logits=model(x)[:,16:25];loss=F.cross_entropy(logits.reshape(-1,10),y.reshape(-1))
 opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
 if step==5000:
  for g in opt.param_groups:g['lr']=3e-6
 if step%1000==0:print(step,loss.item(),time.time()-start,flush=True)
torch.save({'model':model.state_dict(),'step':8000},'/workspace/trained_edge.pt')
