
import random,time,torch
import torch.nn.functional as F
from submission import AdditionTransformer
D='cuda';torch.manual_seed(191);random.seed(191)
def make(batch,focus):
 a=torch.randint(0,10,(batch,14),device=D);b=torch.randint(0,10,(batch,14),device=D)
 if focus:
  # Every row gets a carry chain of independently sampled length and location.
  starts=torch.randint(0,5,(batch,),device=D); lens=torch.randint(1,15,(batch,),device=D)
  for i in range(batch):
   st=int(starts[i]); ln=min(int(lens[i]),14-st)
   a[i,st:st+ln]=9;b[i,st:st+ln]=0;b[i,st]=1
  # Include canonical chains starting at the least significant digit.
  for ln in range(1,15):
   rows=slice((ln-1)*batch//14,ln*batch//14)
   a[rows]=0;b[rows]=0;a[rows,:ln]=9;b[rows,0]=1
 x=torch.zeros(batch,15,2,dtype=torch.long,device=D);x[:,:14,0]=a;x[:,:14,1]=b
 y=torch.empty(batch,15,dtype=torch.long,device=D);c=torch.zeros(batch,dtype=torch.long,device=D)
 for p in range(14):t=a[:,p]+b[:,p]+c;y[:,p]=t%10;c=t//10
 y[:,14]=c;return x,y
m=AdditionTransformer(width=12,rounds=6).to(D);m.load_state_dict(torch.load('/workspace/model_w12.pt',weights_only=True));m.train()
o=torch.optim.AdamW(m.parameters(),lr=3e-5,weight_decay=0)
for step in range(1,1501):
 x,y=make(2048,random.random()<.35);o.zero_grad(set_to_none=True);z=m(x);loss=F.cross_entropy(z.flatten(0,1),y.flatten());loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);o.step()
 if step%500==0:
  m.eval();good=0
  with torch.no_grad():
   for _ in range(10):
    q,t=make(2000,False);good+=m(q).argmax(-1).eq(t).all(1).sum().item()
  print(step,loss.item(),good/20000,flush=True);m.train()
torch.save(m.state_dict(),'/workspace/model_w12_edge.pt')
