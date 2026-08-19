import random, time, torch
import torch.nn.functional as F
from submission import AdditionTransformer
from train import batch_data, evaluate
D='cuda'; torch.manual_seed(812); random.seed(812)

def focused(batch):
 a=torch.randint(0,10,(batch,14),device=D); b=torch.randint(0,10,(batch,14),device=D)
 kind=random.randrange(4)
 if kind==0:
  # Arbitrary complementary run, with an incoming carry trigger.
  start=random.randrange(0,10); length=random.randrange(3,15-start)
  a[:,start:start+length]=torch.randint(0,10,(batch,length),device=D)
  b[:,start:start+length]=9-a[:,start:start+length]
  if start: a[:,start-1]=9; b[:,start-1]=torch.randint(1,10,(batch,),device=D)
 elif kind==1:
  # Long run of nines plus a small trigger operand.
  start=random.randrange(0,10); length=random.randrange(3,15-start)
  a[:,start:start+length]=9; b[:,start:start+length]=0
  b[:,start]=torch.randint(1,10,(batch,),device=D)
 elif kind==2:
  da=torch.randint(0,10,(batch,1),device=D); db=torch.randint(0,10,(batch,1),device=D)
  a[:]=da; b[:]=db
 else:
  # Whole-sequence complements, optionally carrying from the first digit.
  a=torch.randint(0,10,(batch,14),device=D); b=9-a
  trigger=torch.randint(0,2,(batch,),device=D); b[:,0]=(b[:,0]+trigger).remainder(10)
 x=torch.zeros(batch,15,2,dtype=torch.long,device=D);x[:,:14,0]=a;x[:,:14,1]=b
 y=torch.empty(batch,15,dtype=torch.long,device=D);carry=torch.zeros(batch,dtype=torch.long,device=D)
 for p in range(14):
  t=a[:,p]+b[:,p]+carry;y[:,p]=t%10;carry=t//10
 y[:,14]=carry
 return x,y

m=AdditionTransformer(width=10).to(D);m.load_state_dict(torch.load('/workspace/model_w10.pt',weights_only=True));m.train()
o=torch.optim.AdamW(m.parameters(),lr=7e-5,weight_decay=0.001)
start=time.time()
for step in range(1,5001):
 x,y=focused(2048) if random.random()<0.5 else batch_data(2048)
 o.zero_grad(set_to_none=True);z=m(x);loss=F.cross_entropy(z.reshape(-1,10),y.reshape(-1));loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);o.step()
 if step%250==0:
  acc,dig=evaluate(m,20,1000);print(step,loss.item(),acc,dig,time.time()-start,flush=True)
  if acc>=.995: torch.save(m.state_dict(),'/workspace/model_w10_focused.pt')
