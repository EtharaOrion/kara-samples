import sys,torch
import torch.nn.functional as F
sys.path.insert(0,'/workspace')
from train import Adder,mixed_batch,result_digits,evaluate,export_submission,N
D='cuda';torch.manual_seed(2112);m=Adder(10,8).to(D);m.load_state_dict(torch.load('/workspace/model_h8_uncalibrated.pt',weights_only=True));o=torch.optim.AdamW(m.parameters(),lr=8e-6,weight_decay=.001)
for step in range(1,4001):
 a,b,y=mixed_batch(4096,.4); n=2048
 ca=torch.randint(0,10,(n,N),device=D);cb=torch.randint(0,10,(n,N),device=D);ca[:,-1]=0;cb[:,-1]=0
 start=torch.randint(0,14,(n,),device=D);pos=torch.arange(14,device=D)[None];chain=pos>start[:,None];av=torch.randint(0,10,(n,14),device=D);ca[:,:14]=torch.where(chain,av,ca[:,:14]);cb[:,:14]=torch.where(chain,9-av,cb[:,:14]);r=torch.arange(n,device=D);ia=torch.randint(1,10,(n,),device=D);ca[r,start]=ia;cb[r,start]=10-ia+torch.randint(0,10,(n,),device=D).remainder(ia)
 cy=result_digits(ca,cb);a[:n],b[:n],y[:n]=ca,cb,cy
 logits=m(a,b); losses=F.cross_entropy(logits.reshape(-1,10),y.reshape(-1),reduction='none').view(-1,N); weights=torch.ones(N,device=D);weights[12:]=3;loss=(losses*weights).mean();o.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);o.step()
 if step%1000==0:print(step,loss.item(),evaluate(m,16,8192,False),evaluate(m,8,8192,True),flush=True)
torch.save(m.state_dict(),'/workspace/model_h8_top.pt');export_submission(m,'/workspace/submission.py')
