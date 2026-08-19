import sys,torch
import torch.nn.functional as F
sys.path.insert(0,'/workspace')
from train import Adder,mixed_batch,result_digits,evaluate,export_submission,N
D='cuda';torch.manual_seed(2118);m=Adder(10,8).to(D);m.load_state_dict(torch.load('/workspace/model_h8_top.pt',weights_only=True));o=torch.optim.AdamW(m.parameters(),lr=1e-5,weight_decay=.001)
for step in range(1,2501):
 a,b,y=mixed_batch(4096,.45);n=1536;ca=torch.zeros(n,N,dtype=torch.long,device=D);cb=torch.zeros_like(ca);length=torch.randint(1,15,(n,),device=D);first=torch.randint(1,10,(n,),device=D);ca[:,0]=first;cb[:,0]=10-first;pos=torch.arange(14,device=D)[None];run=(pos>0)&(pos<length[:,None]);av=torch.randint(0,10,(n,14),device=D);ca[:,:14]=torch.where(run,av,ca[:,:14]);cb[:,:14]=torch.where(run,9-av,cb[:,:14]);cy=result_digits(ca,cb);a[:n],b[:n],y[:n]=ca,cb,cy;log=m(a,b);loss=F.cross_entropy(log.reshape(-1,10),y.reshape(-1));o.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);o.step()
 if step%500==0:print(step,loss.item(),evaluate(m,16,8192,False),evaluate(m,8,8192,True),flush=True)
torch.save(m.state_dict(),'/workspace/model_h8_exact.pt');export_submission(m,'/workspace/submission.py')
