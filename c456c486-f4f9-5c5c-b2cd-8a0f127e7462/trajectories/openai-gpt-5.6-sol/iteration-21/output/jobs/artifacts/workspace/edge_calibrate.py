import sys,torch
import torch.nn.functional as F
sys.path.insert(0,'/workspace')
from train import Adder,mixed_batch,result_digits,evaluate,export_submission,N
D='cuda';torch.manual_seed(2116);m=Adder(10,8).to(D);m.load_state_dict(torch.load('/workspace/model_h8_final.pt',weights_only=True));o=torch.optim.AdamW(m.parameters(),lr=2e-6,weight_decay=.001)
for step in range(1,1501):
 a,b,y=mixed_batch(4096,.4);n=1024;ca=torch.zeros(n,N,dtype=torch.long,device=D);cb=torch.zeros_like(ca);r=torch.arange(n,device=D);end=torch.randint(2,15,(n,),device=D);ca[:,0]=torch.randint(1,10,(n,),device=D);cb[:,0]=10-ca[:,0]+torch.randint(0,10,(n,),device=D).remainder(ca[:,0]);pos=torch.arange(14,device=D)[None];run=(pos>0)&(pos<end[:,None]);av=torch.randint(0,10,(n,14),device=D);ca[:,:14]=torch.where(run,av,ca[:,:14]);cb[:,:14]=torch.where(run,9-av,cb[:,:14]);cy=result_digits(ca,cb);a[:n],b[:n],y[:n]=ca,cb,cy;loss=F.cross_entropy(m(a,b).reshape(-1,10),y.reshape(-1));o.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);o.step()
 if step%500==0:print(step,loss.item(),evaluate(m,16,8192,False),evaluate(m,8,8192,True),flush=True)
torch.save(m.state_dict(),'/workspace/model_h8_edge.pt');export_submission(m,'/workspace/submission.py')
