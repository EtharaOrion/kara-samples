import sys,torch
import torch.nn.functional as F
sys.path.insert(0,'/workspace')
from train import Adder,mixed_batch,result_digits,evaluate,export_submission,N
D='cuda';torch.manual_seed(2114);m=Adder(10,8).to(D);m.load_state_dict(torch.load('/workspace/model_h8_top.pt',weights_only=True));o=torch.optim.AdamW(m.parameters(),lr=3e-6,weight_decay=.001)
for step in range(1,2001):
 a,b,y=mixed_batch(4096,.4);n=1024
 ca=torch.randint(0,10,(n,N),device=D);cb=torch.randint(0,10,(n,N),device=D);ca[:,-1]=0;cb[:,-1]=0
 start=torch.randint(0,13,(n,),device=D);end=start+1+(torch.rand(n,device=D)*(14-start)).long();pos=torch.arange(14,device=D)[None];inside=(pos>start[:,None])&(pos<end[:,None]);av=torch.randint(0,10,(n,14),device=D);ca[:,:14]=torch.where(inside,av,ca[:,:14]);cb[:,:14]=torch.where(inside,9-av,cb[:,:14]);r=torch.arange(n,device=D);ia=torch.randint(1,10,(n,),device=D);ca[r,start]=ia;cb[r,start]=10-ia+torch.randint(0,10,(n,),device=D).remainder(ia);stop=end<14;sr=r[stop];sp=end[stop];sa=torch.randint(0,9,(sr.numel(),),device=D);ca[sr,sp]=sa;cb[sr,sp]=torch.randint(0,9,(sr.numel(),),device=D).remainder(9-sa)
 cy=result_digits(ca,cb);a[:n],b[:n],y[:n]=ca,cb,cy;loss=F.cross_entropy(m(a,b).reshape(-1,10),y.reshape(-1));o.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);o.step()
 if step%1000==0:print(step,loss.item(),evaluate(m,16,8192,False),evaluate(m,8,8192,True),flush=True)
torch.save(m.state_dict(),'/workspace/model_h8_final.pt');export_submission(m,'/workspace/submission.py')
