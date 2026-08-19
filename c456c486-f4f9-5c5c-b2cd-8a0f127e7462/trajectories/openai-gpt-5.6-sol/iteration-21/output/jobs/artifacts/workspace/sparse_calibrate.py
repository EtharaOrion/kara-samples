import sys, torch
import torch.nn.functional as F
sys.path.insert(0,"/workspace")
from train import Adder, mixed_batch, result_digits, evaluate, export_submission, N
D="cuda"; torch.manual_seed(2110)
m=Adder(10,8).to(D); m.load_state_dict(torch.load('/workspace/model_h8_uncalibrated.pt',weights_only=True))
o=torch.optim.AdamW(m.parameters(),lr=1e-5,weight_decay=.001)
for step in range(1,3001):
 a,b,y=mixed_batch(4096,.4)
 n=2048
 sa=torch.zeros(n,N,dtype=torch.long,device=D); sb=torch.zeros_like(sa)
 # 1-4 occupied columns, arbitrary aligned sums including carry/no-carry boundaries.
 for _ in range(4):
  active=torch.rand(n,device=D)<.65
  p=torch.randint(0,14,(n,),device=D); r=torch.arange(n,device=D)[active]
  sa[r,p[active]]=torch.randint(0,10,(r.numel(),),device=D)
  sb[r,p[active]]=torch.randint(0,10,(r.numel(),),device=D)
 # Half also contain a randomized exact-9 carry/no-carry run.
 start=torch.randint(0,14,(n,),device=D); end=start+1+(torch.rand(n,device=D)*(14-start)).long()
 pos=torch.arange(14,device=D)[None]; chosen=(torch.rand(n,device=D)<.5)[:,None]&(pos>=start[:,None])&(pos<end[:,None])
 av=torch.randint(0,10,(n,14),device=D); sa[:,:14]=torch.where(chosen,av,sa[:,:14]); sb[:,:14]=torch.where(chosen,9-av,sb[:,:14])
 sy=result_digits(sa,sb); a[:n],b[:n],y[:n]=sa,sb,sy
 loss=F.cross_entropy(m(a,b).reshape(-1,10),y.reshape(-1)); o.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1); o.step()
 if step%1000==0: print(step,loss.item(),evaluate(m,16,8192,False),evaluate(m,8,8192,True),flush=True)
torch.save(m.state_dict(),'/workspace/model_h8_sparse.pt'); export_submission(m,'/workspace/submission.py')
