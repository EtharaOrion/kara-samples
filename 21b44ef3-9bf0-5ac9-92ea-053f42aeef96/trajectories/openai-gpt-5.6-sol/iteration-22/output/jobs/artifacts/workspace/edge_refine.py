import sys,time,torch
from torch import nn
import torch.nn.functional as F
sys.path.insert(0,'/workspace')
from submission import AdditionTransformer
from train import batch,accuracy,BATCH,LOW,HIGH,POW8,POW9

def edge_batch(n):
    r=torch.randint(1,8,(n,),device='cuda'); s=torch.randint(1,8,(n,),device='cuda')
    pr=10**r; ps=10**s
    hi1=torch.randint(1,10,(n,),device='cuda')*10_000_000
    hi2=torch.randint(1,10,(n,),device='cuda')*10_000_000
    # Diverse near-nine run endings, including unequal runs and carry stop positions.
    off1=torch.randint(0,10,(n,),device='cuda'); off2=torch.randint(0,10,(n,),device='cuda')
    a=(hi1+pr-1-off1).clamp(LOW,HIGH); b=(hi2+ps-1-off2).clamp(LOW,HIGH)
    swap=torch.rand(n,device='cuda')<.3
    b=torch.where(swap,hi2+off2,b).clamp(LOW,HIGH)
    ad=(a[:,None]//POW8)%10;bd=(b[:,None]//POW8)%10;out=(a+b)[:,None]//POW9%10
    x=torch.empty(n,25,dtype=torch.long,device='cuda');x[:,:16:2]=ad;x[:,1:16:2]=bd;x[:,16]=10;x[:,17:]=out[:,:8]
    return x,out

torch.manual_seed(220024);torch.set_float32_matmul_precision('high')
m=AdditionTransformer().cuda();m.load_state_dict(torch.load('/workspace/refined_best.pt',weights_only=True));m.train()
opt=torch.optim.AdamW(m.parameters(),lr=5e-6,betas=(.9,.98),weight_decay=.001,fused=True); start=time.time()
for step in range(1,12001):
    x,y=batch(.45); ex,ey=edge_batch(BATCH//2); x[:BATCH//2]=ex;y[:BATCH//2]=ey
    z=m(x)[:,16:25];loss=F.cross_entropy(z.reshape(-1,10),y.reshape(-1));opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
    if step%1000==0:print(step,loss.item(),time.time()-start,flush=True)
    if step%3000==0:
      r=accuracy(m,100000,'random');s=accuracy(m,100000,'structured');print('VALID',r,s,flush=True);torch.save(m.state_dict(),f'/workspace/edge_model_{step}.pt')
torch.save(m.state_dict(),'/workspace/edge_best.pt')
