import sys,time,torch
from torch import nn
import torch.nn.functional as F
sys.path.insert(0,'/workspace')
from submission import AdditionTransformer
from train import batch,accuracy,BATCH,LOW,HIGH,POW8,POW9

def boundaries(n):
 r=torch.randint(1,8,(n,),device='cuda');s=torch.randint(1,8,(n,),device='cuda');pr=10**r;ps=10**s
 raw1=torch.randint(LOW,HIGH+1,(n,),device='cuda');raw2=torch.randint(LOW,HIGH+1,(n,),device='cuda')
 o1=torch.randint(0,10,(n,),device='cuda');o2=torch.randint(0,10,(n,),device='cuda')
 kind=torch.randint(0,4,(n,),device='cuda')
 a=torch.where(kind%2==0,(raw1//pr)*pr+o1,(raw1//pr)*pr+pr-1-o1).clamp(LOW,HIGH)
 b=torch.where(kind<2,(raw2//ps)*ps+o2,(raw2//ps)*ps+ps-1-o2).clamp(LOW,HIGH)
 ad=(a[:,None]//POW8)%10;bd=(b[:,None]//POW8)%10;y=(a+b)[:,None]//POW9%10
 x=torch.empty(n,25,dtype=torch.long,device='cuda');x[:,:16:2]=ad;x[:,1:16:2]=bd;x[:,16]=10;x[:,17:]=y[:,:8]
 return x,y

torch.manual_seed(220025);torch.set_float32_matmul_precision('high');m=AdditionTransformer().cuda();m.load_state_dict(torch.load('/workspace/edge_best.pt',weights_only=True));m.train()
o=torch.optim.AdamW(m.parameters(),lr=2e-6,betas=(.9,.98),weight_decay=0,fused=True)
for step in range(1,8001):
 x,y=batch(.4);ex,ey=boundaries(BATCH*3//4);x[:len(ex)]=ex;y[:len(ex)]=ey
 z=m(x)[:,16:25];loss=F.cross_entropy(z.reshape(-1,10),y.reshape(-1));o.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);o.step()
 if step%2000==0: print(step,loss.item(),accuracy(m,100000,'random'),accuracy(m,100000,'structured'),flush=True);torch.save(m.state_dict(),f'/workspace/final_model_{step}.pt')
torch.save(m.state_dict(),'/workspace/final_best.pt')
