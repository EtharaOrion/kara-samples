import sys, random
sys.path.append('/usr/local/lib/python3.11/dist-packages'); sys.path.insert(0,'/workspace')
import torch
import torch.nn.functional as F
from oldmodel import AdditionTransformer
D='cuda'; B=8192; lo=10_000_000; hi=99_999_999
places=torch.tensor([10**i for i in range(10)],device=D)
def digs(x): return (x[:,None]//places[:9]%10).long()
def batch():
 a=torch.randint(lo,hi+1,(B,),device=D); b=torch.randint(lo,hi+1,(B,),device=D)
 n=B//3; k=torch.randint(1,8,(n,),device=D); p=places[k]
 # Prefix + all-nine suffix versus decimal-boundary neighborhoods.
 pa=torch.randint(1,10,(n,),device=D)*10_000_000
 qa=torch.randint(0,10_000_000,(n,),device=D)
 aa=((pa+qa)//p)*p+p-1
 mult=torch.randint(1,100_000_000,(n,),device=D)
 off=torch.randint(-10,11,(n,),device=D)
 bb=(mult//p)*p+off
 good=(aa>=lo)&(aa<=hi)&(bb>=lo)&(bb<=hi)
 a[:n]=torch.where(good,aa,a[:n]);b[:n]=torch.where(good,bb,b[:n])
 # Explicitly replay broad leading-prefix pairs over every carry length.
 q=B//12; kk=torch.randint(1,8,(q,),device=D); pp=places[kk]
 la=torch.randint(1,10,(q,),device=D)*10_000_000
 lb=torch.randint(1,10,(q,),device=D)*10_000_000
 a[n:n+q]=(la//pp)*pp+pp-1
 b[n:n+q]=(lb//pp)*pp+pp-1
 a[n:n+q]=a[n:n+q].clamp(lo,hi);b[n:n+q]=b[n:n+q].clamp(lo,hi)
 da,db,y=digs(a),digs(b),digs(a+b);x=torch.empty(B,25,dtype=torch.long,device=D);x[:,0:16:2]=da[:,:8];x[:,1:16:2]=db[:,:8];x[:,16]=10;x[:,17:]=y[:,:8];return x,y
m=AdditionTransformer().to(D);m.load_state_dict(torch.load('/workspace/final.pt',weights_only=True),strict=False)
opt=torch.optim.AdamW(m.parameters(),lr=2e-6,betas=(.9,.98),weight_decay=.001,fused=True)
for step in range(1,6001):
 x,y=batch(); loss=F.cross_entropy(m(x)[:,16:].reshape(-1,10),y.reshape(-1));opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
 if step%1000==0: print(step,float(loss),flush=True)
torch.save(m.state_dict(),'/workspace/final.pt')
