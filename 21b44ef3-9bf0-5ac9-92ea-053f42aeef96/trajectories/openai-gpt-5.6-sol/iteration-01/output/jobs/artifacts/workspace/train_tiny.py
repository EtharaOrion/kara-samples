import math, sys
sys.path=[p for p in sys.path if 'openhands-venv' not in p]; sys.path += ['/usr/local/lib/python3.11/dist-packages','/usr/lib/python3/dist-packages']
import torch
from torch import nn
import torch.nn.functional as F
class Model(nn.Module):
 def __init__(self,d):
  super().__init__(); self.embedding=nn.Embedding(12,d); self.qkv=nn.Linear(d,3*d,bias=False); self.projection=nn.Linear(d,d,bias=False); self.output=nn.Linear(d,1)
 def forward(self,t):
  x=self.embedding(t); q,k,v=self.qkv(x).chunk(3,-1); a=F.softmax(q@k.transpose(-2,-1)/math.sqrt(q.shape[-1]),-1); x=x+self.projection(a@v); return self.output(x[:,2]).squeeze(-1)
def run(d,seed,steps):
 torch.manual_seed(seed); dev='cuda'; x=torch.tensor([(a,b,10+c) for c in range(2) for a in range(10) for b in range(10)],device=dev); y=torch.tensor([a+b+c for c in range(2) for a in range(10) for b in range(10)],device=dev,dtype=torch.float); m=Model(d).to(dev); o=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=0)
 best=99.; beststate=None
 for s in range(steps):
  o.zero_grad(set_to_none=True); p=m(x); loss=F.mse_loss(p,y); loss.backward(); o.step()
  if s%250==0:
   with torch.no_grad(): err=(m(x)-y).abs(); mx=err.max().item()
   if mx<best: best=mx; beststate={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
   if s%2000==0: print(s,loss.item(),mx,best,flush=True)
 torch.save(beststate,f'/workspace/tiny_{d}_{seed}.pt'); print('params',sum(p.numel() for p in m.parameters()),'bestmax',best)
if __name__=='__main__':run(*map(int,sys.argv[1:]))
