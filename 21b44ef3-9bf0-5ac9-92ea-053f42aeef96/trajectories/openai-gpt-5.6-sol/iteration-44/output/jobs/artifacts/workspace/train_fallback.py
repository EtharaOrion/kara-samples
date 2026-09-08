import sys
sys.path.insert(0,'/workspace')
from train import *
import torch
from torch import nn
import torch.nn.functional as F

class FullTransformer(nn.Module):
 def __init__(self):
  super().__init__(); self.token=nn.Embedding(11,20); self.pos=nn.Parameter(torch.empty(25,20)); nn.init.normal_(self.pos,std=.02)
  self.q=nn.ModuleList([nn.Linear(20,20,bias=False) for _ in range(2)]); self.k=nn.ModuleList([nn.Linear(20,20,bias=False) for _ in range(2)]); self.v=nn.ModuleList([nn.Linear(20,20,bias=False) for _ in range(2)]); self.o=nn.ModuleList([nn.Linear(20,20,bias=False) for _ in range(2)])
  self.an=nn.ModuleList([nn.LayerNorm(20) for _ in range(2)]); self.fn=nn.ModuleList([nn.LayerNorm(20) for _ in range(2)]); self.fi=nn.ModuleList([nn.Linear(20,4,bias=False) for _ in range(2)]); self.fo=nn.ModuleList([nn.Linear(4,20,bias=False) for _ in range(2)])
  self.final=nn.LayerNorm(20); self.classifier=nn.Linear(20,10,bias=False)
 def forward(self,t):
  B,L=t.shape; x=self.token(t)+self.pos[:L]; mask=torch.ones(L,L,dtype=torch.bool,device=t.device).tril()
  for i in range(2):
   z=self.an[i](x); q=self.q[i](z).view(B,L,4,5).transpose(1,2); k=self.k[i](z).view(B,L,4,5).transpose(1,2); v=self.v[i](z).view(B,L,4,5).transpose(1,2)
   s=(q@k.transpose(-2,-1))*(5**-.5); s=s.masked_fill(~mask,torch.finfo(s.dtype).min); y=(s.softmax(-1)@v).transpose(1,2).reshape(B,L,20)
   x=x+self.o[i](y); x=x+self.fo[i](F.gelu(self.fi[i](self.fn[i](x))))
  return self.classifier(self.final(x))

def main():
 torch.manual_seed(2025); random.seed(2025); torch.backends.cuda.matmul.allow_tf32=True
 m=FullTransformer().cuda(); print('params',sum(p.numel() for p in m.parameters()),flush=True)
 opt=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.01,fused=True)
 train_phase(m,opt,18000,.002,.18,'full','/workspace/full.pt')
 train_phase(m,opt,8000,5e-5,.30,'full_polish','/workspace/full_final.pt')
 print('large',evaluate(m,500000,0),evaluate(m,500000,.8),flush=True); torch.save({'state':m.state_dict()},'/workspace/full_final.pt')
if __name__=='__main__': main()
