import sys
from pathlib import Path
sys.path.append('/usr/local/lib/python3.11/dist-packages'); sys.path.insert(0,'/workspace')
import torch
from torch import nn
class A(nn.Module):
 def __init__(self):
  super().__init__(); self.q=nn.ModuleList([nn.Linear(20,20,bias=False) for _ in range(2)]); self.o=nn.ModuleList([nn.Linear(20,20,bias=False) for _ in range(2)]); self.k=nn.Linear(20,5,bias=False); self.v=nn.Linear(20,5,bias=False)
 def forward(self,x,l):
  B,L,_=x.shape; q=self.q[l](x).view(B,L,4,5).transpose(1,2); k=self.k(x); v=self.v(x); y=torch.nn.functional.scaled_dot_product_attention(q,k[:,None],v[:,None],is_causal=True,enable_gqa=True); return self.o[l](y.transpose(1,2).reshape(B,L,20))
class AdditionTransformer(nn.Module):
 def __init__(self):
  super().__init__(); self.token=nn.Embedding(11,20); self.pos_a=nn.Parameter(torch.empty(25,2)); self.pos_b=nn.Parameter(torch.empty(2,20)); self.attn_norm=nn.LayerNorm(20); self.attn=A(); self.ff_norm=nn.LayerNorm(20); self.ff1=nn.Linear(20,2,bias=False); self.ff2=nn.Linear(2,20,bias=False); self.final_norm=nn.LayerNorm(20); self.head=nn.Linear(20,10,bias=False)
 def forward(self,t):
  x=self.token(t)+(self.pos_a@self.pos_b)[:t.shape[1]]
  for l in range(2): x=x+self.attn(self.attn_norm(x),l); x=x+self.ff2(torch.nn.functional.gelu(self.ff1(self.ff_norm(x))))
  return self.head(self.final_norm(x))
