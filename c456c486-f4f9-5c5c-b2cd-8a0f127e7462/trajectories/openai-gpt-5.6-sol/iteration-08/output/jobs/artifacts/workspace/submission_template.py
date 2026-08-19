import math
import torch
from torch import nn
from torch.nn import functional as F

D=16; H=2; HD=8; M=32

class BiaslessLN(nn.Module):
    def __init__(self,d):
        super().__init__(); self.weight=nn.Parameter(torch.ones(d))
    def forward(self,x): return F.layer_norm(x,(x.shape[-1],),self.weight,None,1e-5)

class Block(nn.Module):
    def __init__(self,bias):
        super().__init__(); self.n1=BiaslessLN(D); self.qkv=nn.Linear(D,48,bias=False); self.proj=nn.Linear(D,D,bias=False)
        self.n2=BiaslessLN(D); self.fc1=nn.Linear(D,M,bias=False); self.fc2=nn.Linear(M,D,bias=False); self.shared_bias=bias
    def forward(self,x):
        n=x.shape[1]; z=self.n1(x); q,k,v=self.qkv(z).chunk(3,-1)
        q=q.view(1,n,H,HD).transpose(1,2); k=k.view(1,n,H,HD).transpose(1,2); v=v.view(1,n,H,HD).transpose(1,2)
        score=q@k.transpose(-2,-1)/math.sqrt(HD); ix=torch.arange(n)
        allowed=(ix[:,None]>=ix[None,:]) & ((ix[:,None]-ix[None,:])<=5)
        y=(score.masked_fill(~allowed,-1e4).softmax(-1)@v).transpose(1,2).reshape(1,n,D)
        x=x+self.proj(y)
        return x+self.fc2(F.gelu(self.fc1(self.n2(x))+self.shared_bias))

class AdditionTransformer(nn.Module):
    def __init__(self):
        super().__init__(); self.digit=nn.Embedding(10,D); self.role=nn.Embedding(3,D); self.mb=nn.Parameter(torch.zeros(M))
        self.blocks=nn.ModuleList([Block(self.mb),Block(self.mb)]); self.norm=BiaslessLN(D); self.head=nn.Linear(D,10,bias=False)
    def forward(self,tokens):
        n=tokens.shape[1]; x=self.digit(tokens)+self.role(torch.arange(n)%3)
        for block in self.blocks: x=block(x)
        return self.head(self.norm(x))

_STATE=__STATE__

def build_model():
    model=AdditionTransformer(); current=model.state_dict()
    with torch.no_grad():
        for name,flat in _STATE.items(): current[name].copy_(torch.tensor(flat).view_as(current[name]))
    model.eval()
    return model,{'architecture':'two-block local causal addition transformer','parameters':sum(p.numel() for p in model.parameters())}

def add(model,a:int,b:int)->int:
    tokens=torch.zeros((1,45),dtype=torch.long)
    x=a; y=b
    for i in range(14):
        tokens[0,3*i]=x%10; tokens[0,3*i+1]=y%10; x//=10; y//=10
    for p in range(2,45,3): tokens[0,p]=model(tokens[:,:p])[:,-1].argmax()
    digits=tokens[0,2::3].tolist(); value=0
    for digit in reversed(digits): value=value*10+digit
    return value
