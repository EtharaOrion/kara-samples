import math
import torch
from torch import nn
import torch.nn.functional as F

D,H,FH=10,2,4
class Block(nn.Module):
    def __init__(self):
        super().__init__(); self.n1=nn.LayerNorm(D); self.qkv=nn.Linear(D,3*D); self.proj=nn.Linear(D,D); self.n2=nn.LayerNorm(D); self.f1=nn.Linear(D,FH); self.f2=nn.Linear(FH,D)
    def forward(self,x,causal=False):
        z=self.n1(x); q,k,v=self.qkv(z).chunk(3,-1); B,L,_=q.shape
        q=q.view(B,L,H,D//H).transpose(1,2); k=k.view(B,L,H,D//H).transpose(1,2); v=v.view(B,L,H,D//H).transpose(1,2)
        s=q@k.transpose(-2,-1)/math.sqrt(D//H)
        if causal: s=s.masked_fill(torch.triu(torch.ones(L,L,device=x.device,dtype=torch.bool),1),-1e4)
        x=x+self.proj((s.softmax(-1)@v).transpose(1,2).reshape(B,L,D))
        return x+self.f2(F.gelu(self.f1(self.n2(x))))
class AdditionTransformer(nn.Module):
    def __init__(self):
        super().__init__(); self.ea=nn.Embedding(10,D); self.eb=nn.Embedding(10,D); self.sp=nn.Parameter(torch.empty(14,2)); self.enc=Block(); self.qp=nn.Parameter(torch.empty(15,2))
        self.qn=nn.LayerNorm(D); self.mn=nn.LayerNorm(D); self.cq=nn.Linear(D,D); self.ckv=nn.Linear(D,2*D); self.co=nn.Linear(D,D); self.r1=Block(); self.r2=Block(); self.outn=nn.LayerNorm(D); self.head=nn.Linear(D,10)
    def forward(self,a,b):
        B=a.shape[0]; pad=lambda p:F.pad(p,(0,D-p.shape[-1])); m=torch.cat((self.ea(a)+pad(self.sp)[None],self.eb(b)+pad(self.sp)[None]),1); m=self.enc(m)
        x=pad(self.qp)[None].expand(B,-1,-1); q=self.cq(self.qn(x)); k,v=self.ckv(self.mn(m)).chunk(2,-1)
        def heads(t,L):return t.view(B,L,H,D//H).transpose(1,2)
        z=(heads(q,15)@heads(k,28).transpose(-2,-1)/math.sqrt(D//H)).softmax(-1)@heads(v,28); x=x+self.co(z.transpose(1,2).reshape(B,15,D))
        for _ in range(7):x=self.r1(x,True);x=self.r2(x,True)
        return self.head(self.outn(x))
_STATE=__STATE__
def build_model():
    model=AdditionTransformer(); model.load_state_dict({k:torch.tensor(v) for k,v in _STATE.items()}); model.eval(); return model,{'architecture':'bidirectional encoder with learned-query cross-attention','digits':14}
def add(model,a:int,b:int)->int:
    def digits(value):
        text=str(value).zfill(14)
        return [ord(c)-48 for c in reversed(text)]
    da=torch.tensor([digits(a)],dtype=torch.long); db=torch.tensor([digits(b)],dtype=torch.long)
    with torch.no_grad(): out=model(da,db).argmax(-1)[0].tolist()
    return int(''.join(str(d) for d in reversed(out)))
