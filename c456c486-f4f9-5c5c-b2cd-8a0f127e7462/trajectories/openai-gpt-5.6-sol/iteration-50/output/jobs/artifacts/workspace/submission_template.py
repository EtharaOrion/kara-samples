import math
import torch
from torch import nn
import torch.nn.functional as F

_D=10; _H=2; _FF=8; _PD=2

class _Block(nn.Module):
    def __init__(self):
        super().__init__(); self.n1=nn.LayerNorm(_D); self.qkv=nn.Linear(_D,3*_D); self.proj=nn.Linear(_D,_D)
        self.n2=nn.LayerNorm(_D); self.f1=nn.Linear(_D,_FF); self.f2=nn.Linear(_FF,_D)
    def forward(self,x):
        z=self.n1(x); q,k,v=self.qkv(z).chunk(3,-1); batch,length,_=q.shape; size=_D//_H
        q=q.view(batch,length,_H,size).transpose(1,2); k=k.view(batch,length,_H,size).transpose(1,2); v=v.view(batch,length,_H,size).transpose(1,2)
        scores=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(size)
        mask=torch.ones(length,length,device=x.device,dtype=torch.bool).triu(1)
        weights=torch.softmax(scores.masked_fill(mask,float('-inf')),-1)
        x=x+self.proj(torch.matmul(weights,v).transpose(1,2).reshape(batch,length,_D))
        return x+self.f2(F.gelu(self.f1(self.n2(x))))

class AdditionTransformer(nn.Module):
    def __init__(self):
        super().__init__(); self.ea=nn.Embedding(11,_D); self.eb=nn.Embedding(11,_D); self.eo=nn.Embedding(10,_D)
        self.pos=nn.Parameter(torch.empty(29,_PD)); self.blocks=nn.ModuleList([_Block(),_Block()]); self.norm=nn.LayerNorm(_D); self.head=nn.Linear(_D,10)
    def forward(self,a,b,previous):
        p=F.pad(self.pos,(0,_D-_PD)); source=self.ea(a)+self.eb(b)+p[:14]
        x=torch.cat((source,self.eo(previous)+p[14:14+previous.shape[1]]),1) if previous.shape[1] else source
        for block in self.blocks: x=block(x)
        return self.head(self.norm(x[:,13:]))

_STATE=__STATE__
_SHAPES=__SHAPES__

def build_model():
    model=AdditionTransformer()
    state={name:torch.tensor(values,dtype=torch.float32).reshape(_SHAPES[name]) for name,values in _STATE.items()}
    model.load_state_dict(state); model.eval()
    return model,{'architecture':'decoder-only autoregressive transformer','digits':14,'parameters':sum(p.numel() for p in model.parameters())}

def add(model,a:int,b:int)->int:
    if not (0<=a<100000000000000 and 0<=b<100000000000000): raise ValueError('operands out of range')
    device=next(model.parameters()).device
    ad=[ord(c)-48 for c in reversed(f'{a:014d}')]; bd=[ord(c)-48 for c in reversed(f'{b:014d}')]
    aa=torch.tensor([ad],device=device); bb=torch.tensor([bd],device=device); previous=torch.empty((1,0),dtype=torch.long,device=device)
    with torch.inference_mode():
        for _ in range(15):
            digit=model(aa,bb,previous)[:,-1].argmax(-1,keepdim=True); previous=torch.cat((previous,digit),1)
    chars=[chr(48+int(x)) for x in previous[0].flip(0)]
    return int(''.join(chars))
