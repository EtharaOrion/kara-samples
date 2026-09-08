import math, random, sys, time
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0, '/workspace')
import train as data
ROOT=Path('/workspace'); DEVICE='cuda'; BATCH=8192

class Model(nn.Module):
    def __init__(self):
        super().__init__(); d=20
        self.token=nn.Embedding(11,d); self.pos=nn.Parameter(torch.empty(25,d))
        self.norm1=nn.ModuleList([nn.LayerNorm(d) for _ in range(2)])
        self.q=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(2)])
        self.k=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(2)])
        self.v=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(2)])
        self.o=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(2)])
        self.norm2=nn.ModuleList([nn.LayerNorm(d) for _ in range(2)])
        self.f1=nn.ModuleList([nn.Linear(d,4,bias=False) for _ in range(2)])
        self.f2=nn.ModuleList([nn.Linear(4,d,bias=False) for _ in range(2)])
        self.final=nn.LayerNorm(d); self.head=nn.Linear(d,10,bias=False)
        self.register_buffer('mask',torch.triu(torch.ones(25,25,dtype=torch.bool),1),persistent=False)
        nn.init.normal_(self.pos,std=.2)
    def forward(self,t):
        n=t.shape[1]; x=self.token(t)+self.pos[:n]
        for i in range(2):
            z=self.norm1[i](x); q=self.q[i](z).view(-1,n,4,5).transpose(1,2); k=self.k[i](z).view(-1,n,4,5).transpose(1,2); v=self.v[i](z).view(-1,n,4,5).transpose(1,2)
            a=(q@k.transpose(-2,-1)/math.sqrt(5)).masked_fill(self.mask[:n,:n],-torch.inf).softmax(-1)
            x=x+self.o[i]((a@v).transpose(1,2).reshape(-1,n,20)); x=x+self.f2[i](torch.nn.functional.gelu(self.f1[i](self.norm2[i](x))))
        return self.head(self.final(x))

def acc(m,n=20000,structured=False):
    m.eval(); good=0
    with torch.no_grad():
      for s in range(0,n,5000):
        z=min(5000,n-s); a,b=data.structured(z) if structured else (torch.randint(data.LOW,data.HIGH+1,(z,),device=DEVICE),torch.randint(data.LOW,data.HIGH+1,(z,),device=DEVICE))
        ad,bd=data.digits(a),data.digits(b); seq=torch.cat((torch.stack((ad,bd),2).reshape(z,16),torch.full((z,1),10,device=DEVICE)),1); out=[]
        for _ in range(9):
          d=m(seq)[:,-1].argmax(1); out.append(d); seq=torch.cat((seq,d[:,None]),1)
        good+=(torch.stack(out,1)==data.digits(a+b,9)).all(1).sum().item()
    m.train(); return good/n

def export(m):
    vals=[]
    for p in m.parameters(): vals.append('['+','.join(format(x,'.9g') for x in p.detach().float().cpu().reshape(-1).tolist())+']')
    weights='_TRAINED_STATE = [\n'+',\n'.join(vals)+'\n]'
    source='''import math\nimport torch\nfrom torch import nn\n\nclass AdditionTransformer(nn.Module):\n    def __init__(self):\n        super().__init__(); d=20\n        self.token=nn.Embedding(11,d); self.pos=nn.Parameter(torch.empty(25,d))\n        self.norm1=nn.ModuleList([nn.LayerNorm(d) for _ in range(2)])\n        self.q=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(2)])\n        self.k=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(2)])\n        self.v=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(2)])\n        self.o=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(2)])\n        self.norm2=nn.ModuleList([nn.LayerNorm(d) for _ in range(2)])\n        self.f1=nn.ModuleList([nn.Linear(d,4,bias=False) for _ in range(2)])\n        self.f2=nn.ModuleList([nn.Linear(4,d,bias=False) for _ in range(2)])\n        self.final=nn.LayerNorm(d); self.head=nn.Linear(d,10,bias=False)\n        self.register_buffer("mask",torch.triu(torch.ones(25,25,dtype=torch.bool),1),persistent=False)\n    def forward(self,t):\n        n=t.shape[1]; x=self.token(t)+self.pos[:n]\n        for i in range(2):\n            z=self.norm1[i](x); q=self.q[i](z).view(-1,n,4,5).transpose(1,2); k=self.k[i](z).view(-1,n,4,5).transpose(1,2); v=self.v[i](z).view(-1,n,4,5).transpose(1,2)\n            a=(q@k.transpose(-2,-1)/math.sqrt(5)).masked_fill(self.mask[:n,:n],-torch.inf).softmax(-1)\n            x=x+self.o[i]((a@v).transpose(1,2).reshape(-1,n,20)); x=x+self.f2[i](torch.nn.functional.gelu(self.f1[i](self.norm2[i](x))))\n        return self.head(self.final(x))\n\n'''+weights+'''\n\ndef build_model():\n    model=AdditionTransformer()\n    with torch.no_grad():\n        for p,v in zip(model.parameters(),_TRAINED_STATE): p.copy_(torch.tensor(v,dtype=p.dtype).reshape(p.shape))\n    model.eval(); return model,{"architecture":"two-layer causal autoregressive digit transformer","digit_order":"least-significant-first"}\n\ndef add(model,a:int,b:int)->int:\n    aa=str(a)[::-1]; bb=str(b)[::-1]; tokens=[]\n    for x,y in zip(aa,bb): tokens.extend((ord(x)-48,ord(y)-48))\n    tokens.append(10); device=next(model.parameters()).device\n    with torch.no_grad():\n        for _ in range(9):\n            t=torch.tensor([tokens],dtype=torch.long,device=device); tokens.append(int(model(t)[0,-1].argmax().item()))\n    return int("".join(str(x) for x in tokens[-9:][::-1]))\n'''
    (ROOT/'submission.py').write_text(source)

def main():
 torch.manual_seed(2025); random.seed(2025); torch.backends.cuda.matmul.allow_tf32=True
 m=Model().cuda(); opt=torch.optim.AdamW(m.parameters(),lr=2e-3,betas=(.9,.98),weight_decay=.01,fused=True)
 for step in range(1,30001):
  seq,tgt=data.batch(BATCH,.18 if step<22000 else .35); opt.zero_grad(set_to_none=True); loss=torch.nn.functional.cross_entropy(m(seq)[:,16:25].reshape(-1,10),tgt.reshape(-1)); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1); opt.step()
  if step in (18000,):
   for g in opt.param_groups:g['lr']=8e-5
  if step in (26000,):
   for g in opt.param_groups:g['lr']=2e-5
  if step%1000==0: print(step,loss.item(),opt.param_groups[0]['lr'],flush=True)
  if step%5000==0:
   au,ast=acc(m),acc(m,structured=True); print('AR',au,ast,flush=True); torch.save(m.state_dict(),ROOT/f'fallback_{step}.pt')
   if step>=15000 and min(au,ast)>=.9999: export(m)
 torch.save(m.state_dict(),ROOT/'fallback_final.pt'); print('FINAL',acc(m,200000),acc(m,200000,True),flush=True); export(m)
if __name__=='__main__':main()
