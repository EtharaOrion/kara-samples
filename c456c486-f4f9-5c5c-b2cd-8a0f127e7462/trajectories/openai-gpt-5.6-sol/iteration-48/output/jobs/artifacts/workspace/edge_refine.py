import math, os, random, time
import torch
from torch import nn
import torch.nn.functional as F

D=10; H=2; FF=4; N=14; OUT=15; ROUNDS=7
POW=torch.tensor([10**i for i in range(15)],device='cuda',dtype=torch.long)

class Attn(nn.Module):
    def __init__(self,cross=False):
        super().__init__(); self.cross=cross
        self.q=nn.Linear(D,D); self.k=nn.Linear(D,D); self.v=nn.Linear(D,D); self.o=nn.Linear(D,D)
    def forward(self,x,kv=None,causal=False):
        z=x if kv is None else kv; B,T,_=x.shape; S=z.shape[1]
        q=self.q(x).view(B,T,H,D//H).transpose(1,2)
        k=self.k(z).view(B,S,H,D//H).transpose(1,2)
        v=self.v(z).view(B,S,H,D//H).transpose(1,2)
        s=q@k.transpose(-2,-1)/math.sqrt(D//H)
        if causal: s=s.masked_fill(torch.arange(S,device=x.device)[None,:]>torch.arange(T,device=x.device)[:,None],-1e4)
        return self.o((s.softmax(-1)@v).transpose(1,2).reshape(B,T,D))
class Block(nn.Module):
    def __init__(self,ff=True):
        super().__init__(); self.n1=nn.LayerNorm(D); self.a=Attn(); self.n2=nn.LayerNorm(D) if ff else None
        self.f=nn.Sequential(nn.Linear(D,FF),nn.GELU(),nn.Linear(FF,D)) if ff else None
    def forward(self,x,causal=False):
        x=x+self.a(self.n1(x),causal=causal)
        return x+self.f(self.n2(x)) if self.f else x
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.ea=nn.Embedding(10,D); self.eb=nn.Embedding(10,D)
        self.sp=nn.Parameter(torch.randn(N,2)*.02); self.enc=Block(True)
        self.qp=nn.Parameter(torch.randn(OUT,2)*.02)
        self.qn=nn.LayerNorm(D); self.sn=nn.LayerNorm(D); self.cross=Attn(True)
        self.ref=nn.ModuleList([Block(True),Block(True)])
        self.final=nn.LayerNorm(D); self.head=nn.Linear(D,10)
    @staticmethod
    def pad2(x): return F.pad(x,(0,D-2))
    def forward(self,a,b):
        src=self.ea(a)+self.eb(b)+self.pad2(self.sp)[None]
        src=self.enc(src)
        q=self.pad2(self.qp)[None].expand(a.shape[0],-1,-1)
        q=q+self.cross(self.qn(q),self.sn(src))
        for _ in range(ROUNDS):
            q=self.ref[0](q,True); q=self.ref[1](q,True)
        return self.head(self.final(q))

def digits(x,n): return (x[:,None]//POW[:n])%10

def batch(bs, structured):
    lim=10**14
    a=torch.randint(0,lim,(bs,),device='cuda'); b=torch.randint(0,lim,(bs,),device='cuda')
    if structured:
        m=bs//2; typ=torch.randint(0,5,(m,),device='cuda')
        # shifted runs of nines plus sparse increments; matched near-carries
        st=torch.randint(0,14,(m,),device='cuda'); ln=torch.randint(1,15,(m,),device='cuda'); ln=torch.minimum(ln,14-st)
        run=(POW[st+ln]-POW[st])
        inc=POW[st]
        sa=run.clone(); sb=inc.clone()
        near=typ==1; sa[near]-=inc[near]
        # isolated equal boundaries and repeated digits
        iso=(typ==2)|(typ==3); vals=torch.where(typ==2,torch.full_like(st,5),torch.full_like(st,9))
        sa[iso]=vals[iso]*inc[iso]; sb[iso]=vals[iso]*inc[iso]
        rep=typ==4; ds=torch.randint(1,10,(m,),device='cuda'); whole=(POW[14]-1)//9
        sa[rep]=10**14-1; sb[rep]=inc[rep]
        a[:m]=sa; b[:m]=sb
    return digits(a,N),digits(b,N),digits(a+b,OUT)

@torch.no_grad()
def validate(model,n=131072,structured=False,bs=8192):
    model.eval(); err=0
    for _ in range((n+bs-1)//bs):
        aa,bb,y=batch(min(bs,n),structured); n-=aa.shape[0]
        err+=(model(aa,bb).argmax(-1).ne(y).any(-1)).sum().item()
    model.train(); return err

def export(model,path='/workspace/checkpoint.pt'):
    torch.save(model.state_dict(),path)

model=Model().cuda(); model.load_state_dict(torch.load('/workspace/checkpoint.pt',weights_only=True)); print('params',sum(p.numel() for p in model.parameters()),flush=True)
opt=torch.optim.AdamW(model.parameters(),lr=3e-3,weight_decay=.003)
# compile after construction
## eager
start=time.time(); best=10**9
phases=[(12000,1e-5,True)]
step=0
for count,lr,mix in phases:
    for g in opt.param_groups:g['lr']=lr
    for i in range(count):
        aa,bb,y=batch(4096,mix and (i%2==0))
        loss=F.cross_entropy(model(aa,bb).reshape(-1,10),y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); step+=1
        if step%4000==0:
            er=validate(model,65536,False); es=validate(model,32768,True)
            print(step,lr,float(loss),er,es,'sec',int(time.time()-start),flush=True)
            score=er+es*2
            if score<=best: best=score; export(model)
export(model,'/workspace/final.pt')
print('FINAL',validate(model,524288,False),validate(model,262144,True),flush=True)
