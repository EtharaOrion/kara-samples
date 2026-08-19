import math, random
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

D,H,FH,N=10,2,4,15
DEV='cuda'
torch.manual_seed(49); random.seed(49)

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
class Model(nn.Module):
    def __init__(self):
        super().__init__(); self.ea=nn.Embedding(10,D); self.eb=nn.Embedding(10,D); self.sp=nn.Parameter(torch.randn(14,2)*.02); self.enc=Block(); self.qp=nn.Parameter(torch.randn(15,2)*.02)
        self.qn=nn.LayerNorm(D); self.mn=nn.LayerNorm(D); self.cq=nn.Linear(D,D); self.ckv=nn.Linear(D,2*D); self.co=nn.Linear(D,D)
        self.r1=Block(); self.r2=Block(); self.outn=nn.LayerNorm(D); self.head=nn.Linear(D,10)
    def forward(self,a,b):
        B=a.shape[0]; pad=lambda p:F.pad(p,(0,D-p.shape[-1]))
        m=torch.cat((self.ea(a)+pad(self.sp)[None],self.eb(b)+pad(self.sp)[None]),1); m=self.enc(m)
        x=pad(self.qp)[None].expand(B,-1,-1); q=self.cq(self.qn(x)); k,v=self.ckv(self.mn(m)).chunk(2,-1)
        def heads(t,L): return t.view(B,L,H,D//H).transpose(1,2)
        z=(heads(q,15)@heads(k,28).transpose(-2,-1)/math.sqrt(D//H)).softmax(-1)@heads(v,28)
        x=x+self.co(z.transpose(1,2).reshape(B,15,D))
        for _ in range(7): x=self.r1(x,True); x=self.r2(x,True)
        return self.head(self.outn(x))

def sum_digits(a,b):
    # Labels exist only in this external trainer.
    B=a.shape[0]; out=torch.empty(B,15,device=DEV,dtype=torch.long); carry=torch.zeros(B,device=DEV,dtype=torch.long)
    for i in range(15):
        av=a[:,i] if i<14 else 0; bv=b[:,i] if i<14 else 0; s=av+bv+carry; out[:,i]=s.remainder(10); carry=s.div(10,rounding_mode='floor')
    return out

def uniform(B):
    return torch.randint(0,10,(B,14),device=DEV),torch.randint(0,10,(B,14),device=DEV)

def structured(B, mode=None):
    # Broad randomized families, with locations/lengths/directions regenerated every batch.
    a,b=uniform(B); mode=torch.randint(0,10,(B,),device=DEV) if mode is None else torch.full((B,),mode,device=DEV)
    idx=torch.arange(14,device=DEV)[None]; start=torch.randint(0,14,(B,1),device=DEV); length=torch.randint(1,15,(B,1),device=DEV); end=(start+length).clamp(max=14); run=(idx>=start)&(idx<end)
    swap=torch.rand(B,1,device=DEV)<.5
    # exact long carry: 9-run plus a sparse one at its start, either operand order
    m=mode==0; x=torch.where(run,torch.full_like(a,9),a); y=torch.where(run,torch.zeros_like(b),b); y.scatter_(1,start,1); a[m]=torch.where(swap,x,y)[m]; b[m]=torch.where(swap,y,x)[m]
    # near-carry controls: same geometry but start digit 0, 8, or opposing digit 0
    m=mode==1; x=torch.where(run,torch.full_like(a,9),a); y=torch.where(run,torch.zeros_like(b),b); y.scatter_(1,start,torch.randint(0,2,(B,1),device=DEV)); x.scatter_(1,start,torch.randint(7,10,(B,1),device=DEV)); a[m]=torch.where(swap,x,y)[m]; b[m]=torch.where(swap,y,x)[m]
    # complementary columns with random sum 9 or 10
    m=mode==2; x=torch.randint(0,10,a.shape,device=DEV); target=torch.randint(9,11,(B,1),device=DEV); y=(target-x).clamp(0,9); a[m]=torch.where(run,x,a)[m]; b[m]=torch.where(run,y,b)[m]
    # sparse powers and unequal lengths / leading-zero transitions
    m=mode==3; x=torch.zeros_like(a); y=torch.zeros_like(b); x.scatter_(1,start,torch.randint(1,10,(B,1),device=DEV)); pos=torch.randint(0,14,(B,1),device=DEV); y.scatter_(1,pos,torch.randint(1,10,(B,1),device=DEV)); a[m]=torch.where(swap,x,y)[m]; b[m]=torch.where(swap,y,x)[m]
    # repeated and alternating operands
    m=mode==4; da=torch.randint(0,10,(B,1),device=DEV); db=torch.randint(0,10,(B,1),device=DEV); x=da.expand(-1,14); y=torch.where((idx&1).bool(),db,9-db); a[m]=x[m]; b[m]=y[m]
    # maximum boundary and neighbors: all 9s plus sparse/random short value
    m=mode==5; x=torch.full_like(a,9); y=torch.zeros_like(b); y.scatter_(1,start,torch.randint(0,10,(B,1),device=DEV)); perturb=torch.randint(0,14,(B,1),device=DEV); x.scatter_(1,perturb,torch.randint(7,10,(B,1),device=DEV)); a[m]=torch.where(swap,x,y)[m]; b[m]=torch.where(swap,y,x)[m]
    # isolated equal columns, including 5+5 and 9+9 at every height
    m=mode==6; x=torch.zeros_like(a); y=torch.zeros_like(b); d=torch.where(torch.rand(B,1,device=DEV)<.5,torch.full((B,1),5,device=DEV),torch.full((B,1),9,device=DEV)); x.scatter_(1,start,d); y.scatter_(1,start,d); a[m]=x[m]; b[m]=y[m]
    # blockwise random with zeros above randomized significant length
    m=mode==7; sig=torch.randint(1,15,(B,1),device=DEV); low=idx<sig; x=torch.where(low,a,torch.zeros_like(a)); y=torch.where(idx<torch.randint(1,15,(B,1),device=DEV),b,torch.zeros_like(b)); a[m]=x[m]; b[m]=y[m]
    # dense carry/noncarry boundaries with each column total 8..11
    m=mode==8; x=torch.randint(0,10,a.shape,device=DEV); totals=torch.randint(8,12,a.shape,device=DEV); y=(totals-x).clamp(0,9); a[m]=x[m]; b[m]=y[m]
    # all zeros, one zero operand, swaps and random neighbor patterns
    m=mode==9; x=torch.zeros_like(a); y=torch.where(torch.rand(B,1,device=DEV)<.3,torch.zeros_like(b),b); a[m]=torch.where(swap,x,y)[m]; b[m]=torch.where(swap,y,x)[m]
    return a,b

def batch(B, structured_fraction):
    a,b=uniform(B); n=int(B*structured_fraction)
    if n: a[:n],b[:n]=structured(n)
    return a,b,sum_digits(a,b)

@torch.no_grad()
def evaluate(model, batches, B=4096, structured_eval=False):
    model.eval(); errors=0; total=0
    for j in range(batches):
        a,b=structured(B) if structured_eval else uniform(B); y=sum_digits(a,b); pred=model(a,b).argmax(-1); errors+=(pred!=y).any(1).sum().item(); total+=B
    model.train(); return errors,total

def export(model):
    state={k:v.detach().cpu().float().tolist() for k,v in model.state_dict().items()}
    source=Path('/workspace/inference_template.py').read_text(); source=source.replace('__STATE__',repr(state))
    Path('/workspace/submission.py').write_text(source)

m=Model().to(DEV); print('parameters',sum(p.numel() for p in m.parameters()),flush=True)
opt=torch.optim.AdamW(m.parameters(),lr=3e-3,weight_decay=.002)
# Uniform foundation, increasingly broad mixed refinement, then low-rate edge-safe calibration.
phases=[(10000,3e-3,0.0),(14000,1e-3,.25),(14000,3e-4,.40),(12000,1e-4,.50),(12000,3e-5,.50),(12000,1e-5,.50)]
step=0; best=None; bestscore=10**9
for count,lr,sf in phases:
    for g in opt.param_groups:g['lr']=lr
    for _ in range(count):
        a,b,y=batch(4096,sf); opt.zero_grad(set_to_none=True); loss=F.cross_entropy(m(a,b).reshape(-1,10),y.reshape(-1)); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); step+=1
        if step%2000==0: print(step,float(loss),flush=True)
    er,n=evaluate(m,32); es,ns=evaluate(m,32,structured_eval=True); score=er*4+es
    print('VALID',step,er,n,es,ns,flush=True)
    if score<=bestscore: bestscore=score; best={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}; torch.save(best,'/workspace/best.pt')
    export(m)
if best is not None:m.load_state_dict(best)
export(m)
er,n=evaluate(m,128); es,ns=evaluate(m,128,structured_eval=True); print('FINAL',er,n,es,ns,flush=True)
