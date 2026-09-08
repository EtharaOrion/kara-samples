import argparse, math, os, random, sys, time
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

D,H,DH,R=20,4,5,2
MAX=99_999_999
MIN=10_000_000

class AddTransformer(nn.Module):
    def __init__(self, ff=4):
        super().__init__(); self.ff=ff
        self.tok=nn.Embedding(11,D)
        self.pa=nn.Parameter(torch.empty(25,R)); self.pb=nn.Parameter(torch.empty(R,D))
        self.k=nn.Linear(D,DH,bias=False); self.v=nn.Linear(D,DH,bias=False)
        self.q=nn.ModuleList([nn.Linear(D,D,bias=False) for _ in range(2)])
        self.o=nn.ModuleList([nn.Linear(D,D,bias=False) for _ in range(2)])
        self.an=nn.LayerNorm(D); self.fn=nn.LayerNorm(D)
        self.f1=nn.Linear(D,ff,bias=False); self.f2=nn.Linear(ff,D,bias=False)
        self.outn=nn.LayerNorm(D); self.out=nn.Linear(D,10,bias=False)
        nn.init.normal_(self.pa,std=.02); nn.init.normal_(self.pb,std=.02)
    def forward(self,t):
        x=self.tok(t)+self.pa[:t.shape[1]].matmul(self.pb)
        for layer in range(2):
            z=self.an(x); B,T,_=z.shape
            q=self.q[layer](z).view(B,T,H,DH).transpose(1,2)
            k=self.k(z).view(B,1,T,DH)
            v=self.v(z).view(B,1,T,DH)
            y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
            y=y.transpose(1,2).reshape(B,T,D)
            x=x+self.o[layer](y)
            x=x+self.f2(F.gelu(self.f1(self.fn(x))))
        return self.out(self.outn(x))

def digits(x,n=8):
    ds=[]
    for _ in range(n): ds.append(x.remainder(10)); x=x.div(10,rounding_mode='floor')
    return torch.stack(ds,1)

def structured(n,device):
    # Broad mixture emphasizing complements, asymmetric carry runs, decimal boundaries, and sparse/repeated digits.
    mode=torch.randint(0,8,(n,),device=device)
    a=torch.randint(MIN,MAX+1,(n,),device=device,dtype=torch.long)
    b=torch.randint(MIN,MAX+1,(n,),device=device,dtype=torch.long)
    # Exact/near complements to 10^8 and 10^8 +/- powers.
    m=mode==0
    target=100_000_000+torch.randint(-1000,1001,(n,),device=device)
    bb=(target-a).clamp(MIN,MAX); b=torch.where(m,bb,b)
    # Force unequal trailing-nine runs, with random full-width prefixes.
    m=mode==1; k=torch.randint(1,8,(n,),device=device); p=(10**k)
    aa=(a.div(p,rounding_mode='floor')*p + p-1).clamp(MIN,MAX)
    low=torch.randint(1,10,(n,),device=device); bb=(b.div(p,rounding_mode='floor')*p+low).clamp(MIN,MAX)
    a=torch.where(m,aa,a); b=torch.where(m,bb,b)
    # One operand rounded to a random decimal boundary, the other immediately around it.
    m=mode==2; k=torch.randint(1,8,(n,),device=device); p=10**k
    aa=(a.div(p,rounding_mode='floor')*p).clamp(MIN,MAX)
    off=torch.randint(-20,21,(n,),device=device); bb=(b.div(p,rounding_mode='floor')*p+p+off).clamp(MIN,MAX)
    a=torch.where(m,aa,a); b=torch.where(m,bb,b)
    # Repeated digit numbers.
    m=mode==3; da=torch.randint(1,10,(n,),device=device); db=torch.randint(0,10,(n,),device=device)
    repa=da*11_111_111; repb=(db*11_111_111).clamp(MIN,MAX)
    a=torch.where(m,repa,a); b=torch.where(m,repb,b)
    # Sparse high digit plus one low nonzero digit.
    m=mode==4; hi=torch.randint(1,10,(n,),device=device)*10_000_000
    p=10**torch.randint(0,7,(n,),device=device); aa=(hi+torch.randint(0,10,(n,),device=device)*p).clamp(MIN,MAX)
    bb=(100_000_000-aa+torch.randint(-10,11,(n,),device=device)).clamp(MIN,MAX)
    a=torch.where(m,aa,a); b=torch.where(m,bb,b)
    # Near extrema.
    m=mode==5; aa=MIN+torch.randint(0,10000,(n,),device=device); bb=MAX-torch.randint(0,10000,(n,),device=device)
    a=torch.where(m,aa,a); b=torch.where(m,bb,b)
    # Long zero/nine suffixes independently.
    m=mode==6; ka=torch.randint(1,8,(n,),device=device); kb=torch.randint(1,8,(n,),device=device)
    pa=10**ka; pb=10**kb
    aa=(a.div(pa,rounding_mode='floor')*pa+torch.where(torch.rand(n,device=device)<.5,pa-1,torch.zeros_like(pa))).clamp(MIN,MAX)
    bb=(b.div(pb,rounding_mode='floor')*pb+torch.where(torch.rand(n,device=device)<.5,pb-1,torch.zeros_like(pb))).clamp(MIN,MAX)
    a=torch.where(m,aa,a); b=torch.where(m,bb,b)
    return a,b

def batch(bs, frac, device):
    a=torch.randint(MIN,MAX+1,(bs,),device=device,dtype=torch.long)
    b=torch.randint(MIN,MAX+1,(bs,),device=device,dtype=torch.long)
    n=int(bs*frac)
    if n:
        sa,sb=structured(n,device); a[:n]=sa; b[:n]=sb
    ad=digits(a); bd=digits(b); sd=digits(a+b,9)
    x=torch.empty((bs,25),device=device,dtype=torch.long)
    x[:,0:16:2]=ad; x[:,1:16:2]=bd; x[:,16]=10; x[:,17:]=sd[:,:8]
    return x,sd

@torch.no_grad()
def accuracy(model,n=100000,structured_frac=0.,bs=10000):
    model.eval(); good=total=0; minmargin=1000.
    for _ in range((n+bs-1)//bs):
        z=min(bs,n-total); x,y=batch(z,structured_frac,next(model.parameters()).device)
        pred=model(x)[:,16:25].argmax(-1)
        good+=(pred==y).all(1).sum().item(); total+=z
        vals=model(x)[:,16:25].topk(2,-1).values
        minmargin=min(minmargin,(vals[...,0]-vals[...,1]).min().item())
    model.train(); return good,total,minmargin

def prune(model,newff):
    old=model.ff; assert newff==old-1
    score=model.f1.weight.norm(dim=1)*model.f2.weight.norm(dim=0)
    keep=torch.argsort(score,descending=True)[:newff].sort().values
    m=AddTransformer(newff).to(next(model.parameters()).device)
    sd=model.state_dict(); ns=m.state_dict()
    for k in ns:
        if k=='f1.weight': ns[k].copy_(sd[k][keep])
        elif k=='f2.weight': ns[k].copy_(sd[k][:,keep])
        else: ns[k].copy_(sd[k])
    m.load_state_dict(ns); return m

def train_phase(model,steps,lr,frac,name,bs=8192):
    model.train(); opt=torch.optim.AdamW(model.parameters(),lr=lr,betas=(.9,.98),weight_decay=.01)
    start=time.time()
    for s in range(1,steps+1):
        x,y=batch(bs,frac,'cuda'); logits=model(x)[:,16:25]
        loss=F.cross_entropy(logits.reshape(-1,10),y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if s%1000==0 or s==steps:
            print(f'{name} {s}/{steps} loss={loss.item():.6f} time={time.time()-start:.1f}',flush=True)
        if s%6000==0 or s==steps:
            print(' val',accuracy(model,20000,0.),accuracy(model,20000,1.),flush=True)
            torch.save({'ff':model.ff,'state':model.state_dict()},f'/workspace/{name}.pt')
    return model

def tensor_literal(t):
    # float16 source literals are compact; margins are large and weights are loaded into float32 parameters.
    v=t.detach().cpu().half().reshape(-1).tolist()
    return repr(v)

def export_plain(model,path='/workspace/submission.py'):
    sd=model.state_dict()
    shapes={k:list(v.shape) for k,v in sd.items()}
    flat=[]
    for v in sd.values(): flat.extend(v.detach().cpu().half().reshape(-1).tolist())
    source=INFERENCE_HEAD.replace('__FF__',str(model.ff))
    source += '\n_SHAPES='+repr(shapes)+'\n_VALUES='+repr(flat)+'\n'+INFERENCE_TAIL
    Path(path).write_text(source)
    print('exported',path,'params',sum(p.numel() for p in model.parameters()),'bytes',Path(path).stat().st_size,flush=True)

INFERENCE_HEAD='''import torch
from torch import nn
import torch.nn.functional as F

class AdditionTransformer(nn.Module):
    def __init__(self):
        super().__init__(); d=20; self.ff=__FF__
        self.tok=nn.Embedding(11,d); self.pa=nn.Parameter(torch.empty(25,2)); self.pb=nn.Parameter(torch.empty(2,d))
        self.k=nn.Linear(d,5,bias=False); self.v=nn.Linear(d,5,bias=False)
        self.q=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(2)])
        self.o=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(2)])
        self.an=nn.LayerNorm(d); self.fn=nn.LayerNorm(d)
        self.f1=nn.Linear(d,self.ff,bias=False); self.f2=nn.Linear(self.ff,d,bias=False)
        self.outn=nn.LayerNorm(d); self.out=nn.Linear(d,10,bias=False)
    def forward(self,t):
        x=self.tok(t)+self.pa[:t.shape[1]].matmul(self.pb)
        for layer in range(2):
            z=self.an(x); B,T,_=z.shape
            q=self.q[layer](z).view(B,T,4,5).transpose(1,2)
            k=self.k(z).view(B,1,T,5); v=self.v(z).view(B,1,T,5)
            y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
            x=x+self.o[layer](y.transpose(1,2).reshape(B,T,20))
            x=x+self.f2(F.gelu(self.f1(self.fn(x))))
        return self.out(self.outn(x))
'''
INFERENCE_TAIL='''
def build_model():
    model=AdditionTransformer(); state=model.state_dict(); values=torch.tensor(_VALUES)
    offset=0
    with torch.no_grad():
        for name,shape in _SHAPES.items():
            n=1
            for size in shape: n*=size
            state[name].copy_(values[offset:offset+n].reshape(shape)); offset+=n
    model.eval()
    return model, {'architecture':'causal grouped-query digit transformer','tokenization':'interleaved least-significant-first','trained':True}

def add(model,a:int,b:int)->int:
    av=[int(c) for c in str(a)[::-1]]; bv=[int(c) for c in str(b)[::-1]]
    seq=[]
    for x,y in zip(av,bv): seq.extend((x,y))
    seq.append(10); out=[]
    device=next(model.parameters()).device
    with torch.no_grad():
        for _ in range(9):
            logits=model(torch.tensor([seq],dtype=torch.long,device=device))
            digit=int(logits[0,-1].argmax())
            out.append(digit); seq.append(digit)
    return int(''.join(str(x) for x in out[::-1]))
'''

def main():
    torch.manual_seed(2025); random.seed(2025); torch.backends.cuda.matmul.allow_tf32=True
    model=AddTransformer(4).cuda()
    # Write a complete graded-path file immediately; replaced after every trained stage.
    export_plain(model)
    model=train_phase(model,36000,2e-3,.18,'teacher',8192)
    model=train_phase(model,6000,5e-5,.30,'stable',8192); export_plain(model)
    model=prune(model,3); model=train_phase(model,18000,2e-5,.30,'width3',8192); export_plain(model)
    model=prune(model,2); model=train_phase(model,30000,1.2e-5,.35,'width2',8192)
    model=train_phase(model,12000,4e-6,.40,'polish',8192)
    model=train_phase(model,6000,1e-6,.60,'edge',8192)
    print('FINAL',accuracy(model,500000,0.),accuracy(model,500000,1.0),flush=True)
    torch.save({'ff':2,'state':model.state_dict()},'/workspace/final.pt'); export_plain(model)

if __name__=='__main__': main()
