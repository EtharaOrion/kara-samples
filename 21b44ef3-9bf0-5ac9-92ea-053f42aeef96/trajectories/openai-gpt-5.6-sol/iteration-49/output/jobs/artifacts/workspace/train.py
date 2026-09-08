import math, os, random, sys, time
import torch
from torch import nn
import torch.nn.functional as F

D=20; H=4; HD=5; NPOS=25

class AddTransformer(nn.Module):
    def __init__(self, ff=4):
        super().__init__()
        self.tok=nn.Embedding(11,D)
        self.pa=nn.Parameter(torch.empty(NPOS,2)); self.pb=nn.Parameter(torch.empty(2,D))
        self.q=nn.ModuleList([nn.Linear(D,D,bias=False) for _ in range(2)])
        self.o=nn.ModuleList([nn.Linear(D,D,bias=False) for _ in range(2)])
        self.k=nn.Linear(D,HD,bias=False); self.v=nn.Linear(D,HD,bias=False)
        self.an=nn.LayerNorm(D); self.fn=nn.LayerNorm(D)
        self.f1=nn.Linear(D,ff,bias=False); self.f2=nn.Linear(ff,D,bias=False)
        self.outn=nn.LayerNorm(D); self.head=nn.Linear(D,10,bias=False)
        nn.init.normal_(self.pa,std=.02); nn.init.normal_(self.pb,std=.02)
    def forward(self,t):
        L=t.shape[1]; x=self.tok(t)+self.pa[:L]@self.pb
        mask=torch.triu(torch.ones(L,L,device=t.device,dtype=torch.bool),1)
        for z in range(2):
            y=self.an(x); q=self.q[z](y).view(-1,L,H,HD).transpose(1,2)
            k=self.k(y).unsqueeze(1); v=self.v(y).unsqueeze(1)
            a=F.scaled_dot_product_attention(q,k,v,attn_mask=~mask)
            x=x+self.o[z](a.transpose(1,2).reshape(-1,L,D))
            x=x+self.f2(F.gelu(self.f1(self.fn(x))))
        return self.head(self.outn(x))

def digits(n):
    out=[]
    for _ in range(9): out.append(n%10); n=n//10
    return out

def batch(bs, structured=.18, device='cuda'):
    lo=10_000_000; hi=99_999_999
    a=torch.randint(lo,hi+1,(bs,),device=device); b=torch.randint(lo,hi+1,(bs,),device=device)
    m=int(bs*structured)
    if m:
        n=m//8; idx=0
        # Exact and near complements, critical carry chains.
        if n:
            aa=torch.randint(lo,hi+1,(n,),device=device); jitter=torch.randint(-9,10,(n,),device=device)
            bb=(100_000_000-aa+jitter).clamp(lo,hi); a[idx:idx+n]=aa; b[idx:idx+n]=bb; idx+=n
        # Long suffixes of 9 against arbitrary/sparse operands.
        if n:
            p=10**torch.randint(1,8,(n,),device=device); pref=torch.randint(1,100_000_000,(n,),device=device)
            aa=(pref//p*p+(p-1)).clamp(lo,hi); bb=torch.randint(lo,hi+1,(n,),device=device)
            a[idx:idx+n]=aa; b[idx:idx+n]=bb; idx+=n
        # Decimal boundaries and nearby values.
        if n:
            p=10**torch.randint(1,8,(n,),device=device); base=torch.randint(lo,hi+1,(n,),device=device)//p*p
            aa=(base+torch.randint(-2,3,(n,),device=device)).clamp(lo,hi)
            bb=torch.randint(lo,hi+1,(n,),device=device); a[idx:idx+n]=aa;b[idx:idx+n]=bb;idx+=n
        # Repeated digits.
        if n:
            da=torch.randint(1,10,(n,),device=device); db=torch.randint(1,10,(n,),device=device)
            a[idx:idx+n]=da*11_111_111; b[idx:idx+n]=db*11_111_111; idx+=n
        # Sparse decimal values.
        if n:
            p=10**torch.randint(0,8,(n,),device=device); q=10**torch.randint(0,8,(n,),device=device)
            aa=(torch.randint(1,10,(n,),device=device)*p+10_000_000).clamp(lo,hi)
            bb=(torch.randint(1,10,(n,),device=device)*q+10_000_000).clamp(lo,hi)
            a[idx:idx+n]=aa;b[idx:idx+n]=bb;idx+=n
        # Near extrema.
        if n:
            a[idx:idx+n]=torch.where(torch.rand(n,device=device)<.5,lo+torch.randint(0,10000,(n,),device=device),hi-torch.randint(0,10000,(n,),device=device))
            b[idx:idx+n]=torch.where(torch.rand(n,device=device)<.5,lo+torch.randint(0,10000,(n,),device=device),hi-torch.randint(0,10000,(n,),device=device));idx+=n
        # Unequal trailing zero/nine runs.
        if n:
            p=10**torch.randint(1,8,(n,),device=device); q=10**torch.randint(1,8,(n,),device=device)
            aa=(torch.randint(1,100_000_000,(n,),device=device)//p*p+(p-1)).clamp(lo,hi)
            bb=(torch.randint(1,100_000_000,(n,),device=device)//q*q).clamp(lo,hi)
            a[idx:idx+n]=aa;b[idx:idx+n]=bb;idx+=n
        # High-prefix asymmetric complement patterns.
        r=m-idx
        if r:
            p=10**torch.randint(2,8,(r,),device=device)
            aa=(torch.randint(lo,hi+1,(r,),device=device)//p*p+torch.randint(0,10,(r,),device=device)).clamp(lo,hi)
            bb=(torch.randint(lo,hi+1,(r,),device=device)//p*p+(p-1)).clamp(lo,hi)
            a[idx:m]=aa;b[idx:m]=bb
    vals=[]; aa=a; bb=b
    for _ in range(8): vals.extend([aa%10,bb%10]); aa=aa//10;bb=bb//10
    s=a+b; ds=[]
    for _ in range(9):ds.append(s%10);s=s//10
    inp=torch.empty(bs,25,dtype=torch.long,device=device)
    for i,v in enumerate(vals): inp[:,i]=v
    inp[:,16]=10
    for i in range(8):inp[:,17+i]=ds[i]
    target=torch.stack(ds,1)
    return inp,target

@torch.no_grad()
def accuracy(model,n=100000,structured=.0,bs=10000):
    model.eval(); good=0; total=0
    for _ in range((n+bs-1)//bs):
        z=min(bs,n-total); inp,y=batch(z,structured)
        seq=inp[:,:17].clone(); pred=[]
        for j in range(9):
            d=model(seq)[:,-1].argmax(-1);pred.append(d)
            if j<8:seq=torch.cat((seq,d[:,None]),1)
        good+=(torch.stack(pred,1)==y).all(1).sum().item();total+=z
    model.train(); return good,total

def train_stage(model,steps,lr,structured,name,bs=8192):
    model.cuda().train(); opt=torch.optim.AdamW(model.parameters(),lr=lr,betas=(.9,.98),weight_decay=.01)
    fwd=model
    start=time.time()
    for st in range(1,steps+1):
        inp,y=batch(bs,structured)
        logits=fwd(inp)[:,16:25]
        loss=F.cross_entropy(logits.reshape(-1,10),y.reshape(-1))
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        if st%2000==0 or st==steps:
            ac=accuracy(model,20000,0)[0]/20000; sc=accuracy(model,20000,.7)[0]/20000
            print(name,st,float(loss),ac,sc,'sec',int(time.time()-start),flush=True)
            torch.save({'model':model.state_dict(),'ff':model.f1.out_features},'/workspace/latest.pt')
    return model

def prune(model,newff):
    old=model.f1.out_features
    score=model.f1.weight.norm(dim=1)*model.f2.weight.norm(dim=0)
    keep=score.topk(newff).indices.sort().values
    out=AddTransformer(newff).cuda(); sd=model.state_dict(); ns=out.state_dict()
    for k in ns:
        if k=='f1.weight':ns[k].copy_(sd[k][keep])
        elif k=='f2.weight':ns[k].copy_(sd[k][:,keep])
        else:ns[k].copy_(sd[k])
    return out

def export(model,path='/workspace/submission.py'):
    sd={k:v.detach().cpu() for k,v in model.state_dict().items()}
    source='''import torch\nfrom torch import nn\nimport torch.nn.functional as F\nD=20; H=4; HD=5\nclass Model(nn.Module):\n def __init__(self):\n  super().__init__(); self.tok=nn.Embedding(11,D); self.pa=nn.Parameter(torch.empty(25,2)); self.pb=nn.Parameter(torch.empty(2,D)); self.q=nn.ModuleList([nn.Linear(D,D,bias=False) for _ in range(2)]); self.o=nn.ModuleList([nn.Linear(D,D,bias=False) for _ in range(2)]); self.k=nn.Linear(D,HD,bias=False); self.v=nn.Linear(D,HD,bias=False); self.an=nn.LayerNorm(D); self.fn=nn.LayerNorm(D); self.f1=nn.Linear(D,2,bias=False); self.f2=nn.Linear(2,D,bias=False); self.outn=nn.LayerNorm(D); self.head=nn.Linear(D,10,bias=False)\n def forward(self,t):\n  L=t.shape[1]; x=self.tok(t)+self.pa[:L]@self.pb; mask=torch.triu(torch.ones(L,L,device=t.device,dtype=torch.bool),1)\n  for z in range(2):\n   y=self.an(x); q=self.q[z](y).view(-1,L,H,HD).transpose(1,2); k=self.k(y).unsqueeze(1); v=self.v(y).unsqueeze(1); w=torch.softmax((q@k.transpose(-2,-1))/(HD**.5)+mask.to(x.dtype)*-1e4,dim=-1); x=x+self.o[z]((w@v).transpose(1,2).reshape(-1,L,D)); x=x+self.f2(F.gelu(self.f1(self.fn(x))))\n  return self.head(self.outn(x))\n_STATE='''+repr({k:v.tolist() for k,v in sd.items()})+'''\ndef build_model():\n m=Model(); m.load_state_dict({k:torch.tensor(v) for k,v in _STATE.items()}); m.eval(); return m,{'architecture':'trained causal grouped-query digit transformer','training':'AdamW on generated 8-digit addition examples'}\ndef add(model,a:int,b:int)->int:\n da=[int(c) for c in str(a)[::-1]]; db=[int(c) for c in str(b)[::-1]]; tokens=[]\n for x,y in zip(da,db): tokens.extend((x,y))\n seq=torch.tensor([tokens+[10]],dtype=torch.long,device=next(model.parameters()).device); out=[]\n with torch.no_grad():\n  for i in range(9):\n   d=int(model(seq)[0,-1].argmax()); out.append(d)\n   if i<8: seq=torch.cat((seq,torch.tensor([[d]],device=seq.device)),1)\n return int(''.join(str(x) for x in out[::-1]))\n'''
    open(path,'w').write(source)
    print('exported',path,sum(x.numel() for x in model.parameters()),flush=True)

if __name__=='__main__':
    torch.manual_seed(2025); random.seed(2025); torch.set_float32_matmul_precision('high')
    m=AddTransformer(4).cuda()
    m=train_stage(m,36000,2e-3,.18,'teacher')
    m=train_stage(m,6000,5e-5,.30,'stabilize')
    m=prune(m,3);m=train_stage(m,18000,2e-5,.32,'ff3')
    m=prune(m,2);m=train_stage(m,30000,1.2e-5,.35,'ff2')
    m=train_stage(m,12000,4e-6,.40,'polish')
    m=train_stage(m,6000,1e-6,.55,'edge')
    print('FINAL',accuracy(m,500000,0),accuracy(m,500000,.7),flush=True)
    torch.save({'model':m.state_dict(),'ff':2},'/workspace/final.pt');export(m)
