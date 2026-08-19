import math, random, time
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

D=10; H=2; FF=8; N=29; PD=2; DEVICE='cuda'; MAX=100_000_000_000_000

torch.manual_seed(50); random.seed(50)
torch.backends.cuda.matmul.allow_tf32=True

class Block(nn.Module):
    def __init__(self):
        super().__init__(); self.n1=nn.LayerNorm(D); self.qkv=nn.Linear(D,3*D); self.proj=nn.Linear(D,D)
        self.n2=nn.LayerNorm(D); self.f1=nn.Linear(D,FF); self.f2=nn.Linear(FF,D)
    def forward(self,x):
        z=self.n1(x); q,k,v=self.qkv(z).chunk(3,-1); B,L,_=q.shape; dh=D//H
        q=q.view(B,L,H,dh).transpose(1,2); k=k.view(B,L,H,dh).transpose(1,2); v=v.view(B,L,H,dh).transpose(1,2)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(B,L,D)
        x=x+self.proj(y); return x+self.f2(F.gelu(self.f1(self.n2(x))))

class AddTransformer(nn.Module):
    def __init__(self):
        super().__init__(); self.ea=nn.Embedding(11,D); self.eb=nn.Embedding(11,D); self.eo=nn.Embedding(10,D)
        self.pos=nn.Parameter(torch.randn(N,PD)*.02); self.blocks=nn.ModuleList([Block(),Block()]); self.norm=nn.LayerNorm(D); self.head=nn.Linear(D,10)
    def forward(self,a,b,prev):
        B=a.shape[0]; p=F.pad(self.pos,(0,D-PD)); src=self.ea(a)+self.eb(b)+p[:14]
        x=torch.cat((src,self.eo(prev)+p[14:14+prev.shape[1]]),1) if prev.shape[1] else src
        for block in self.blocks:x=block(x)
        return self.head(self.norm(x[:,13:]))

def targets(a,b):
    out=[]; carry=torch.zeros(a.shape[0],device=a.device,dtype=torch.long)
    for i in range(14):
        s=a[:,i]+b[:,i]+carry; out.append(s.remainder(10)); carry=torch.div(s,10,rounding_mode='floor')
    out.append(carry); return torch.stack(out,1)

def batch(bs, structured):
    a=torch.randint(0,10,(bs,14),device=DEVICE); b=torch.randint(0,10,(bs,14),device=DEVICE)
    if structured:
        m=bs//2; typ=torch.randint(0,6,(m,),device=DEVICE)
        # Random shifted carry/non-carry runs, including long and top-ending chains.
        start=torch.randint(0,14,(m,),device=DEVICE); length=torch.randint(1,15,(m,),device=DEVICE)
        end=torch.minimum(start+length,torch.full_like(start,14)); cols=torch.arange(14,device=DEVICE)[None,:]
        run=(cols>=start[:,None])&(cols<end[:,None]); first=cols==start[:,None]
        x=torch.randint(1,10,(m,14),device=DEVICE)
        carry_b=torch.where(first,10-x,9-x).clamp(0,9)
        non_b=(9-x).clamp(0,9)
        usecarry=(typ<3)[:,None]
        bb=torch.where(usecarry,carry_b,non_b)
        a[:m]=torch.where(run,x,a[:m]); b[:m]=torch.where(run,bb,b[:m])
        # Sparse isolated equal columns (5+5, 9+9) and powers/boundaries.
        sparse=(typ>=4); idx=torch.where(sparse)[0]
        if idx.numel():
            a[idx]=0; b[idx]=0; c=start[idx]; vals=torch.where((typ[idx]&1)==0,torch.full_like(c,5),torch.full_like(c,9))
            a[idx,c]=vals; b[idx,c]=vals
        # Repeated and complementary full operands.
        rep=torch.where(typ==3)[0]
        if rep.numel():
            da=torch.randint(0,10,(rep.numel(),1),device=DEVICE); db=torch.randint(0,10,(rep.numel(),1),device=DEVICE)
            a[rep]=da.expand(-1,14); b[rep]=db.expand(-1,14)
    y=targets(a,b); return a,b,y

@torch.inference_mode()
def eval_ar(model,n=32768,structured=False,bs=4096):
    model.eval(); errors=0
    for _ in range((n+bs-1)//bs):
        a,b,y=batch(min(bs,n),structured); n-=a.shape[0]; prev=torch.empty((a.shape[0],0),device=DEVICE,dtype=torch.long)
        for i in range(15):
            d=model(a,b,prev)[:,-1].argmax(-1); prev=torch.cat((prev,d[:,None]),1)
        errors+=(prev!=y).any(1).sum().item()
    model.train(); return errors

def export(model,path='/workspace/submission.py'):
    sd={k:v.detach().cpu().reshape(-1).tolist() for k,v in model.state_dict().items()}
    shapes={k:list(v.shape) for k,v in model.state_dict().items()}
    template=Path('/workspace/submission_template.py').read_text()
    text=template.replace('__STATE__',repr(sd)).replace('__SHAPES__',repr(shapes))
    Path(path).write_text(text)

model=AddTransformer().to(DEVICE); print('params',sum(p.numel() for p in model.parameters()),flush=True)
# Export an importable checkpoint immediately; it will be overwritten by trained checkpoints.
Path('/workspace/submission_template.py').exists() and export(model)
opt=torch.optim.AdamW(model.parameters(),lr=3e-3,weight_decay=.003,betas=(.9,.98))
steps=70000; bs=4096; best=10**9; t=time.time()
for step in range(1,steps+1):
    if step==12001:
        for g in opt.param_groups:g['lr']=1e-3
    if step==30001:
        for g in opt.param_groups:g['lr']=3e-4
    if step==50001:
        for g in opt.param_groups:g['lr']=1e-4
    if step==62001:
        for g in opt.param_groups:g['lr']=3e-5
    a,b,y=batch(bs,step>10000); logits=model(a,b,y[:,:-1]); loss=F.cross_entropy(logits.reshape(-1,10),y.reshape(-1))
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
    if step%2000==0:
        err=eval_ar(model,32768,step>10000)
        print(step,float(loss),err,'sec',round(time.time()-t,1),flush=True)
        # Prefer latest strong checkpoints, and always leave completed trained weights.
        if err<=best:
            best=err; export(model); torch.save(model.state_dict(),'/workspace/model.pt')
print('final large uniform',eval_ar(model,262144,False),'structured',eval_ar(model,262144,True),flush=True)
# Export final if it is at least as strong on a fresh mixed check as the retained checkpoint is likely to be.
err=eval_ar(model,65536,True); print('final select',err,flush=True)
if err<=best: export(model); torch.save(model.state_dict(),'/workspace/model.pt')
