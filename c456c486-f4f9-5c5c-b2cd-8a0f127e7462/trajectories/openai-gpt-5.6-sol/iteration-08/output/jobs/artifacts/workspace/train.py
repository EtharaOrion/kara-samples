import math, random, sys, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

D=16; H=2; HD=8; M=32; N=14; T=45

class BiaslessLN(nn.Module):
    def __init__(self, d):
        super().__init__(); self.weight=nn.Parameter(torch.ones(d))
    def forward(self,x): return F.layer_norm(x,(x.shape[-1],),self.weight,None,1e-5)

class Block(nn.Module):
    def __init__(self, shared_bias):
        super().__init__(); self.n1=BiaslessLN(D); self.qkv=nn.Linear(D,3*D,bias=False); self.proj=nn.Linear(D,D,bias=False)
        self.n2=BiaslessLN(D); self.fc1=nn.Linear(D,M,bias=False); self.fc2=nn.Linear(M,D,bias=False); self.shared_bias=shared_bias
    def forward(self,x):
        B,L,_=x.shape; z=self.n1(x); q,k,v=self.qkv(z).chunk(3,-1)
        q=q.view(B,L,H,HD).transpose(1,2); k=k.view(B,L,H,HD).transpose(1,2); v=v.view(B,L,H,HD).transpose(1,2)
        s=q@k.transpose(-2,-1)/math.sqrt(HD)
        idx=torch.arange(L,device=x.device); ok=(idx[:,None]>=idx[None,:]) & ((idx[:,None]-idx[None,:])<=5)
        s=s.masked_fill(~ok,-1e4); y=(s.softmax(-1)@v).transpose(1,2).reshape(B,L,D)
        x=x+self.proj(y); x=x+self.fc2(F.gelu(self.fc1(self.n2(x))+self.shared_bias)); return x

class Model(nn.Module):
    def __init__(self):
        super().__init__(); self.digit=nn.Embedding(10,D); self.role=nn.Embedding(3,D); self.mb=nn.Parameter(torch.zeros(M))
        self.blocks=nn.ModuleList([Block(self.mb),Block(self.mb)]); self.norm=BiaslessLN(D); self.head=nn.Linear(D,10,bias=False)
    def forward(self,tok):
        L=tok.shape[1]; roles=torch.arange(L,device=tok.device)%3
        x=self.digit(tok)+self.role(roles)
        for b in self.blocks:x=b(x)
        return self.head(self.norm(x))

def data(bs, device, structured=0.0):
    # Random complete 14-digit pairs. Structured batches emphasize long carries/repeated digits.
    a=torch.randint(0,10,(bs,N),device=device); b=torch.randint(0,10,(bs,N),device=device)
    if structured and random.random()<structured:
        kind=random.randrange(4)
        if kind==0:
            a[:]=torch.randint(0,10,(bs,1),device=device); b[:]=torch.randint(0,10,(bs,1),device=device)
        elif kind==1:
            a=torch.randint(5,10,(bs,N),device=device); b=torch.randint(5,10,(bs,N),device=device)
        elif kind==2:
            a=torch.randint(0,10,(bs,N),device=device); b=9-a
            noise=torch.rand((bs,N),device=device)<.12; b=torch.where(noise,torch.randint(0,10,(bs,N),device=device),b)
        else:
            a=torch.randint(0,10,(bs,N),device=device); b=torch.randint(0,10,(bs,N),device=device)
            start=torch.randint(0,N-5,(bs,),device=device)
            for j in range(bs):
                q=int(start[j]); a[j,q:q+6]=torch.randint(5,10,(6,),device=device); b[j,q:q+6]=torch.randint(5,10,(6,),device=device)
    out=torch.empty((bs,N+1),dtype=torch.long,device=device); carry=torch.zeros(bs,dtype=torch.long,device=device)
    for i in range(N):
        z=a[:,i]+b[:,i]+carry; out[:,i]=z%10; carry=z//10
    out[:,N]=carry
    seq=torch.zeros((bs,T),dtype=torch.long,device=device)
    seq[:,0:3*N:3]=a; seq[:,1:3*N:3]=b; seq[:,2:3*N:3]=out[:,:N]
    # final operands are zero padding, final output is carry
    seq[:,3*N]=0; seq[:,3*N+1]=0; seq[:,3*N+2]=out[:,N]
    return seq

@torch.no_grad()
def greedy(model, cases=2000, bs=500):
    model.eval(); good=0
    for _ in range((cases+bs-1)//bs):
        n=min(bs,cases-good) if False else bs
        truth=data(n,next(model.parameters()).device); inp=truth.clone(); inp[:,2::3]=0
        for p in range(2,T,3): inp[:,p]=model(inp[:,:p])[:,-1].argmax(-1)
        good+=(inp[:,2::3]==truth[:,2::3]).all(1).sum().item()
    return good/(((cases+bs-1)//bs)*bs)

def export(model):
    sd={k:v.detach().float().cpu().flatten().tolist() for k,v in model.state_dict().items()}
    code=Path('/workspace/submission_template.py').read_text()
    code=code.replace('__STATE__',repr(sd))
    Path('/workspace/submission.py').write_text(code)

if __name__=='__main__':
    torch.set_float32_matmul_precision('high'); dev='cuda' if torch.cuda.is_available() else 'cpu'; torch.manual_seed(8); random.seed(8)
    model=Model().to(dev); print('params',sum(p.numel() for p in model.parameters()),flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-3,weight_decay=.005,betas=(.9,.98))
    # Schedule can be extended by passing number of steps (default 12000).
    steps=int(sys.argv[1]) if len(sys.argv)>1 else 12000; t=time.time()
    for step in range(1,steps+1):
        if step==3501:
            for g in opt.param_groups:g['lr']=1.5e-3
        if step==5501:
            for g in opt.param_groups:g['lr']=7e-4
        if step==7501:
            for g in opt.param_groups:g['lr']=3e-4
        if step==9001:
            for g in opt.param_groups:g['lr']=1e-4
        seq=data(4096 if step<=7500 else 2048,dev,0 if step<=4500 else .35)
        logits=model(seq[:,:-1]); target=seq[:,1:]
        # predict output-role positions only (targets whose absolute position is 2 mod 3)
        pos=torch.arange(1,T,device=dev); mask=pos%3==2
        loss=F.cross_entropy(logits[:,mask].reshape(-1,10),target[:,mask].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step%250==0:
            acc=greedy(model,1000,500); print(step,round(loss.item(),6),acc,'lr',opt.param_groups[0]['lr'],'sec',round(time.time()-t),flush=True)
            export(model)
    torch.save(model.state_dict(),'/workspace/model.pt'); export(model)
