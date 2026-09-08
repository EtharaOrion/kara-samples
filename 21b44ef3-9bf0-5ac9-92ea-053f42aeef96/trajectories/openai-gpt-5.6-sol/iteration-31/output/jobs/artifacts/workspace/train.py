import math, random, sys, time
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, '/workspace')
from submission import AdditionTransformer

DEVICE='cuda'; B=8192

def digits(x):
    p=torch.tensor([1,10,100,1000,10000,100000,1000000,10000000,100000000],device=DEVICE)
    return (x[:,None]//p%10).long()

def uniform(n):
    return torch.randint(10_000_000,100_000_000,(n,),device=DEVICE),torch.randint(10_000_000,100_000_000,(n,),device=DEVICE)

def structured(n):
    a,b=uniform(n); typ=torch.randint(0,8,(n,),device=DEVICE)
    # Exact/near complements exercise all-length carry chains.
    m=typ==0; b[m]=(100_000_000-a[m]+torch.randint(-9,10,(int(m.sum()),),device=DEVICE)).clamp(10_000_000,99_999_999)
    # Unequal trailing runs of 9 and 0.
    for kind in (1,2):
        m=typ==kind; c=int(m.sum())
        if c:
            run=torch.randint(1,8,(c,),device=DEVICE); ten=(10**run)
            if kind==1: a[m]=(a[m]//ten)*ten+(ten-1)
            else: b[m]=(b[m]//ten)*ten
    # Decimal boundaries with jitter.
    m=typ==3; c=int(m.sum())
    if c:
        power=10**torch.randint(1,8,(c,),device=DEVICE)
        a[m]=((a[m]//power)*power+torch.randint(-12,13,(c,),device=DEVICE)).clamp(10_000_000,99_999_999)
    # Repeated digits.
    m=typ==4; c=int(m.sum())
    if c:
        d=torch.randint(1,10,(c,),device=DEVICE); a[m]=d*11_111_111
    # Sparse interior digits while retaining full width.
    m=typ==5; c=int(m.sum())
    if c:
        lead=torch.randint(1,10,(c,),device=DEVICE); place=10**torch.randint(0,7,(c,),device=DEVICE); val=torch.randint(0,10,(c,),device=DEVICE)
        a[m]=lead*10_000_000+val*place
    # Near extremes.
    m=typ==6; c=int(m.sum())
    if c: a[m]=torch.where(torch.rand(c,device=DEVICE)<.5,10_000_000+torch.randint(0,1000,(c,),device=DEVICE),99_999_999-torch.randint(0,1000,(c,),device=DEVICE))
    return a,b

def batch(n=B, frac=.4):
    a,b=uniform(n); c=int(n*frac)
    if c: a[:c],b[:c]=structured(c)
    ad,bd=digits(a),digits(b); sd=digits(a+b)
    inp=torch.empty(n,25,dtype=torch.long,device=DEVICE)
    inp[:,:16:2]=ad[:,:8]; inp[:,1:16:2]=bd[:,:8]; inp[:,16]=10; inp[:,17:]=sd[:,:8]
    return inp,sd

def loss(model, frac=.4):
    x,y=batch(frac=frac)
    return F.cross_entropy(model(x)[:,16:25].reshape(-1,10),y.reshape(-1))

@torch.no_grad()
def evaluate(model,n=100000,structured_only=False,chunk=10000):
    model.eval(); ok=0; total=0; margin=100.
    for _ in range((n+chunk-1)//chunk):
        k=min(chunk,n-total)
        a,b=structured(k) if structured_only else uniform(k)
        ad,bd=digits(a),digits(b); expected=digits(a+b); tok=torch.empty(k,17,dtype=torch.long,device=DEVICE)
        tok[:,:16:2]=ad[:,:8];tok[:,1:16:2]=bd[:,:8];tok[:,16]=10
        pred=[]
        for j in range(9):
            out=model(tok)[:,-1]; vals=out.argmax(1); pred.append(vals)
            top=out.topk(2,1).values; margin=min(margin,float((top[:,0]-top[:,1]).min()))
            tok=torch.cat((tok,vals[:,None]),1)
        p=torch.stack(pred,1);ok+=(p==expected).all(1).sum().item();total+=k
    model.train(); return ok,total,margin

def train_stage(model,steps,lr,frac,label):
    model.train(); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
    start=time.time()
    for s in range(1,steps+1):
        opt.zero_grad(set_to_none=True); l=loss(model,frac);l.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if s%1000==0: print(label,s,float(l),f'{time.time()-start:.0f}s',flush=True)
    torch.save(model.state_dict(),f'/workspace/{label}.pt')

def prune(src,newff):
    dst=AdditionTransformer(newff).to(DEVICE); state=src.state_dict(); ds=dst.state_dict()
    for k in ds:
        if k not in ('ff1.weight','ff1.bias','ff2.weight'): ds[k].copy_(state[k])
    score=state['ff1.weight'].norm(dim=1)*state['ff2.weight'].norm(dim=0)
    keep=score.topk(newff).indices.sort().values
    ds['ff1.weight'].copy_(state['ff1.weight'][keep]);ds['ff1.bias'].copy_(state['ff1.bias'][keep]);ds['ff2.weight'].copy_(state['ff2.weight'][:,keep])
    return dst

def export(model):
    template=Path('/workspace/submission.py').read_text(); before=template.split('_STATE =',1)[0]; after='\ndef build_model():'+template.split('\ndef build_model():',1)[1]
    entries=[]
    for k,v in model.state_dict().items():
        vals=repr(v.detach().float().cpu().tolist())
        entries.append(f"    {k!r}: torch.tensor({vals}),")
    Path('/workspace/submission.py').write_text(before+'_STATE = {\n'+'\n'.join(entries)+'\n}\n'+after)

if __name__=='__main__':
    torch.manual_seed(314159); torch.set_float32_matmul_precision('high')
    m=AdditionTransformer(4).to(DEVICE)
    train_stage(m,36000,2e-3,.4,'teacher')
    print('teacher',evaluate(m),evaluate(m,structured_only=True),flush=True)
    m=prune(m,3);train_stage(m,18000,2e-5,.5,'width3')
    print('w3',evaluate(m),evaluate(m,structured_only=True),flush=True)
    m=prune(m,2);train_stage(m,30000,1e-5,.5,'width2a')
    train_stage(m,16000,3e-6,.55,'width2b')
    print('w2',evaluate(m,500000),evaluate(m,500000,True),flush=True)
    export(m)
