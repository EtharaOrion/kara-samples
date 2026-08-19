import math, os, random, sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace')
from submission import AdditionTransformer

DEVICE='cuda'; BATCH=4096; POW=(10 ** torch.arange(14, device=DEVICE, dtype=torch.long))
torch.manual_seed(38001); random.seed(38001)
torch.set_float32_matmul_precision('high')

model=AdditionTransformer().to(DEVICE)
print('parameters',sum(p.numel() for p in model.parameters()), flush=True)

# Teacher-forced complete-sequence pass: source positions 0..14; output digit j is
# predicted at position 14+j, with only digits <j present there.
def full_logits_eager(ad,bd,y):
    x=torch.cat((model.a_embed(ad)+model.b_embed(bd), model.out_embed(y[:,:-1])),1)
    x=x+model.position
    for block in model.blocks: x=block(x,model.mask)
    return model.head(model.final_norm(x[:,14:]))

full_logits = full_logits_eager

def digits_of(n, count):
    return (n[:,None] // (10 ** torch.arange(count,device=DEVICE))) % 10

def make_batch(structured=0.0):
    a=torch.randint(0,100_000_000_000_000,(BATCH,),device=DEVICE)
    b=torch.randint(0,100_000_000_000_000,(BATCH,),device=DEVICE)
    n=int(BATCH*structured)
    if n:
        d1=torch.randint(0,10,(n,14),device=DEVICE)
        d2=torch.randint(0,10,(n,14),device=DEVICE)
        family=torch.randint(0,8,(n,),device=DEVICE)
        pos=torch.randint(0,14,(n,),device=DEVICE)
        length=torch.randint(1,15,(n,),device=DEVICE)
        cols=torch.arange(14,device=DEVICE)[None,:]
        run=(cols>=pos[:,None]) & (cols<torch.minimum(pos+length,torch.tensor(14,device=DEVICE))[:,None])
        start=cols==pos[:,None]
        # Carry chains: start sums to ten, following columns sum to nine.
        m=family<3
        rr=torch.randint(0,10,(n,14),device=DEVICE)
        d1=torch.where(run & m[:,None],rr,d1)
        d2=torch.where(run & m[:,None],9-rr,d2)
        stv=torch.randint(1,10,(n,1),device=DEVICE)
        d1=torch.where(start & m[:,None],stv,d1)
        d2=torch.where(start & m[:,None],10-stv,d2)
        # Matched non-carry runs (sum eight or nine).
        m=(family==3)|(family==4)
        target=torch.where((family==3)[:,None],torch.tensor(9,device=DEVICE),torch.tensor(8,device=DEVICE))
        rr=torch.minimum(torch.randint(0,10,(n,14),device=DEVICE),target)
        d1=torch.where(run&m[:,None],rr,d1); d2=torch.where(run&m[:,None],target-rr,d2)
        # Sparse exact boundaries, emphasizing 5+5 and 9+9 at arbitrary columns.
        m=family==5
        d1=torch.where(m[:,None],torch.zeros_like(d1),d1); d2=torch.where(m[:,None],torch.zeros_like(d2),d2)
        d1=torch.where(start&m[:,None],torch.tensor(5,device=DEVICE),d1); d2=torch.where(start&m[:,None],torch.tensor(5,device=DEVICE),d2)
        m=family==6
        d1=torch.where(m[:,None],torch.zeros_like(d1),d1); d2=torch.where(m[:,None],torch.zeros_like(d2),d2)
        d1=torch.where(start&m[:,None],torch.tensor(9,device=DEVICE),d1); d2=torch.where(start&m[:,None],torch.tensor(9,device=DEVICE),d2)
        # Repeated digit operands.
        m=family==7
        r1=torch.randint(0,10,(n,1),device=DEVICE); r2=torch.randint(0,10,(n,1),device=DEVICE)
        d1=torch.where(m[:,None],r1,d1); d2=torch.where(m[:,None],r2,d2)
        a[:n]=(d1*POW).sum(1); b[:n]=(d2*POW).sum(1)
    ad=digits_of(a,14); bd=digits_of(b,14)
    sent=torch.full((BATCH,1),10,device=DEVICE,dtype=torch.long)
    y=digits_of(a+b,15)
    return torch.cat((ad,sent),1),torch.cat((bd,sent),1),y

@torch.no_grad()
def ar_errors(count=65536, structured=0.0):
    model.eval(); total=0
    old_bs=globals()['BATCH']
    for _ in range((count+old_bs-1)//old_bs):
        ad,bd,y=make_batch(structured)
        prev=torch.empty((old_bs,0),dtype=torch.long,device=DEVICE)
        for j in range(15):
            pred=model(ad,bd,prev).argmax(1)
            prev=torch.cat((prev,pred[:,None]),1)
        total+=(prev!=y).any(1).sum().item()
    model.train(); return total

opt=torch.optim.AdamW(model.parameters(),lr=3e-3,weight_decay=.002)
# Uniform-first, then at least half uniform via structured fractions <= .5.
phases=[(14000,3e-3,0.0),(18000,1e-3,.35),(18000,3e-4,.5),(20000,1e-4,.5),(20000,3e-5,.5)]
step=0; start_time=time.time(); best=None; best_err=10**9
for steps,lr,mix in phases:
    for g in opt.param_groups:g['lr']=lr
    for local in range(steps):
        ad,bd,y=make_batch(mix)
        logits=full_logits(ad,bd,y)
        loss=F.cross_entropy(logits.reshape(-1,10),y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        step+=1
        if step%2000==0:
            with torch.no_grad():
                te=(logits.argmax(-1)!=y).any(1).float().mean().item()
            print(step,lr,mix,float(loss),te,'sec',round(time.time()-start_time),flush=True)
        if step>=30000 and step%10000==0:
            e1=ar_errors(65536,0.0); e2=ar_errors(65536,.75); score=e1+e2
            print('AR',step,e1,e2,flush=True)
            if score<=best_err:
                best_err=score; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; torch.save(best,'/workspace/best.pt')
if best is not None:model.load_state_dict(best)
torch.save(model.state_dict(),'/workspace/model.pt')
print('FINAL',ar_errors(262144,0),ar_errors(262144,.75), 'best',best_err,flush=True)
