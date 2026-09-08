import math, random, time
import torch
from torch import nn
import torch.nn.functional as F
from submission import AdditionTransformer

DEVICE='cuda'; BS=8192

def digits(x):
    out=[]
    for _ in range(9):
        out.append(x.remainder(10)); x=torch.div(x,10,rounding_mode='floor')
    return torch.stack(out,1)

def batch(bs, structured=0.35):
    lo,hi=10_000_000,100_000_000
    a=torch.randint(lo,hi,(bs,),device=DEVICE,dtype=torch.long)
    b=torch.randint(lo,hi,(bs,),device=DEVICE,dtype=torch.long)
    m=int(bs*structured)
    if m:
        typ=torch.randint(0,8,(m,),device=DEVICE)
        aa=torch.randint(lo,hi,(m,),device=DEVICE,dtype=torch.long)
        bb=torch.randint(lo,hi,(m,),device=DEVICE,dtype=torch.long)
        # Exact complements and near-complements create full carry chains.
        q=torch.randint(1,9,(m,),device=DEVICE,dtype=torch.long)*10_000_000
        cand=q-aa.remainder(q.clamp_min(10_000_000))
        valid=(cand>=lo)&(cand<hi)
        bb=torch.where((typ==0)&valid,cand,bb)
        # asymmetric suffixes of 0/9 at all lengths
        lens=torch.randint(1,8,(m,),device=DEVICE)
        pow10=(10**lens).long()
        prefix=torch.div(aa,pow10,rounding_mode='floor')*pow10
        aa=torch.where(typ==1,prefix+(pow10-1),aa)
        bb=torch.where(typ==1,torch.div(bb,pow10,rounding_mode='floor')*pow10+1,bb)
        aa=torch.where(typ==2,torch.div(aa,pow10,rounding_mode='floor')*pow10,aa)
        bb=torch.where(typ==2,torch.div(bb,pow10,rounding_mode='floor')*pow10+(pow10-1),bb)
        # rounded boundaries, repeated digits, extrema, sparse inner digits
        scale=(10**torch.randint(1,8,(m,),device=DEVICE)).long()
        aa=torch.where(typ==3,(torch.div(aa,scale,rounding_mode='floor')*scale).clamp(lo,hi-1),aa)
        rep=torch.randint(1,10,(m,),device=DEVICE)*11_111_111
        aa=torch.where(typ==4,rep,aa); bb=torch.where(typ==4,(10-rep.div(11_111_111))*11_111_111-1,bb)
        aa=torch.where(typ==5,torch.where(torch.rand(m,device=DEVICE)<.5,torch.full_like(aa,lo),torch.full_like(aa,hi-1)),aa)
        # Force random carry start/stop with complementary low suffix.
        r=torch.randint(1,8,(m,),device=DEVICE); p=(10**r).long()
        low=aa.remainder(p); comp=p-low
        valid=(comp>0)&(comp<p)
        bb=torch.where((typ==6)&valid,torch.div(bb,p,rounding_mode='floor')*p+comp,bb)
        # high prefix plus long nines
        aa=torch.where(typ==7,torch.div(aa,p,rounding_mode='floor')*p+(p-1),aa)
        a[:m]=aa.clamp(lo,hi-1); b[:m]=bb.clamp(lo,hi-1)
    da=digits(a)[:,:8]; db=digits(b)[:,:8]; y=digits(a+b)
    x=torch.empty(bs,25,device=DEVICE,dtype=torch.long)
    x[:,:16:2]=da; x[:,1:16:2]=db; x[:,16]=10; x[:,17:]=y[:,:8]
    return x,y

def evaluate(model,n=100000,structured=0.0):
    model.eval(); good=total=0
    with torch.no_grad():
      for _ in range((n+BS-1)//BS):
        x,y=batch(min(BS,n-total),structured)
        seq=x[:,:17].clone()
        for j in range(9):
          pred=model(seq)[:,-1].argmax(-1)
          if j<8: seq=torch.cat((seq,pred[:,None]),1)
          if j==0: allgood=pred.eq(y[:,j])
          else: allgood &= pred.eq(y[:,j])
        good += allgood.sum().item(); total += len(x)
    model.train(); return good,total

def train_stage(model,steps,lr,structured,save):
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.01,betas=(.9,.98))
    start=time.time()
    for step in range(1,steps+1):
        x,y=batch(BS,structured)
        logits=model(x)[:,16:25]
        loss=F.cross_entropy(logits.reshape(-1,10),y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step%1000==0:
            print(step,loss.item(),time.time()-start,flush=True)
        if step%4000==0:
            torch.save(model.state_dict(),save)
    torch.save(model.state_dict(),save)

def prune(model,neww):
    old=model.ff1.out_features
    score=model.ff1.weight.norm(dim=1)*model.ff2.weight.norm(dim=0)
    keep=score.topk(neww).indices.sort().values
    out=AdditionTransformer(neww).cuda()
    state={k:v for k,v in model.state_dict().items() if not k.startswith('ff1.') and not k.startswith('ff2.')}
    out.load_state_dict(state,strict=False)
    with torch.no_grad(): out.ff1.weight.copy_(model.ff1.weight[keep]); out.ff2.weight.copy_(model.ff2.weight[:,keep])
    return out

if __name__=='__main__':
    torch.manual_seed(3401); torch.set_float32_matmul_precision('high')
    model=AdditionTransformer(4).cuda()
    train_stage(model,36000,0.002,0.35,'/workspace/teacher.pt')
    print('teacher',evaluate(model,100000,0),evaluate(model,100000,.7),flush=True)
    model=prune(model,3); train_stage(model,18000,2e-5,.5,'/workspace/w3.pt')
    print('w3',evaluate(model,100000,0),evaluate(model,100000,.7),flush=True)
    model=prune(model,2); train_stage(model,30000,1e-5,.5,'/workspace/w2.pt')
    train_stage(model,16000,3e-6,.55,'/workspace/w2.pt')
    print('w2',evaluate(model,500000,0),evaluate(model,500000,.7),flush=True)
