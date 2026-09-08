import math
import torch
import torch.nn.functional as F
from train import AdditionTransformer, make_batch, digits, evaluate, eval_curated, SEQ, LO, HI

torch.manual_seed(1414)
torch.backends.cuda.matmul.allow_tf32=True
dev=torch.device('cuda')
model=AdditionTransformer().to(dev)
model.load_state_dict(torch.load('/workspace/final.pt',weights_only=True))

# A broad deterministic basis of sparse boundaries, long 0/9 suffixes, repeats, and complement neighborhoods.
vals=set()
for lead in range(1,10):
    base=lead*10_000_000
    vals.update([base, min(HI,base+1), min(HI,base+9), min(HI,base+99)])
    for p in range(1,8):
        s=10**p
        for d in (-2,-1,0,1,2):
            for v in (base+s+d, base+s-1+d, base+10_000_000-s+d):
                if LO<=v<=HI: vals.add(v)
for d in range(1,10): vals.add(d*11_111_111)
vals.update(range(HI-200,HI+1))
vals=sorted(vals)
pairs=[]
for a in vals:
    # symmetric and asymmetric basis products
    for b in vals: pairs.append((a,b))
    for d in range(-30,31):
        b=100_000_000-a+d
        if LO<=b<=HI:pairs.append((a,b))
A=torch.tensor([a for a,b in pairs],device=dev)
B=torch.tensor([b for a,b in pairs],device=dev)
print('edge pool',len(pairs),'initial',evaluate(model,50000,structured=0.3),eval_curated(model)[0])


def batch(n):
    # 40% broad generated curriculum, 35% combinatorial pool, 25% uniform.
    t,y,a,b=make_batch(n,dev,0.45)
    k=int(n*.60)
    idx=torch.randint(0,len(A),(k,),device=dev)
    aa,bb=A[idx],B[idx]
    t[:k,0:16:2]=digits(aa);t[:k,1:16:2]=digits(bb);y[:k]=digits(aa+bb,9)
    return t,y

opt=torch.optim.AdamW(model.parameters(),lr=2e-6,weight_decay=0.0)
best=-1
for step in range(1,10001):
    if step==1:
        for g in opt.param_groups:g['lr']=5e-6
    t,y=batch(4096)
    opt.zero_grad(set_to_none=True)
    logits=model(t)
    # Emphasize wrong/low-margin positions without changing the learned target.
    ce=F.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='none').view(-1,9)
    loss=(ce.mean(1)+.35*ce.max(1).values).mean()
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
    if step%500==0:
        r=evaluate(model,20000,structured=0.0)[0]
        s=evaluate(model,20000,structured=0.7)[0]
        e=eval_curated(model)[0]
        score=min(r,e)
        print(step,loss.item(),'random',r,'structured',s,'edge',e,flush=True)
        if score>best:
            best=score;torch.save(model.state_dict(),'/workspace/edge_best.pt')
model.load_state_dict(torch.load('/workspace/edge_best.pt',weights_only=True))
from train import export
export(model,'/workspace/submission.py')
torch.save(model.state_dict(),'/workspace/final.pt')
print('FINAL',evaluate(model,500000,structured=0),evaluate(model,500000,structured=.7),eval_curated(model))
