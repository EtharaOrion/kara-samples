import torch, torch.nn.functional as F, time, sys
from data import make_batch
from model_def import TinyAdder
from train import evaluate
dev='cuda'; torch.manual_seed(0)
m=TinyAdder(d_model=8,d_ff=16,tied_head=False).to(dev)
print('params',sum(p.numel() for p in m.parameters()),flush=True)
opt=torch.optim.AdamW(m.parameters(),lr=3e-3,betas=(0.9,0.98),fused=True)
sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=3e-3,total_steps=8000,pct_start=0.05)
t=time.time()
for s in range(8000):
    da,db,y=make_batch(4096,dev)
    l=F.cross_entropy(m(da,db)[:,1:].reshape(-1,10),y[:,1:].reshape(-1))
    opt.zero_grad(set_to_none=True); l.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); sch.step()
    if (s+1)%400==0:
        print(s+1, round(l.item(),4), round(evaluate(m,dev,n=100000,bs=100000),5), int(time.time()-t),'s',flush=True)
