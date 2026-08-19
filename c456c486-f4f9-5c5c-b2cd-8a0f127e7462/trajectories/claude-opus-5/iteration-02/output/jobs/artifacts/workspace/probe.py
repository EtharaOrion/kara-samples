import torch, torch.nn.functional as F, time, sys
from data import make_batch
from model_def import TinyAdder
dev='cuda'; torch.manual_seed(int(sys.argv[1]) if len(sys.argv)>1 else 0)
lr=float(sys.argv[2]) if len(sys.argv)>2 else 6e-3
bs=int(sys.argv[3]) if len(sys.argv)>3 else 8192
N=4000
m=TinyAdder(d_model=3,d_ff=4,tied_head=True).to(dev)
print('params',sum(p.numel() for p in m.parameters()),'lr',lr,'bs',bs,flush=True)
opt=torch.optim.AdamW(m.parameters(),lr=lr,betas=(0.9,0.98),fused=True)
sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=lr,total_steps=N,pct_start=0.05)
for s in range(N):
    da,db,y=make_batch(bs,dev)
    l=F.cross_entropy(m(da,db)[:,1:].reshape(-1,10),y[:,1:].reshape(-1))
    opt.zero_grad(set_to_none=True); l.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); sch.step()
    if (s+1)%500==0:
        with torch.no_grad():
            da,db,y=make_batch(20000,dev,uniform=True)
            p=m(da,db).argmax(-1)
            per=(p[:,1:]==y[:,1:]).float().mean(0)
            ex=(p[:,1:]==y[:,1:]).all(-1).float().mean()
            u=m.digit_code.detach().cpu()
            print(s+1,'loss %.4f exact %.4f'%(l.item(),ex),
                  'perpos',' '.join('%.2f'%v for v in per.tolist()),flush=True)
            print('   U',' '.join('%.2f'%v for v in u.tolist()),'tau %.2f slope %.2f'%(m.tau.item(),m.slope.item()),flush=True)
