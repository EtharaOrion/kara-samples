"""Diagnostic: can the parent learn the single-place task (n=1)?  That task needs
the additive code, the fold, and generate-detection, but no carry routing."""
import torch, arch, data, train as T

dev="cuda"
E=2048
for U,C,ls0 in ((2,1,2.0),(4,1,2.0),(2,1,4.0),(4,1,4.0)):
    cfg=arch.default_cfg(C=C,U=U,ls_init=ls0)
    p=arch.init_params(cfg,E,dev,seed=11)
    for v in p.values(): v.requires_grad_(True)
    opt=torch.optim.AdamW(list(p.values()),lr=0.012,betas=(0.9,0.99))
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=0.012,total_steps=3000,pct_start=0.15)
    g=torch.Generator(device=dev).manual_seed(3)
    eg=torch.Generator(device=dev).manual_seed(9)
    for s in range(3000):
        ta,tb,tgt,_,_=data.batch(512,1,dev,g)
        dl=arch.forward(p,ta,tb,cfg)
        loss,per=T.ce_loss(dl,tgt)
        opt.zero_grad(set_to_none=True); loss.backward(); T.per_member_clip(p,1.0); opt.step(); sch.step()
    acc=T.exact_match(p,cfg,1,dev,eg,N=4096)
    acc8=T.exact_match(p,cfg,8,dev,eg,N=4096)
    print(f"C={C} U={U} ls0={ls0}: n=1 acc max {acc.max():.4f}  #>=0.999 {int((acc>=0.999).sum())}/{E}   n=8 acc max {acc8.max():.4f}")
