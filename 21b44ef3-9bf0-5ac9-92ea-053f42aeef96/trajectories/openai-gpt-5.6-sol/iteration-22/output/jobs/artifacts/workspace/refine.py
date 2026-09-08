import sys, time
import torch
from torch import nn
import torch.nn.functional as F
sys.path.insert(0, '/workspace')
from submission import AdditionTransformer
from train import batch, accuracy, structured, BATCH, LOW, HIGH, POW8, POW9

torch.manual_seed(220023)
torch.set_float32_matmul_precision('high')
model=AdditionTransformer().cuda()
model.load_state_dict(torch.load('/workspace/best.pt', weights_only=True))
opt=torch.optim.AdamW(model.parameters(),lr=1e-5,betas=(.9,.98),weight_decay=.002,fused=True)
best_score=-1
start=time.time()
for step in range(1,15001):
    for g in opt.param_groups: g['lr']=1e-5 if step <= 9000 else 3e-6
    x,y=batch(.65)
    logits=model(x)[:,16:25]
    loss=F.cross_entropy(logits.reshape(-1,10),y.reshape(-1))
    opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
    if step%1000==0: print(step,loss.item(),time.time()-start,flush=True)
    if step%3000==0:
        r=accuracy(model,100000,'random'); s=accuracy(model,100000,'structured')
        score=min(r[0]/r[1],s[0]/s[1])
        print('VALID',step,r,s,flush=True)
        torch.save({'model':model.state_dict(),'step':step,'random':r,'structured':s},f'/workspace/refine_{step}.pt')
        if score>=best_score:
            best_score=score; torch.save(model.state_dict(),'/workspace/refined_best.pt')
print('FINAL',accuracy(model,500000,'random'),accuracy(model,500000,'structured'),flush=True)
