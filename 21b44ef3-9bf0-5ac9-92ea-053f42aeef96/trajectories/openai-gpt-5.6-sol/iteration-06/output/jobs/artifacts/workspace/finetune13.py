import sys
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0,'/workspace')
from train import TinyAdder,batch_data,evaluate,export

torch.manual_seed(29)
torch.set_float32_matmul_precision('high')
m=TinyAdder(13).cuda(); m.load_state_dict(torch.load('/workspace/model_ff13.pt',weights_only=True))
opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=0.0)
best=0.0
for step in range(1,4001):
    if step==2001:
        for g in opt.param_groups:g['lr']=2e-5
    x,y=batch_data(4096,'cuda',0.5 if step<=2000 else 0.75)
    loss=F.cross_entropy(m(x)[:,16:25].reshape(-1,10),y.reshape(-1))
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    if step%250==0:
        r=evaluate(m,20000,0.0); e=evaluate(m,10000,0.8); print(step,loss.item(),r,e,flush=True)
        if min(r,e)>best:
            best=min(r,e); torch.save(m.state_dict(),'/workspace/model_ff13_fine.pt'); export(m,13,Path('/workspace/submission_ff13.py'))
print('best',best)
