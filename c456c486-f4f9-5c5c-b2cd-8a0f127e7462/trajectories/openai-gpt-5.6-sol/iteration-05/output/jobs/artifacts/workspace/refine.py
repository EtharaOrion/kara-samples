import os
os.environ.pop('PYTHONPATH',None)
import torch
import torch.nn.functional as F
from train import Model,batch,evaluate,export

torch.set_num_threads(16)
model=Model('none')
model.load_state_dict(torch.load('/workspace/model_none.pt',weights_only=True))
model.train()
opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.005)
for step in range(1,1001):
    x=batch(512,'cpu',structured=True)
    y=model(x)
    loss=F.cross_entropy(y[:,1::3].reshape(-1,10),x[:,2::3].reshape(-1))
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if step in (500,800):
        for g in opt.param_groups:g['lr']*=.4
    if step%250==0: print(step,float(loss),evaluate(model,500,12000+step),flush=True)
model.eval()
print('final',evaluate(model,3000,67123),flush=True)
export(model,'none')
torch.save(model.state_dict(),'/workspace/model_none.pt')
