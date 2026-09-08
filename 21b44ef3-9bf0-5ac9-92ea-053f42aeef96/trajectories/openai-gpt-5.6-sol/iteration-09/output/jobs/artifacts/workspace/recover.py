import argparse, torch
import torch.nn.functional as F
from train import Model, batch, validate, export

ap=argparse.ArgumentParser(); ap.add_argument('--rank',type=int,required=True); ap.add_argument('--resume',required=True); ap.add_argument('--steps',type=int,default=8000); ap.add_argument('--lr',type=float,default=2e-5); ap.add_argument('--seed',type=int,default=91)
a=ap.parse_args(); torch.manual_seed(a.seed); torch.set_float32_matmul_precision('high')
m=Model(a.rank).cuda(); m.load_state_dict(torch.load(a.resume,weights_only=True)); opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=.001)
print('parameters',sum(p.numel() for p in m.parameters()),flush=True)
for step in range(1,a.steps+1):
 inp,target,_,_=batch(4096,.55)
 loss=F.cross_entropy(m(inp)[:,16:25].reshape(-1,10),target.reshape(-1))
 opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
 if step%1000==0:
  print(step,float(loss),validate(m,20000,0),validate(m,20000,.75),flush=True)
print('FINAL',validate(m,300000,0),validate(m,300000,.75),flush=True)
export(m,a.rank)
