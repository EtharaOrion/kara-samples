import torch
from train import Model,evaluate
base=Model(4).cuda(); base.load_state_dict(torch.load('/workspace/teacher_stable.pt',weights_only=True))
print('base',evaluate(base,50000,.5),flush=True)
state=base.state_dict()
for remove in range(4):
 m=Model(3).cuda(); d=m.state_dict(); keep=torch.tensor([i for i in range(4) if i!=remove],device='cuda')
 for name in d:
  if name=='ff1.weight': d[name].copy_(state[name][keep])
  elif name=='ff2.weight': d[name].copy_(state[name][:,keep])
  else: d[name].copy_(state[name])
 print('remove',remove,evaluate(m,50000,.5),flush=True)
 torch.save(m.state_dict(),f'/workspace/candidate{remove}.pt')
