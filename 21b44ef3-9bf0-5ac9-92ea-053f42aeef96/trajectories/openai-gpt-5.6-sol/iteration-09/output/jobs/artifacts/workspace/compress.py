import sys, torch
sys.path.insert(0, '/workspace')
from train import Model, validate
state=torch.load('/workspace/rank8.pt', weights_only=True)
full=state['position_left'] @ state['position_right']
u,s,vh=torch.linalg.svd(full, full_matrices=False)
print('singular',s.cpu().tolist())
for rank in range(1,8):
 m=Model(rank).cuda(); sd=m.state_dict()
 for k in sd:
  if k not in ('position_left','position_right'): sd[k]=state[k]
 root=torch.sqrt(s[:rank])
 sd['position_left']=u[:,:rank]*root
 sd['position_right']=root[:,None]*vh[:rank]
 m.load_state_dict(sd)
 print(rank, sum(p.numel() for p in m.parameters()), validate(m,20000,0), validate(m,20000,.75), flush=True)
 torch.save(m.state_dict(),f'/workspace/rank{rank}_init.pt')
