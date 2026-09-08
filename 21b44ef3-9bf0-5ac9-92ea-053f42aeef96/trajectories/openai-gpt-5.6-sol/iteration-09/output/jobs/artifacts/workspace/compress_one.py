import sys, torch
sys.path.insert(0,'/workspace')
from train import Model,validate
src=torch.load('/workspace/rank7.pt',weights_only=True); pos=src['position_left']@src['position_right']; u,s,v=torch.linalg.svd(pos,full_matrices=False); r=6
m=Model(r).cuda(); dst=m.state_dict()
for k in dst:
 if k not in ('position_left','position_right'): dst[k]=src[k]
root=s[:r].sqrt(); dst['position_left']=u[:,:r]*root; dst['position_right']=root[:,None]*v[:r]; m.load_state_dict(dst)
print(s.tolist()); print(validate(m,100000,0),validate(m,100000,.75)); torch.save(m.state_dict(),'/workspace/rank6_from7.pt')
