import copy, sys, time, torch
sys.path.insert(0,'/workspace')
from submission import Model
from train_ar import make, ev, export
m=Model().cuda().train(); m.load_state_dict(torch.load('/workspace/best_ar.pt',weights_only=True))
opt=torch.optim.AdamW(m.parameters(),lr=1e-5,betas=(.9,.98),weight_decay=.001)
best=copy.deepcopy(m.state_dict()); score=10**9; t=time.time()
for s in range(60000):
 a,b,y=make(120000+s); z=m(a,b,y[:,:-1])[:,14:]; loss=torch.nn.functional.cross_entropy(z.reshape(-1,10),y.reshape(-1)); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1); opt.step()
 if (s+1)%10000==0:
  eu=ev(m,131072); es=ev(m,131072,'structured'); sc=eu[0]*4+es[0]; print(s+1,float(loss),eu,es,sc,'sec',round(time.time()-t),flush=True)
  if sc<score: score=sc; best=copy.deepcopy(m.state_dict()); torch.save(best,'/workspace/refined.pt')
m.load_state_dict(best); print('final-u',ev(m,1048576),flush=True); print('final-s',ev(m,524288,'structured'),flush=True); export(m)
