import torch
import torch.nn.functional as F
from train import AdditionTransformer, make_batch, curated, evaluate, eval_curated, export

torch.manual_seed(1416);torch.backends.cuda.matmul.allow_tf32=True
d=torch.device('cuda');m=AdditionTransformer().to(d);m.load_state_dict(torch.load('/workspace/cal_best.pt',weights_only=True))
T,Y,_=curated(d);opt=torch.optim.AdamW(m.parameters(),lr=5e-7,weight_decay=0);best=-1
print('initial',evaluate(m,100000,structured=0),eval_curated(m)[0])
for s in range(1,3001):
 t,y,_,_=make_batch(2048,d,.25);k=1433;i=torch.randint(len(T),(k,),device=d);t[:k]=T[i];y[:k]=Y[i]
 opt.zero_grad(set_to_none=True);z=m(t);ce=F.cross_entropy(z.flatten(0,1),y.flatten(),reduction='none').view(-1,9);loss=(ce.mean(1)+.4*ce.max(1).values).mean();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
 if s%250==0:
  r=evaluate(m,20000,structured=0)[0];e=eval_curated(m)[0];print(s,r,e,flush=True)
  if min(r,e)>best:best=min(r,e);torch.save(m.state_dict(),'/workspace/cal2_best.pt')
m.load_state_dict(torch.load('/workspace/cal2_best.pt',weights_only=True));torch.save(m.state_dict(),'/workspace/final.pt');export(m,'/workspace/submission.py')
print('FINAL',evaluate(m,500000,structured=0),evaluate(m,200000,structured=.7),eval_curated(m))
