import sys,time,torch
sys.path.insert(0,'/workspace')
from submission import AdditionTransformer
from train import to_digits, uniform, carry_examples, boundary_examples, evaluate, POWERS
import torch.nn.functional as F
B=4096

def contrasts(n):
    start=torch.randint(0,14,(n,),device='cuda')
    maxlen=14-start
    length=(torch.rand(n,device='cuda')*maxlen).long()+1
    scale=POWERS[start]
    run=(POWERS[length]-1)*scale
    d=torch.randint(1,10,(n,),device='cuda')*scale
    kind=torch.randint(0,4,(n,),device='cuda')
    # Full 9-run plus small digit; and close non-carry boundaries with shifted runs.
    a=torch.where(kind<2,run,(POWERS[length]-2)*scale)
    b=torch.where(kind==0,d,scale)
    b=torch.where(kind==2,scale,b)
    b=torch.where(kind==3,torch.maximum(scale,scale*torch.randint(1,3,(n,),device='cuda')),b)
    return a,b

m=AdditionTransformer().cuda();m.load_state_dict(torch.load('/workspace/best.pt',weights_only=True)['model']);m.train()
opt=torch.optim.AdamW(m.parameters(),lr=1e-5,weight_decay=0.001)
best=None;bestscore=10**9;start_time=time.time()
for step in range(16000):
 a0,b0=uniform(B//2);a1,b1=contrasts(B//4);a2,b2=carry_examples(B-B//2-B//4)
 a=torch.cat((a0,a1,a2));b=torch.cat((b0,b1,b2));ix=torch.randperm(B,device='cuda');a=a[ix];b=b[ix]
 logits=m(to_digits(a),to_digits(b));target=to_digits(a+b)
 loss=F.cross_entropy(logits.reshape(-1,10),target.reshape(-1))
 opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
 if (step+1)%2000==0:
  # Fresh random and fresh randomized contrast distributions.
  er=evaluate(m,131072,False)
  with torch.no_grad():
   ca,cb=contrasts(65536); ec=(m(to_digits(ca),to_digits(cb)).argmax(-1).ne(to_digits(ca+cb)).any(1)).sum().item()
  print(step+1,loss.item(),er,ec,round(time.time()-start_time),flush=True)
  score=er+4*ec
  if score<=bestscore:bestscore=score;best={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};torch.save({'model':best,'step':step+1,'random_errors':er,'contrast_errors':ec},'/workspace/calibrated.pt')
