import importlib.util
from pathlib import Path
import torch
from torch.nn import functional as F
ROOT=Path('/workspace')
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
t=load('trainer',ROOT/'train.py');s=load('sub',ROOT/'submission.py')
torch.set_float32_matmul_precision('high');torch.manual_seed(240025)
model=s.AdditionTransformer(2).cuda();model.load_state_dict(torch.load(ROOT/'hard_final.pt',weights_only=True))
opt=torch.optim.AdamW(model.parameters(),lr=3e-6,betas=(.9,.98),weight_decay=.005)

def batch():
 x,y=t.make_batch(t.BATCH,.25);n=t.BATCH//2
 # Random high prefixes combined with complementary low suffixes. This spans
 # every carry start/stop column rather than memorizing particular operands.
 p=torch.tensor([10**i for i in range(2,8)],device='cuda')[torch.randint(0,6,(n,),device='cuda')]
 ah=torch.randint(10_000_000,100_000_000,(n,),device='cuda');bh=torch.randint(10_000_000,100_000_000,(n,),device='cuda')
 low=torch.randint(1,100,(n,),device='cuda')
 a=ah-ah%p+low
 b=bh-bh%p+(p-low)+torch.randint(-1,2,(n,),device='cuda')
 valid=(a>=10_000_000)&(a<100_000_000)&(b>=10_000_000)&(b<100_000_000)
 a=torch.where(valid,a,ah);b=torch.where(valid,b,bh)
 # Include the mirrored orientation naturally by swapping half.
 swap=torch.rand(n,device='cuda')<.5;aa=torch.where(swap,b,a);bb=torch.where(swap,a,b)
 rd=t.digits(aa+bb,9);ad=t.digits(aa,8);bd=t.digits(bb,8)
 x[:n,0:16:2]=ad;x[:n,1:16:2]=bd;x[:n,16]=10;x[:n,17:]=rd[:,:8];y[:n]=rd
 return x,y
for step in range(1,16001):
 x,y=batch();loss=F.cross_entropy(model(x)[:,16:25].reshape(-1,10),y.reshape(-1))
 opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
 if step==1 or step%1000==0:
  ar,m=t.accuracy(model,8,0);ae,_=t.accuracy(model,8,.8)
  print(f'adv {step} loss={loss.item():.6f} random={ar:.6f} structured={ae:.6f} margin={m:.4f}',flush=True)
  torch.save(model.state_dict(),ROOT/'adversarial_latest.pt')
for g in opt.param_groups:g['lr']=5e-7
for step in range(3000):
 x,y=batch();loss=F.cross_entropy(model(x)[:,16:25].reshape(-1,10),y.reshape(-1));opt.zero_grad(set_to_none=True);loss.backward();opt.step()
torch.save(model.state_dict(),ROOT/'adversarial_final.pt');t.export(model)
