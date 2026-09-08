import importlib.util
from pathlib import Path
import torch
from torch.nn import functional as F

ROOT=Path('/workspace')
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
t=load('trainer',ROOT/'train.py'); s=load('sub',ROOT/'submission.py')
torch.set_float32_matmul_precision('high'); torch.manual_seed(240024)
model=s.AdditionTransformer(2).cuda(); model.load_state_dict(torch.load(ROOT/'final2.pt',weights_only=True))
vals=set(t.EDGE_VALUES.cpu().tolist())
for p in range(1,8):
 for base in (10_000_000,20_000_000,50_000_000,90_000_000,100_000_000):
  for delta in (-10**p-1,-10**p,-10**p+1,-99,-10,-9,-1,0,1,9,10,99,10**p-1,10**p,10**p+1):
   if 10_000_000 <= base+delta < 100_000_000: vals.add(base+delta)
HARD=torch.tensor(sorted(vals),device='cuda')
BASE_BATCH=t.make_batch

def hard_batch(batch=t.BATCH,structured=.35):
 x,y=BASE_BATCH(batch,structured)
 n=batch//5
 a=HARD[torch.randint(0,len(HARD),(n,),device='cuda')]
 b=torch.randint(10_000_000,100_000_000,(n,),device='cuda')
 # Half hard-hard; half hard paired to near complements or arbitrary values.
 b[:n//3]=HARD[torch.randint(0,len(HARD),(n//3,),device='cuda')]
 target=100_000_000+torch.randint(-1001,1002,(n//3,),device='cuda')
 candidate=target-a[n//3:2*n//3]
 b[n//3:2*n//3]=torch.where((candidate>=10_000_000)&(candidate<100_000_000),candidate,b[n//3:2*n//3])
 rd=t.digits(a+b,9); ad=t.digits(a,8); bd=t.digits(b,8)
 x[:n,0:16:2]=ad; x[:n,1:16:2]=bd; x[:n,16]=10; x[:n,17:]=rd[:,:8]; y[:n]=rd
 return x,y

t.make_batch=hard_batch
opt=torch.optim.AdamW(model.parameters(),lr=5e-6,betas=(.9,.98),weight_decay=.01)
for step in range(1,20001):
 x,y=hard_batch(); loss=F.cross_entropy(model(x)[:,16:25].reshape(-1,10),y.reshape(-1))
 opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
 if step==1 or step%1000==0:
  ar,m=t.accuracy(model,8,0); ae,_=t.accuracy(model,8,.75)
  print(f'hard {step} loss={loss.item():.6f} random={ar:.6f} edge={ae:.6f} margin={m:.4f}',flush=True)
  torch.save(model.state_dict(),ROOT/'hard_latest.pt')
# Low-rate consolidation.
for g in opt.param_groups:g['lr']=1e-6
for step in range(1,5001):
 x,y=hard_batch(); loss=F.cross_entropy(model(x)[:,16:25].reshape(-1,10),y.reshape(-1))
 opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
 if step%1000==0: print(f'finish {step} loss={loss.item():.6f}',flush=True)
torch.save(model.state_dict(),ROOT/'hard_final.pt'); t.export(model)
