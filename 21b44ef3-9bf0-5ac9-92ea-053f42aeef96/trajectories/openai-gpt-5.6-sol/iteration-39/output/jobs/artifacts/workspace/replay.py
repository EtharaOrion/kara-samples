import torch
import torch.nn.functional as F
from compress_export import StandardModel
from train import make_batch, digits

model=StandardModel(2).cuda()
model.load_state_dict(torch.load('/workspace/final.pt',weights_only=True))
opt=torch.optim.AdamW(model.parameters(),lr=1e-6,betas=(.9,.98),weight_decay=.001)
# Include observed patterns and systematic leading-digit variants with long 9 suffixes.
hard=[]
for x in range(1,10):
 for y in range(1,10):
  for suffix in (999_999,9_999_999,99_999,9999,999):
   for dx in (-1,0,1):
    a=x*10_000_000+suffix+dx
    b=y*10_000_000+suffix-dx
    if 10_000_000<=a<=99_999_999 and 10_000_000<=b<=99_999_999: hard.append((a,b))
hard += [(69_999_999,36_495_978),(10_999_999,50_999_999),(40_999_999,30_999_999),(90_999_999,80_999_999)]
hard=torch.tensor(hard,device='cuda')
for step in range(1,6001):
 x,y=make_batch(8192,.35)
 n=1024; ix=torch.randint(0,len(hard),(n,),device='cuda'); pairs=hard[ix]
 if step%2: pairs=pairs.flip(1)
 target=digits(pairs[:,0]+pairs[:,1])
 x[:n,0:16:2]=digits(pairs[:,0])[:,:8]; x[:n,1:16:2]=digits(pairs[:,1])[:,:8]
 x[:n,17:]=target[:,:8]; y[:n]=target
 logits=model(x)[:,16:25]
 loss=F.cross_entropy(logits.reshape(-1,10),y.reshape(-1))
 opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
 if step%1000==0: print(step,loss.item(),flush=True)
torch.save(model.state_dict(),'/workspace/final.pt')
