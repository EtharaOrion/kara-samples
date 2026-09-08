from pathlib import Path
import sys, torch
import torch.nn.functional as F
sys.path.insert(0,'/workspace')
import submission
from train import AdditionTransformer, random_batch, export

torch.manual_seed(8128)
source,_=submission.build_model()
model=AdditionTransformer().cuda()
model.load_state_dict(source.state_dict())
model.train()
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,betas=(.9,.98),weight_decay=0)
pow10=10**torch.arange(8,device='cuda')
edge=torch.tensor([10_000_000,10_000_001,10_000_009,10_000_010,10_000_099,10_000_100,10_000_999,10_001_000,10_009_999,10_010_000,10_099_999,10_100_000,10_999_999,11_111_111,19_999_999,20_000_000,49_999_999,50_000_000,88_888_888,89_999_999,90_000_000,98_999_999,99_000_000,99_900_000,99_990_000,99_999_000,99_999_900,99_999_990,99_999_998,99_999_999],device='cuda')
for step in range(1,2001):
    x,y=random_batch(4096,'cuda',.35)
    n=1024
    a=edge[torch.randint(len(edge),(n,),device='cuda')]
    b=edge[torch.randint(len(edge),(n,),device='cuda')]
    ad=(a[:,None]//pow10)%10; bd=(b[:,None]//pow10)%10
    ans=a+b; yd=(ans[:,None]//(10**torch.arange(9,device='cuda')))%10
    x[:n,0:16:2]=ad; x[:n,1:16:2]=bd; x[:n,16]=10; x[:n,17:]=yd[:,:-1]; y[:n]=yd
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        logits=model(x)[:,16:25]
        loss=F.cross_entropy(logits.reshape(-1,10),y.reshape(-1))
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step()
    if step%250==0: print(step,loss.item(),(logits.argmax(-1)==y).float().mean().item(),flush=True)
export(model,Path('/workspace/submission.py'))
