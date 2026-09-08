import sys, torch
from torch import nn
sys.path.insert(0,'/workspace')
import submission, train
LO,HI=10_000_000,99_999_999
model,_=submission.build_model(); model.cuda().train()
opt=torch.optim.AdamW(model.parameters(),lr=1e-5,betas=(.9,.98),weight_decay=0)
lossfn=nn.CrossEntropyLoss()
B=8192
for step in range(1,2001):
    a=torch.randint(LO,HI+1,(B,),device='cuda'); b=torch.randint(LO,HI+1,(B,),device='cuda')
    # Exact complements spanning the range.
    x=torch.randint(LO,90_000_001,(B//4,),device='cuda'); a[:B//4]=x; b[:B//4]=100_000_000-x
    # Repeated-digit and nearby complements, both operand orders.
    d=torch.randint(1,10,(B//4,),device='cuda'); delta=torch.randint(-20,21,(B//4,),device='cuda')
    x=(d*11_111_111+delta).clamp(LO,89_999_999); y=100_000_000-x
    swap=torch.rand(B//4,device='cuda')<.5
    a[B//4:B//2]=torch.where(swap,x,y); b[B//4:B//2]=torch.where(swap,y,x)
    inp,target=train.encode(a,b); logits=model(inp); loss=lossfn(logits.reshape(-1,10),target.reshape(-1))
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
    if step%200==0:
        ra,_=train.evaluate(model,20000,0); ea,_=train.evaluate(model,20000,.8)
        print(step,float(loss),ra,ea,flush=True)
train.export(model,'/workspace/submission.py')
