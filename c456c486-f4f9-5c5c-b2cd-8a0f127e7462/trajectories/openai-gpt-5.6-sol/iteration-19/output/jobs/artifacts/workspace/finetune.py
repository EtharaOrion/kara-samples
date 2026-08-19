import sys, torch
import torch.nn.functional as F
sys.path.insert(0,'/workspace')
import submission
from train import digits, evaluate, export, LIMIT, DEVICE, BATCH, POW10

def mixed(batch):
    a=torch.randint(0,LIMIT,(batch,),device=DEVICE); b=torch.randint(0,LIMIT,(batch,),device=DEVICE)
    n=batch//2; q=n//8
    # Small values / leading zeros
    for j,cap in enumerate((10,1000,1000000)):
        a[j*q:(j+1)*q]=torch.randint(0,cap,(q,),device=DEVICE); b[j*q:(j+1)*q]=torch.randint(0,cap,(q,),device=DEVICE)
    # Complementary complete pairs
    x=torch.randint(0,LIMIT,(q,),device=DEVICE); a[3*q:4*q]=x; b[3*q:4*q]=LIMIT-1-x
    # repeated digits, including repeated+repeated
    d1=torch.randint(0,10,(q,),device=DEVICE); d2=torch.randint(0,10,(q,),device=DEVICE)
    a[4*q:5*q]=sum(d1*10**i for i in range(14)); b[4*q:5*q]=sum(d2*10**i for i in range(14))
    # explicit randomized carry chain at arbitrary place
    z=n-5*q; length=torch.randint(1,15,(z,),device=DEVICE); start=torch.minimum(torch.randint(0,14,(z,),device=DEVICE),14-length)
    p=POW10[start]; run=(POW10[length]-1)*p
    a[5*q:n]=run; b[5*q:n]=p
    swap=torch.rand(n,device=DEVICE)<.5; aa=a[:n].clone(); a[:n]=torch.where(swap,b[:n],a[:n]); b[:n]=torch.where(swap,aa,b[:n])
    target=digits(a+b,15); source=torch.cat((digits(a,14),digits(b,14)),1)
    previous=torch.cat((torch.full((batch,1),10,device=DEVICE),target[:,:-1]),1)
    return source,previous,target

m,_=submission.build_model();m.to(DEVICE).train(); opt=torch.optim.AdamW(m.parameters(),lr=8e-5,weight_decay=.001)
for step in range(1,8001):
    if step==4001:
        for g in opt.param_groups:g['lr']=3e-5
    s,p,t=mixed(BATCH); logits=m(s,p); loss=F.cross_entropy(logits.reshape(-1,10),t.reshape(-1))
    opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
    if step%1000==0:
        rb,rt=evaluate(m,2,4096,0); print(step,float(loss),rb,rt,flush=True);export(m,'/workspace/submission.py')
export(m,'/workspace/submission.py')
print('final',evaluate(m,16,8192,0),evaluate(m,16,8192,.8))
