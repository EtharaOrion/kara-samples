import argparse
import importlib.util
import math
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path('/workspace')
LO, HI = 10_000_000, 99_999_999
POW10 = torch.tensor([10 ** i for i in range(9)], device='cuda', dtype=torch.long)


class Model(nn.Module):
    def __init__(self, ff=4):
        super().__init__(); d=20
        self.token=nn.Embedding(11,d)
        self.pos_left=nn.Parameter(torch.randn(25,2)*.02)
        self.pos_right=nn.Parameter(torch.randn(2,d)*.02)
        self.norm_attn=nn.LayerNorm(d); self.norm_ff=nn.LayerNorm(d)
        self.queries=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(2)])
        self.keys=nn.Linear(d,5,bias=False); self.values=nn.Linear(d,5,bias=False)
        self.outputs=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(2)])
        self.ff1=nn.Linear(d,ff,bias=False); self.ff2=nn.Linear(ff,d,bias=False)
        self.final_norm=nn.LayerNorm(d); self.classifier=nn.Linear(d,10,bias=False)

    def forward(self,t):
        n,l=t.shape; x=self.token(t)+self.pos_left[:l]@self.pos_right
        mask=torch.ones(l,l,device=t.device,dtype=torch.bool).triu(1)
        for qproj,oproj in zip(self.queries,self.outputs):
            z=self.norm_attn(x)
            q=qproj(z).view(n,l,4,5).transpose(1,2)
            k=self.keys(z).unsqueeze(1); v=self.values(z).unsqueeze(1)
            s=(q@k.transpose(-2,-1))*(5**-.5)
            att=F.softmax(s.masked_fill(mask,-torch.inf),-1)@v
            x=x+oproj(att.transpose(1,2).reshape(n,l,20))
            x=x+self.ff2(F.gelu(self.ff1(self.norm_ff(x))))
        return self.classifier(self.final_norm(x))


def uniform(n):
    return torch.randint(LO,HI+1,(n,),device='cuda'),torch.randint(LO,HI+1,(n,),device='cuda')


def batch_pairs(n, structured=.38):
    a,b=uniform(n); m=int(n*structured)
    if not m: return a,b
    mode=torch.randint(0,8,(m,),device='cuda'); x=torch.randint(LO,HI+1,(m,),device='cuda')
    y=torch.randint(LO,HI+1,(m,),device='cuda')
    # Complements around powers/bounds, with jitter.
    targets=torch.tensor([20_000_000,50_000_000,90_000_000,100_000_000,110_000_000,150_000_000,190_000_000],device='cuda')
    target=targets[torch.randint(0,len(targets),(m,),device='cuda')]
    comp=(target-x+torch.randint(-12,13,(m,),device='cuda')).clamp(LO,HI)
    y=torch.where(mode<=1,comp,y)
    # Explicit trailing runs of 0/9, independently on each operand.
    run=torch.randint(1,8,(m,),device='cuda'); p=POW10[run]
    prefix=(x//p)*p
    suffix=torch.where((mode&1)==0,p-1,torch.zeros_like(p))
    sx=(prefix+suffix).clamp(LO,HI)
    run2=torch.randint(1,8,(m,),device='cuda'); p2=POW10[run2]
    sy=((y//p2)*p2+torch.where((mode&1)==1,p2-1,torch.zeros_like(p2))).clamp(LO,HI)
    x=torch.where((mode==2)|(mode==3),sx,x); y=torch.where((mode==2)|(mode==3),sy,y)
    # Rounded boundaries and nearby values.
    p3=POW10[torch.randint(1,8,(m,),device='cuda')]
    rounded=((x//p3)*p3+torch.randint(-10,11,(m,),device='cuda')).clamp(LO,HI)
    x=torch.where(mode==4,rounded,x)
    # Repeated and sparse patterns.
    reps=torch.tensor([11_111_111,22_222_222,33_333_333,44_444_444,55_555_555,66_666_666,77_777_777,88_888_888,99_999_999],device='cuda')
    rr=reps[torch.randint(0,9,(m,),device='cuda')]
    x=torch.where(mode==5,rr,x)
    sparse=(torch.randint(1,10,(m,),device='cuda')*POW10[torch.randint(7,8,(m,),device='cuda')] + torch.randint(0,10,(m,),device='cuda')*POW10[torch.randint(0,7,(m,),device='cuda')]).clamp(LO,HI)
    x=torch.where(mode==6,sparse,x)
    extrema=torch.where(torch.rand(m,device='cuda')<.5,LO+torch.randint(0,1001,(m,),device='cuda'),HI-torch.randint(0,1001,(m,),device='cuda'))
    x=torch.where(mode==7,extrema,x)
    a[:m]=x; b[:m]=y
    return a,b


def encode(a,b):
    n=a.numel(); da=(a[:,None]//POW10[:8])%10; db=(b[:,None]//POW10[:8])%10
    operands=torch.stack((da,db),2).reshape(n,16)
    out=((a+b)[:,None]//POW10)%10
    inp=torch.cat((operands,torch.full((n,1),10,device='cuda'),out[:,:8]),1)
    return inp,out


def eval_model(model,n=100000,structured=.0,bs=10000):
    model.eval(); good=0; margin=100.
    with torch.no_grad():
      for _ in range((n+bs-1)//bs):
        k=min(bs,n-good if False else bs); a,b=batch_pairs(k,structured)
        base,_=encode(a,b); seq=base[:,:17]
        for j in range(9):
            logits=model(seq)[:,-1]; pred=logits.argmax(1)
            vals=logits.topk(2,1).values; margin=min(margin,float((vals[:,0]-vals[:,1]).min()))
            seq=torch.cat((seq,pred[:,None]),1)
        digits=seq[:,17:26]; got=(digits*POW10).sum(1); good+=int((got==a+b).sum())
    model.train(); return good,n,margin


def train_stage(model,steps,lr,batch,structured,label):
    model.train(); opt=torch.optim.AdamW(model.parameters(),lr=lr,betas=(.9,.98),weight_decay=.01)
    start=time.time()
    for step in range(1,steps+1):
        a,b=batch_pairs(batch,structured); inp,target=encode(a,b)
        logits=model(inp)[:,16:25]
        loss=F.cross_entropy(logits.reshape(-1,10),target.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step%1000==0:
            print(label,step,'loss',round(float(loss),6),'sec',round(time.time()-start,1),flush=True)
    return model


def prune(model,new_width):
    old=model.ff1.out_features
    score=model.ff1.weight.detach().norm(dim=1)*model.ff2.weight.detach().norm(dim=0)
    keep=score.topk(new_width).indices.sort().values
    new=Model(new_width).cuda(); state=model.state_dict()
    for name,p in new.named_parameters():
        if name=='ff1.weight': p.data.copy_(model.ff1.weight.data[keep])
        elif name=='ff2.weight': p.data.copy_(model.ff2.weight.data[:,keep])
        else: p.data.copy_(state[name])
    print('prune',old,'->',new_width,'keep',keep.tolist()); return new


def save_checkpoint(model,name):
    torch.save(model.state_dict(),ROOT/name)


def export(model):
    template=(ROOT/'submission.py').read_text(); marker='_TRAINED = '
    start=template.index(marker)+len(marker); end=template.index('\n\n\ndef build_model',start)
    arrays=[]
    for p in model.parameters(): arrays.append(p.detach().float().cpu().reshape(-1).tolist())
    text=template[:start]+repr(arrays)+template[end:]
    (ROOT/'submission.py').write_text(text)
    print('exported',sum(p.numel() for p in model.parameters()),'parameters',len(text),'bytes')


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--resume'); ap.add_argument('--batch',type=int,default=8192); args=ap.parse_args()
    torch.manual_seed(3029); random.seed(3029); torch.backends.cuda.matmul.allow_tf32=True
    if args.resume:
        width=int(args.resume.split('w')[-1].split('.')[0]); model=Model(width).cuda(); model.load_state_dict(torch.load(ROOT/args.resume,map_location='cuda'))
    else:
        model=Model(4).cuda()
        train_stage(model,36000,2e-3,args.batch,.38,'teacher')
        train_stage(model,6000,8e-5,args.batch,.45,'stabilize')
        save_checkpoint(model,'model_w4.pt')
    print('w4 eval',eval_model(model,200000,0),eval_model(model,200000,.65))
    if model.ff1.out_features==4:
        model=prune(model,3); train_stage(model,18000,2e-5,args.batch,.5,'width3'); save_checkpoint(model,'model_w3.pt')
        print('w3 eval',eval_model(model,300000,0),eval_model(model,300000,.7))
    if model.ff1.out_features==3:
        model=prune(model,2); train_stage(model,30000,1e-5,args.batch,.5,'width2a'); train_stage(model,16000,3e-6,args.batch,.6,'width2b'); save_checkpoint(model,'model_w2.pt')
    print('w2 eval',eval_model(model,500000,0),eval_model(model,500000,.7)); export(model)

if __name__=='__main__': main()
