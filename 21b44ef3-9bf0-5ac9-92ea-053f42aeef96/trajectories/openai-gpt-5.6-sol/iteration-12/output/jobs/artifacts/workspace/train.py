import argparse
import importlib.util
import math
import random
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

D=20

class TrainModel(nn.Module):
    def __init__(self, rank=8, pass_rank=2):
        super().__init__()
        self.rank=rank
        self.token=nn.Embedding(11,D)
        self.pos_left=nn.Parameter(torch.randn(25,rank)*.02)
        self.pos_right=nn.Parameter(torch.randn(rank,D)*.02)
        if pass_rank == 2:
            self.pass_full=nn.Parameter(torch.randn(2,D)*.02)
            self.pass_left=None; self.pass_right=None
        else:
            self.register_parameter('pass_full',None)
            self.pass_left=nn.Parameter(torch.randn(2,pass_rank)*.02)
            self.pass_right=nn.Parameter(torch.randn(pass_rank,D)*.02)
        self.norm1=nn.LayerNorm(D)
        self.attn=nn.MultiheadAttention(D,4,batch_first=True)
        self.norm2=nn.LayerNorm(D)
        self.ff1=nn.Linear(D,4)
        self.ff2=nn.Linear(4,D)
        self.final_norm=nn.LayerNorm(D)
        self.head=nn.Linear(D,10,bias=False)

    def passes(self):
        return self.pass_full if self.pass_full is not None else self.pass_left @ self.pass_right

    def forward(self,t):
        n=t.shape[1]
        x=self.token(t)+(self.pos_left[:n]@self.pos_right)
        mask=torch.triu(torch.ones(n,n,dtype=torch.bool,device=t.device),1)
        p=self.passes()
        for k in range(2):
            y=self.norm1(x+p[k])
            x=x+self.attn(y,y,y,attn_mask=mask,need_weights=False)[0]
            x=x+self.ff2(F.gelu(self.ff1(self.norm2(x))))
        return self.head(self.final_norm(x))

def uniform(n,dev):
    return torch.randint(10_000_000,100_000_000,(n,),device=dev), torch.randint(10_000_000,100_000_000,(n,),device=dev)

def structured(n,dev):
    # Diverse mixture targeting carries, decimal boundaries and range extremes.
    kind=torch.randint(0,7,(n,),device=dev)
    a,b=uniform(n,dev)
    m=kind==0
    if m.any():
        x=torch.randint(10_000_000,90_000_001,(int(m.sum()),),device=dev)
        delta=torch.randint(-999,1000,x.shape,device=dev)
        a[m]=x; b[m]=(100_000_000-x+delta).clamp(10_000_000,99_999_999)
    m=kind==1
    if m.any():
        z=int(m.sum()); a[m]=torch.randint(90_000_000,100_000_000,(z,),device=dev); b[m]=torch.randint(90_000_000,100_000_000,(z,),device=dev)
    m=kind==2
    if m.any():
        z=int(m.sum()); k=torch.randint(1,8,(z,),device=dev); scale=10**k
        x=torch.randint(10_000_000,100_000_000,(z,),device=dev)
        y=torch.randint(10_000_000,100_000_000,(z,),device=dev)
        a[m]=(x//scale*scale).clamp_min(10_000_000); b[m]=(y//scale*scale).clamp_min(10_000_000)
    m=kind==3
    if m.any():
        z=int(m.sum()); d1=torch.randint(1,10,(z,),device=dev); d2=torch.randint(1,10,(z,),device=dev)
        a[m]=d1*11_111_111; b[m]=d2*11_111_111
    m=kind==4
    if m.any():
        z=int(m.sum()); k=torch.randint(1,8,(z,),device=dev); scale=10**k
        raw=torch.randint(10_000_000,100_000_000,(z,),device=dev)
        x=(raw//scale*scale+(scale-1)).clamp(10_000_000,99_999_999)
        y=torch.randint(10_000_000,100_000_000,(z,),device=dev)
        a[m]=x; b[m]=y
    m=kind==5
    if m.any():
        z=int(m.sum()); choices=torch.tensor([10_000_000,10_000_001,10_000_009,10_000_010,90_000_000,99_000_000,99_900_000,99_990_000,99_999_000,99_999_900,99_999_990,99_999_999],device=dev)
        a[m]=choices[torch.randint(0,len(choices),(z,),device=dev)]; b[m]=choices[torch.randint(0,len(choices),(z,),device=dev)]
    # kind 6 deliberately remains uniform
    return a,b

def batch(n,dev,ratio):
    a,b=uniform(n,dev)
    count=int(n*ratio)
    if count:
        a[:count],b[:count]=structured(count,dev)
    # Decimal extraction is training-data construction only.
    powers=torch.tensor([10**i for i in range(8)],device=dev)
    ad=(a[:,None]//powers)%10; bd=(b[:,None]//powers)%10
    operands=torch.stack((ad,bd),2).reshape(n,16)
    s=a+b
    targets=(s[:,None]//torch.tensor([10**i for i in range(9)],device=dev))%10
    inp=torch.cat((operands,torch.full((n,1),10,device=dev),targets[:,:8]),1).long()
    return inp,targets.long(),a,b

def train_phase(model,steps,lr,ratio,batch_size,dev,label):
    model.train(); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
    scaler=None
    for step in range(1,steps+1):
        x,y,_,_=batch(batch_size,dev,ratio)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(x)[:,16:25]
            loss=F.cross_entropy(logits.reshape(-1,10),y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step%500==0 or step==1:
            acc=(logits.argmax(-1)==y).all(1).float().mean().item()
            print(label,step,steps,float(loss),acc,flush=True)
    return model

@torch.no_grad()
def eval_teacher(model,n,ratio,dev,bs=8192):
    model.eval(); ok=0; total=0; minmargin=100.
    for _ in range(math.ceil(n/bs)):
        z=min(bs,n-total); x,y,_,_=batch(z,dev,ratio)
        out=model(x)[:,16:25].float(); pred=out.argmax(-1)
        ok+=(pred==y).all(1).sum().item(); total+=z
        true=out.gather(2,y[:,:,None]).squeeze(2)
        alt=out.masked_fill(F.one_hot(y,10).bool(),-1e9).max(2).values
        minmargin=min(minmargin,(true-alt).min().item())
    return ok,total,minmargin

def compress(model,rank,pass_rank):
    out=TrainModel(rank,pass_rank).to(next(model.parameters()).device)
    common=['token','norm1','attn','norm2','ff1','ff2','final_norm','head']
    for name in common: getattr(out,name).load_state_dict(getattr(model,name).state_dict())
    pos=model.pos_left@model.pos_right
    u,s,v=torch.linalg.svd(pos.float(),full_matrices=False)
    out.pos_left.data.copy_((u[:,:rank]*s[:rank].sqrt()).to(out.pos_left.dtype)); out.pos_right.data.copy_((s[:rank,None].sqrt()*v[:rank]).to(out.pos_right.dtype))
    p=model.passes().float(); u,s,v=torch.linalg.svd(p,full_matrices=False)
    if pass_rank==2: out.pass_full.data.copy_(p)
    else:
        out.pass_left.data.copy_((u[:,:pass_rank]*s[:pass_rank].sqrt()).to(out.pass_left.dtype)); out.pass_right.data.copy_((s[:pass_rank,None].sqrt()*v[:pass_rank]).to(out.pass_right.dtype))
    return out

def lit(t):
    return repr(t.detach().cpu().float().tolist())

def export(model,path):
    template=Path('/workspace/submission.py').read_text()
    marker='\n_TRAINED_STATE = '
    if marker in template: template=template.split(marker)[0]
    state=model.state_dict()
    text=template+marker+'{\n'+''.join(f'    {k!r}: torch.tensor({lit(v)}),\n' for k,v in state.items())+'}\n\n_original_build_model = build_model\ndef build_model():\n    model, metadata = _original_build_model()\n    model.load_state_dict(_TRAINED_STATE)\n    model.eval()\n    return model, metadata\n'
    Path(path).write_text(text)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base-steps',type=int,default=26000); ap.add_argument('--batch',type=int,default=4096); args=ap.parse_args()
    torch.manual_seed(7231); random.seed(7231); torch.set_float32_matmul_precision('high'); dev='cuda'
    m=TrainModel(8,2).to(dev)
    train_phase(m,args.base_steps,2e-3,.20,args.batch,dev,'base')
    train_phase(m,6000,1e-4,.45,args.batch,dev,'stabilize')
    print('base eval',eval_teacher(m,200000,.5,dev),flush=True); torch.save(m.state_dict(),'/workspace/base.pt')
    m=compress(m,7,2); train_phase(m,9000,2e-5,.60,args.batch,dev,'rank7')
    print('rank7 eval',eval_teacher(m,300000,.65,dev),flush=True); torch.save(m.state_dict(),'/workspace/rank7.pt')
    m=compress(m,7,1); train_phase(m,7000,1e-5,.65,args.batch,dev,'pass1')
    print('final eval',eval_teacher(m,500000,.7,dev),flush=True); torch.save(m.state_dict(),'/workspace/final.pt'); export(m,'/workspace/submission.py')

if __name__=='__main__': main()
