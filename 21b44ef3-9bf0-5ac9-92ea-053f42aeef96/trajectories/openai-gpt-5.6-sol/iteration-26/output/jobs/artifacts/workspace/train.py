import math
import random
from pathlib import Path
import torch
from torch import nn

DEVICE = "cuda"
BATCH = 8192
LO = 10_000_000
HI = 99_999_999
BASE = 100_000_000


class Model(nn.Module):
    def __init__(self, ff_width=4, pos_rank=2):
        super().__init__()
        d = 20
        self.ff_width = ff_width
        self.token = nn.Embedding(11, d)
        self.pos_left = nn.Parameter(torch.empty(25, pos_rank))
        self.pos_right = nn.Parameter(torch.empty(pos_rank, d))
        self.norm_attn = nn.LayerNorm(d)
        self.norm_ff = nn.LayerNorm(d)
        self.final_norm = nn.LayerNorm(d)
        self.queries = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(2)])
        self.outputs = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(2)])
        self.key = nn.Linear(d, 5, bias=False)
        self.value = nn.Linear(d, 5, bias=False)
        self.ff1 = nn.Linear(d, ff_width, bias=False)
        self.ff2 = nn.Linear(ff_width, d, bias=False)
        self.head = nn.Linear(d, 10, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.token.weight, std=.2)
        nn.init.normal_(self.pos_left, std=.2)
        nn.init.normal_(self.pos_right, std=.2)
        for m in list(self.queries)+list(self.outputs)+[self.key,self.value,self.ff1,self.ff2,self.head]:
            nn.init.xavier_uniform_(m.weight)

    def forward(self, tokens):
        n = tokens.shape[1]
        x = self.token(tokens) + self.pos_left[:n] @ self.pos_right
        mask = torch.ones(n,n,dtype=torch.bool,device=tokens.device).triu(1)
        for qproj,oproj in zip(self.queries,self.outputs):
            z=self.norm_attn(x)
            q=qproj(z).view(tokens.shape[0],n,4,5).transpose(1,2)
            k=self.key(z).unsqueeze(1); v=self.value(z).unsqueeze(1)
            s=(q@k.transpose(-2,-1))*(5**-.5)
            att=torch.softmax(s.masked_fill(mask,-torch.inf),-1)@v
            x=x+oproj(att.transpose(1,2).reshape(tokens.shape[0],n,20))
            x=x+self.ff2(torch.nn.functional.gelu(self.ff1(self.norm_ff(x))))
        return self.head(self.final_norm(x))


def digits(x, count=8):
    out=[]
    for _ in range(count):
        out.append(x.remainder(10)); x=x.div(10,rounding_mode='floor')
    return torch.stack(out,1)


def make_pairs(n, structured=.35):
    a=torch.randint(LO,HI+1,(n,),device=DEVICE)
    b=torch.randint(LO,HI+1,(n,),device=DEVICE)
    m=int(n*structured)
    if not m: return a,b
    kinds=torch.randint(0,8,(m,),device=DEVICE)
    aa=torch.randint(LO,HI+1,(m,),device=DEVICE)
    bb=torch.randint(LO,HI+1,(m,),device=DEVICE)
    # Exact and near complements exercise full carry chains.
    sel=kinds==0
    bb[sel]=(BASE-aa[sel]).clamp(LO,HI)
    sel=kinds==1
    delta=torch.randint(-20,21,(m,),device=DEVICE)
    bb[sel]=(BASE-aa[sel]+delta[sel]).clamp(LO,HI)
    # Unequal trailing runs of 0/9 and carry starts/stops at every column.
    sel=kinds==2
    p10=(10**torch.randint(1,8,(m,),device=DEVICE)).long()
    aa2=(aa.div(p10,rounding_mode='floor')*p10 + torch.randint(0,10,(m,),device=DEVICE)).clamp(LO,HI)
    bb2=(bb.div(p10,rounding_mode='floor')*p10 + p10-1).clamp(LO,HI)
    aa[sel]=aa2[sel]; bb[sel]=bb2[sel]
    sel=kinds==3
    p10b=(10**torch.randint(1,8,(m,),device=DEVICE)).long()
    aa2=(aa.div(p10b,rounding_mode='floor')*p10b).clamp(LO,HI)
    bb2=(bb.div(p10b,rounding_mode='floor')*p10b + p10b-1).clamp(LO,HI)
    aa[sel]=aa2[sel]; bb[sel]=bb2[sel]
    # Decimal boundaries, extrema, repeated and sparse-ish patterns.
    sel=kinds==4
    p=(10**torch.randint(4,9,(m,),device=DEVICE)).long()
    aa2=(torch.randint(1,10,(m,),device=DEVICE)*p-1).clamp(LO,HI)
    bb2=(torch.randint(1,10,(m,),device=DEVICE)*p+torch.randint(-9,10,(m,),device=DEVICE)).clamp(LO,HI)
    aa[sel]=aa2[sel]; bb[sel]=bb2[sel]
    reps=torch.tensor([11_111_111,22_222_222,33_333_333,44_444_444,55_555_555,66_666_666,77_777_777,88_888_888,99_999_999],device=DEVICE)
    sel=kinds==5
    aa[sel]=reps[torch.randint(0,9,(m,),device=DEVICE)][sel]
    bb[sel]=reps[torch.randint(0,9,(m,),device=DEVICE)][sel]
    sel=kinds==6
    aa2=torch.randint(10,100,(m,),device=DEVICE)*1_000_000+torch.randint(0,20,(m,),device=DEVICE)
    bb2=torch.randint(10,100,(m,),device=DEVICE)*1_000_000+(999_999-torch.randint(0,20,(m,),device=DEVICE))
    aa[sel]=aa2[sel].clamp(LO,HI);bb[sel]=bb2[sel].clamp(LO,HI)
    sel=kinds==7
    ends=torch.tensor([LO,LO+1,LO+9,LO+99,HI,HI-1,HI-9,HI-99],device=DEVICE)
    aa[sel]=ends[torch.randint(0,len(ends),(m,),device=DEVICE)][sel]
    bfix=torch.randint(LO,HI+1,(m,),device=DEVICE); bb[sel]=bfix[sel]
    a[:m]=aa;b[:m]=bb
    return a,b


def batch(n=BATCH,structured=.35):
    a,b=make_pairs(n,structured)
    da=digits(a);db=digits(b); ds=digits(a+b,9)
    x=torch.empty(n,25,dtype=torch.long,device=DEVICE)
    x[:,0:16:2]=da;x[:,1:16:2]=db;x[:,16]=10;x[:,17:]=ds[:,:8]
    return x,ds


def accuracy(model,n=100000,structured=.0,chunk=10000):
    model.eval();good=total=0; margin=99.
    with torch.no_grad():
      for _ in range((n+chunk-1)//chunk):
        k=min(chunk,n-total);x,y=batch(k,structured)
        logits=model(x)[:,16:25]
        pred=logits.argmax(-1)
        good+=(pred==y).all(1).sum().item();total+=k
        top=logits.topk(2,-1).values
        margin=min(margin,(top[...,0]-top[...,1]).min().item())
    model.train();return good/total,margin


def train_stage(model,steps,lr,structured,label,weight_decay=0.01):
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay,betas=(.9,.98))
    for step in range(1,steps+1):
        x,y=batch(structured=structured)
        logits=model(x)[:,16:25]
        loss=nn.functional.cross_entropy(logits.reshape(-1,10),y.reshape(-1))
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        if step%2000==0:
            print(label,step,float(loss),accuracy(model,10000,0)[0],accuracy(model,10000,.7)[0],flush=True)
    return model


def prune(model,new_width):
    score=model.ff1.weight.norm(dim=1)*model.ff2.weight.norm(dim=0)
    keep=score.topk(new_width).indices.sort().values
    out=Model(new_width).to(DEVICE)
    state=model.state_dict(); ns=out.state_dict()
    for k in ns:
        if k=='ff1.weight': ns[k].copy_(state[k][keep])
        elif k=='ff2.weight': ns[k].copy_(state[k][:,keep])
        else: ns[k].copy_(state[k])
    out.load_state_dict(ns);return out


def export(model,path='/workspace/submission.py'):
    template=Path('/workspace/submission.py').read_text()
    start=template.index('_WEIGHTS =')
    end=template.index('\n\n\ndef build_model',start)
    state={k: v.detach().cpu() for k, v in model.state_dict().items()}
    lines=['_WEIGHTS = {']
    for name,t in state.items():
        vals=repr(t.float().reshape(-1).tolist())
        lines.append(f"    {name!r}: torch.tensor({vals}).reshape({tuple(t.shape)}),")
    lines.append('}')
    text=template[:start]+'\n'.join(lines)+template[end:]
    # Ensure constructor matches compressed width.
    text=text.replace('AdditionTransformer(4)','AdditionTransformer(%d)'%model.ff_width).replace('AdditionTransformer(3)','AdditionTransformer(%d)'%model.ff_width)
    Path(path).write_text(text)


def main():
    torch.manual_seed(2605); random.seed(2605)
    torch.set_float32_matmul_precision('high')
    source=Model(2,2).to(DEVICE)
    source.load_state_dict(torch.load('/workspace/width2.pt',weights_only=True))
    target=Model(2,1).to(DEVICE)
    old=source.state_dict(); new=target.state_dict()
    for k in new:
        if k not in ('pos_left','pos_right'): new[k].copy_(old[k])
    u,s,v=torch.linalg.svd(old['pos_left']@old['pos_right'],full_matrices=False)
    new['pos_left'].copy_(u[:,:1]*s[:1].sqrt());new['pos_right'].copy_(s[:1,None].sqrt()*v[:1])
    target.load_state_dict(new)
    print('rank1 initial',sum(p.numel() for p in target.parameters()),accuracy(target,100000,0),accuracy(target,100000,.7),flush=True)
    train_stage(target,24000,1e-5,.45,'r1a')
    train_stage(target,12000,3e-6,.6,'r1b')
    print('rank1 final',accuracy(target,1000000,0),accuracy(target,500000,.75),flush=True)
    torch.save(target.state_dict(),'/workspace/rank1.pt')

if __name__=='__main__': main()
