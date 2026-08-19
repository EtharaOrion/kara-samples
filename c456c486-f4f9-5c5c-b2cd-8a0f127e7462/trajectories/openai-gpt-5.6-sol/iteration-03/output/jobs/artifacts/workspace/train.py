import argparse, math
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

MAX = 100_000_000_000_000

class Block(nn.Module):
    def __init__(self, d, hidden, heads):
        super().__init__(); self.heads=heads; self.dk=d//heads
        self.norm1=nn.LayerNorm(d); self.qkv=nn.Linear(d,3*d); self.proj=nn.Linear(d,d)
        self.norm2=nn.LayerNorm(d); self.ff1=nn.Linear(d,hidden); self.ff2=nn.Linear(hidden,d)
    def forward(self,x,mask):
        B,N,D=x.shape; z=self.norm1(x); q,k,v=self.qkv(z).chunk(3,-1)
        q=q.view(B,N,self.heads,self.dk).transpose(1,2); k=k.view(B,N,self.heads,self.dk).transpose(1,2); v=v.view(B,N,self.heads,self.dk).transpose(1,2)
        att=torch.softmax((q@k.transpose(-2,-1)/math.sqrt(self.dk)).masked_fill(~mask,-1e4),-1)
        x=x+self.proj((att@v).transpose(1,2).reshape(B,N,D))
        return x+self.ff2(F.gelu(self.ff1(self.norm2(x))))

class AddTransformer(nn.Module):
    def __init__(self,d=16,hidden=32,heads=2):
        super().__init__(); self.d=d; self.hidden=hidden; self.heads=heads
        self.digit=nn.Embedding(10,d); self.role=nn.Embedding(3,d)
        self.blocks=nn.ModuleList([Block(d,hidden,heads),Block(d,hidden,heads)])
        self.out_norm=nn.LayerNorm(d); self.head=nn.Linear(d,10)
    def forward(self,tokens):
        n=tokens.shape[1]; p=torch.arange(n,device=tokens.device); x=self.digit(tokens)+self.role(p%3)
        mask=((p[:,None]>=p[None,:])&(p[:,None]-p[None,:]<=5))[None,None]
        for block in self.blocks:x=block(x,mask)
        return self.head(self.out_norm(x))

def batch_data(batch,device):
    a=torch.randint(0,10,(batch,14),device=device); b=torch.randint(0,10,(batch,14),device=device)
    # Half the complete random pairs use repeated digit pairs, giving rare
    # column combinations much denser supervision without enumerating states.
    n=batch//2
    a[:n]=torch.randint(0,10,(n,1),device=device).expand(n,14)
    b[:n]=torch.randint(0,10,(n,1),device=device).expand(n,14)
    carry=torch.zeros(batch,dtype=torch.long,device=device); outs=[]
    for i in range(14):
        s=a[:,i]+b[:,i]+carry; outs.append(s%10); carry=s//10
    outs.append(carry); o=torch.stack(outs,1); z=torch.zeros(batch,1,dtype=torch.long,device=device)
    seq=torch.stack((torch.cat((a,z),1),torch.cat((b,z),1),o),2).reshape(batch,45)[:,:-1]
    return seq,o

@torch.no_grad()
def evaluate(model,count=10000,batch=2000,seed=9127):
    g=torch.Generator().manual_seed(seed); model.eval(); good=dg=total=0; device=next(model.parameters()).device
    for start in range(0,count,batch):
        n=min(batch,count-start); av=torch.randint(0,MAX,(n,),generator=g); bv=torch.randint(0,MAX,(n,),generator=g)
        a=torch.stack([(av//10**i)%10 for i in range(14)],1).to(device); b=torch.stack([(bv//10**i)%10 for i in range(14)],1).to(device)
        carry=torch.zeros(n,dtype=torch.long,device=device); target=[]
        for i in range(14):s=a[:,i]+b[:,i]+carry;target.append(s%10);carry=s//10
        target.append(carry);target=torch.stack(target,1);seq=torch.empty(n,0,dtype=torch.long,device=device);pred=[];z=torch.zeros(n,dtype=torch.long,device=device)
        for i in range(15):
            seq=torch.cat((seq,(a[:,i] if i<14 else z)[:,None],(b[:,i] if i<14 else z)[:,None]),1)
            p=model(seq)[:,-1].argmax(-1);pred.append(p);seq=torch.cat((seq,p[:,None]),1)
        pred=torch.stack(pred,1);good+=(pred==target).all(1).sum().item();dg+=(pred==target).sum().item();total+=n*15
    return good/count,dg/total

def export(model,path):
    d,h,heads=model.d,model.hidden,model.heads; state={k:v.detach().cpu().tolist() for k,v in model.state_dict().items()}
    source=f'''import math\nimport torch\nfrom torch import nn\nimport torch.nn.functional as F\nD,H,HEADS={d},{h},{heads}\n_WEIGHTS={state!r}\nclass Block(nn.Module):\n def __init__(self):\n  super().__init__();self.dk=D//HEADS;self.norm1=nn.LayerNorm(D);self.qkv=nn.Linear(D,3*D);self.proj=nn.Linear(D,D);self.norm2=nn.LayerNorm(D);self.ff1=nn.Linear(D,H);self.ff2=nn.Linear(H,D)\n def forward(self,x,mask):\n  B,N,_=x.shape;z=self.norm1(x);q,k,v=self.qkv(z).chunk(3,-1);q=q.view(B,N,HEADS,self.dk).transpose(1,2);k=k.view(B,N,HEADS,self.dk).transpose(1,2);v=v.view(B,N,HEADS,self.dk).transpose(1,2);att=torch.softmax((q@k.transpose(-2,-1)/math.sqrt(self.dk)).masked_fill(~mask,-1e4),-1);x=x+self.proj((att@v).transpose(1,2).reshape(B,N,D));return x+self.ff2(F.gelu(self.ff1(self.norm2(x))))\nclass AddTransformer(nn.Module):\n def __init__(self):\n  super().__init__();self.digit=nn.Embedding(10,D);self.role=nn.Embedding(3,D);self.blocks=nn.ModuleList([Block(),Block()]);self.out_norm=nn.LayerNorm(D);self.head=nn.Linear(D,10)\n def forward(self,tokens):\n  n=tokens.shape[1];p=torch.arange(n,device=tokens.device);x=self.digit(tokens)+self.role(p%3);mask=((p[:,None]>=p[None,:])&(p[:,None]-p[None,:]<=5))[None,None]\n  for block in self.blocks:x=block(x,mask)\n  return self.head(self.out_norm(x))\ndef build_model():\n model=AddTransformer();model.load_state_dict({{k:torch.tensor(v) for k,v in _WEIGHTS.items()}});model.eval();return model,{{"architecture":"two-block two-head local causal autoregressive transformer","digits":14,"sequence_order":"least-significant-first","training":"random complete operand pairs"}}\n@torch.inference_mode()\ndef add(model,a:int,b:int)->int:\n ad=[];bd=[]\n for _ in range(14):a,r=divmod(a,10);ad.append(r);b,r=divmod(b,10);bd.append(r)\n seq=[];result=0;place=1\n for i in range(15):\n  seq.extend((ad[i] if i<14 else 0,bd[i] if i<14 else 0));token=torch.tensor([seq],device=next(model.parameters()).device);digit=int(model(token)[0,-1].argmax());seq.append(digit);result+=digit*place;place*=10\n return result\n''';Path(path).write_text(source)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--d',type=int,default=16);ap.add_argument('--hidden',type=int,default=32);ap.add_argument('--heads',type=int,default=2);ap.add_argument('--steps',type=int,default=1400);ap.add_argument('--batch',type=int,default=4096);ap.add_argument('--lr',type=float,default=.003);ap.add_argument('--output',default='/workspace/submission.py');ap.add_argument('--resume');ap.add_argument('--seed',type=int,default=2);args=ap.parse_args()
    torch.manual_seed(args.seed);device='cuda' if torch.cuda.is_available() else 'cpu';model=AddTransformer(args.d,args.hidden,args.heads)
    if args.resume:
        import importlib.util
        spec=importlib.util.spec_from_file_location('checkpoint',args.resume);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);saved,_=module.build_model();model.load_state_dict(saved.state_dict(),strict=False)
    model=model.to(device);print('parameters',sum(p.numel() for p in model.parameters()),device,flush=True);opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.005)
    for step in range(1,args.steps+1):
        model.train();seq,target=batch_data(args.batch,device);pred=model(seq)[:,1::3];loss=F.cross_entropy(pred.reshape(-1,10),target.reshape(-1));opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if step%500==0 or step==1:
            acc,da=evaluate(model,2000);print(step,float(loss),acc,da,flush=True);export(model,args.output)
    acc,da=evaluate(model,20000);print('FINAL',acc,da,flush=True);export(model,args.output)
if __name__=='__main__':main()
