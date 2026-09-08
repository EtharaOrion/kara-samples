import torch, torch.nn as nn, torch.nn.functional as F, random, os, io, gzip, base64, time
DEVICE="cuda"
D_MODEL=18; N_LAYERS=2; N_HEADS=2; D_FF=32
VOCAB=12; PLUS_ID=10; SEQ_IN=29; SEQ_OUT=15
def encode_pair(a,b):
    a_digits=[(a//10**i)%10 for i in range(14)]
    b_digits=[(b//10**i)%10 for i in range(14)]
    s=a+b
    s_digits=[(s//10**i)%10 for i in range(15)]
    return a_digits+[PLUS_ID]+b_digits, s_digits
def sample_int():
    if random.random()<0.5: return random.randint(0,99_999_999_999_999)
    else:
        d=random.randint(1,14)
        if d==14: return random.randint(10**13,99_999_999_999_999)
        elif d==1: return random.randint(0,9)
        else: return random.randint(10**(d-1),10**d-1)
def make_batch(bs):
    full=[]
    for _ in range(bs):
        r=random.random()
        if r<0.05:
            nines=random.randint(1,14)
            a=int("9"*nines) if nines<14 else 99_999_999_999_999
            if random.random()<0.5:
                a=random.randint(0,99_999_999_999_999)
                b=(10**random.randint(1,14)-a%10**random.randint(1,14))%10**14
                b=b%100_000_000_000_000
            else:
                b=random.randint(1,10**nines-1) if nines<14 else random.randint(0,99_999_999_999_999)
        elif r<0.10:
            a=0; b=sample_int()
            if random.random()<0.5: a,b=b,a
        else:
            a=sample_int(); b=sample_int()
        inp,tgt=encode_pair(a,b)
        full.append(inp+tgt)
    return torch.tensor(full,dtype=torch.long)
def make_eval_batch(bs):
    full=[]
    for _ in range(bs):
        a=random.randint(0,99_999_999_999_999); b=random.randint(0,99_999_999_999_999)
        inp,tgt=encode_pair(a,b)
        full.append(inp+tgt)
    return torch.tensor(full,dtype=torch.long)
class TB(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1=nn.LayerNorm(D_MODEL)
        self.attn=nn.MultiheadAttention(D_MODEL,N_HEADS,batch_first=True)
        self.ln2=nn.LayerNorm(D_MODEL)
        self.ff=nn.Sequential(nn.Linear(D_MODEL,D_FF),nn.GELU(),nn.Linear(D_FF,D_MODEL))
    def forward(self,x,mask):
        h=self.ln1(x)
        a,_=self.attn(h,h,h,attn_mask=mask,need_weights=False)
        x=x+a
        x=x+self.ff(self.ln2(x))
        return x
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb=nn.Embedding(VOCAB,D_MODEL)
        self.pos_emb=nn.Embedding(44,D_MODEL)
        self.layers=nn.ModuleList([TB() for _ in range(N_LAYERS)])
        self.ln_f=nn.LayerNorm(D_MODEL)
        self.head=nn.Linear(D_MODEL,VOCAB,bias=False)
    def forward(self,idx):
        B,T=idx.shape
        pos=torch.arange(T,device=idx.device)
        x=self.tok_emb(idx)+self.pos_emb(pos)[None,:,:]
        mask=torch.triu(torch.ones(T,T,device=idx.device,dtype=torch.bool),diagonal=1)
        fm=torch.zeros(T,T,device=idx.device); fm.masked_fill_(mask,float('-inf'))
        for l in self.layers: x=l(x,fm)
        return self.head(self.ln_f(x))
model=M().to(DEVICE)
print("Params", sum(p.numel() for p in model.parameters()))
opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=0.01)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=50000)
best=0
t0=time.time()
for step in range(1,50001):
    full=make_batch(512).to(DEVICE)
    logits=model(full)
    loss=F.cross_entropy(logits[:,SEQ_IN-1:-1].reshape(-1,VOCAB), full[:,SEQ_IN:].reshape(-1))
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sched.step()
    if step%500==0:
        print(f"step {step} loss {loss.item():.4f} lr {opt.param_groups[0]['lr']:.2e} time {(time.time()-t0)/60:.1f}m")
    if step%1000==0:
        model.eval()
        correct=0; total=2000
        with torch.no_grad():
            for _ in range(0,total,256):
                bs=min(256,total-_)
                full2=make_eval_batch(bs).to(DEVICE)
                inp=full2[:,:SEQ_IN]
                cur=inp
                for _p in range(SEQ_OUT):
                    logits2=model(cur)
                    nxt=logits2[:,-1].argmax(-1,keepdim=True)
                    cur=torch.cat([cur,nxt],dim=1)
                pred=cur[:,SEQ_IN:]
                true=full2[:,SEQ_IN:]
                correct+=(pred==true).all(1).sum().item()
        acc=correct/total
        print(f" EVAL {step}: {acc*100:.2f}%")
        if acc>best:
            best=acc
            torch.save(model.state_dict(),"/workspace/best_6544.pt")
            print(f"  new best {acc*100:.2f}%")
        if acc>=0.995:
            print("early stop")
            break
        model.train()
print(f"best {best*100:.2f}%")
# final eval
model.load_state_dict(torch.load("/workspace/best_6544.pt",map_location=DEVICE))
model.eval()
correct=0; total=5000
with torch.no_grad():
    for _ in range(0,total,256):
        bs=min(256,total-_)
        full2=make_eval_batch(bs).to(DEVICE)
        inp=full2[:,:SEQ_IN]
        cur=inp
        for _p in range(SEQ_OUT):
            logits2=model(cur)
            nxt=logits2[:,-1].argmax(-1,keepdim=True)
            cur=torch.cat([cur,nxt],dim=1)
        pred=cur[:,SEQ_IN:]
        true=full2[:,SEQ_IN:]
        correct+=(pred==true).all(1).sum().item()
print(f"final 5000: {correct/total*100:.2f}%")
# write submission
sd=torch.load("/workspace/best_6544.pt", map_location="cpu")
buf=io.BytesIO(); torch.save(sd,buf); raw=buf.getvalue(); comp=gzip.compress(raw); b64=base64.b64encode(comp).decode()
code=f'''import torch
import torch.nn as nn
import torch.nn.functional as F
import base64, io, gzip
VOCAB=12
PLUS_ID=10
SEQ_IN=29
SEQ_OUT=15
D_MODEL={D_MODEL}
N_LAYERS={N_LAYERS}
N_HEADS={N_HEADS}
D_FF={D_FF}
_B64="""{b64}"""
class TransformerBlock(nn.Module):
    def __init__(self,d_model,n_heads,d_ff):
        super().__init__()
        self.ln1=nn.LayerNorm(d_model)
        self.attn=nn.MultiheadAttention(d_model,n_heads,dropout=0.0,batch_first=True)
        self.ln2=nn.LayerNorm(d_model)
        self.ff=nn.Sequential(nn.Linear(d_model,d_ff),nn.GELU(),nn.Linear(d_ff,d_model))
    def forward(self,x,mask):
        h=self.ln1(x)
        a,_=self.attn(h,h,h,attn_mask=mask,need_weights=False)
        x=x+a
        h2=self.ln2(x)
        x=x+self.ff(h2)
        return x
class AddModel(nn.Module):
    def __init__(self,vocab=VOCAB,d_model=D_MODEL,n_layers=N_LAYERS,n_heads=N_HEADS,d_ff=D_FF,max_len=44):
        super().__init__()
        self.tok_emb=nn.Embedding(vocab,d_model)
        self.pos_emb=nn.Embedding(max_len,d_model)
        self.layers=nn.ModuleList([TransformerBlock(d_model,n_heads,d_ff) for _ in range(n_layers)])
        self.ln_f=nn.LayerNorm(d_model)
        self.head=nn.Linear(d_model,vocab,bias=False)
        self.max_len=max_len
    def forward(self,idx):
        B,T=idx.shape
        pos=torch.arange(T,device=idx.device)
        x=self.tok_emb(idx)+self.pos_emb(pos)[None,:,:]
        mask=torch.triu(torch.ones(T,T,device=idx.device,dtype=torch.bool),diagonal=1)
        float_mask=torch.zeros(T,T,device=idx.device)
        float_mask.masked_fill_(mask,float('-inf'))
        for layer in self.layers:
            x=layer(x,float_mask)
        x=self.ln_f(x)
        logits=self.head(x)
        return logits
def _load_model():
    m=AddModel()
    raw=gzip.decompress(base64.b64decode(_B64.encode()))
    buf=io.BytesIO(raw)
    sd=torch.load(buf,map_location="cpu")
    m.load_state_dict(sd)
    m.eval()
    return m
def build_model():
    model=_load_model()
    metadata={{"vocab":VOCAB,"d_model":D_MODEL,"n_layers":N_LAYERS,"n_heads":N_HEADS,"d_ff":D_FF}}
    return model,metadata
def add(model,a:int,b:int)->int:
    model.eval()
    device=next(model.parameters()).device
    a_digits=[(a//10**i)%10 for i in range(14)]
    b_digits=[(b//10**i)%10 for i in range(14)]
    inp=a_digits+[PLUS_ID]+b_digits
    cur=torch.tensor([inp],dtype=torch.long,device=device)
    with torch.no_grad():
        for _ in range(SEQ_OUT):
            logits=model(cur)
            nxt=logits[0,-1].argmax().item()
            cur=torch.cat([cur,torch.tensor([[nxt]],device=device)],dim=1)
    out_digits=cur[0,SEQ_IN:].tolist()
    val=0
    for i,d in enumerate(out_digits):
        if d<10:
            val+=d*(10**i)
    return val
'''
open("/workspace/submission.py","w").write(code)
print("wrote submission")
import importlib.util
spec=importlib.util.spec_from_file_location("submission","/workspace/submission.py")
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
m2,_=mod.build_model()
m2=m2.to(DEVICE)
for aa,bb in [(0,0),(1,1),(999,1),(12345678901234,98765432109876),(99999999999999,1)]:
    got=mod.add(m2,aa,bb)
    print(aa,bb,got,aa+bb, "OK" if got==aa+bb else "FAIL")
print("params", sum(p.numel() for p in m2.parameters()))
