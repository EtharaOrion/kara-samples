import torch, random, math
import torch.nn as nn
import torch.nn.functional as F

D_MODEL=10; NHEAD=2; D_FF=20; VOCAB=12; SEQ_LEN=44; INPUT_LEN=29; OUTPUT_LEN=15
MAX_VAL=99999999999999; BATCH=512; STEPS=60000; LR=4e-4; WD=0.01

torch.manual_seed(0); random.seed(0)

def _sinusoidal_pe(seq_len, d_model):
    pe=torch.zeros(seq_len,d_model)
    pos=torch.arange(seq_len,dtype=torch.float).unsqueeze(1)
    div=torch.exp(torch.arange(0,d_model,2,dtype=torch.float)*(-math.log(10000.0)/d_model))
    pe[:,0::2]=torch.sin(pos*div)
    pe[:,1::2]=torch.cos(pos*div[:d_model//2])
    return pe

class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb=nn.Embedding(VOCAB,D_MODEL)
        pe=_sinusoidal_pe(SEQ_LEN,D_MODEL)
        self.register_buffer('pos_emb',pe)
        self.attn=nn.MultiheadAttention(D_MODEL,NHEAD,batch_first=True)
        self.ln1=nn.LayerNorm(D_MODEL)
        self.ln2=nn.LayerNorm(D_MODEL)
        self.ln_f=nn.LayerNorm(D_MODEL)
        self.ff1=nn.Linear(D_MODEL,D_FF)
        self.ff2=nn.Linear(D_FF,D_MODEL)
        self.head=nn.Linear(D_MODEL,VOCAB)
        mask=torch.triu(torch.full((SEQ_LEN,SEQ_LEN),float('-inf')),diagonal=1)
        self.register_buffer('causal_mask',mask)
    def forward(self,x):
        B,S=x.shape
        h=self.tok_emb(x)+self.pos_emb[:S].unsqueeze(0)
        h_norm=self.ln1(h)
        attn_out,_=self.attn(h_norm,h_norm,h_norm,attn_mask=self.causal_mask[:S,:S])
        h=h+attn_out
        h_norm2=self.ln2(h)
        ff=self.ff2(F.gelu(self.ff1(h_norm2)))
        h=h+ff
        h=self.ln_f(h)
        return self.head(h)

def encode_batch(a_list,b_list):
    B=len(a_list)
    inp=torch.zeros(B,INPUT_LEN,dtype=torch.long)
    tgt=torch.zeros(B,OUTPUT_LEN,dtype=torch.long)
    for i,(a,b) in enumerate(zip(a_list,b_list)):
        for k in range(14):
            inp[i,k]=(a//(10**k))%10
        inp[i,14]=10
        for k in range(14):
            inp[i,15+k]=(b//(10**k))%10
        s=a+b
        for k in range(15):
            tgt[i,k]=(s//(10**k))%10
    return torch.cat([inp,tgt],dim=1)

def sample_batch(bs):
    a=[];b=[]
    for _ in range(bs):
        r=random.random()
        if r<0.35:
            a.append(random.randint(0,MAX_VAL));b.append(random.randint(0,MAX_VAL))
        elif r<0.50:
            da=random.randint(1,14);db=random.randint(1,14)
            a.append(random.randint(0,10**da-1));b.append(random.randint(0,10**db-1))
        elif r<0.62:
            def cn():
                v=0
                for k in range(14):
                    d=random.randint(5,9) if random.random()<0.7 else random.randint(0,9)
                    v+=d*(10**k)
                return min(v,MAX_VAL)
            a.append(cn());b.append(cn())
        elif r<0.72:
            if random.random()<0.5: a.append(0);b.append(random.randint(0,MAX_VAL))
            else: a.append(random.randint(0,MAX_VAL));b.append(0)
        elif r<0.82:
            k=random.randint(0,13);base=10**k
            a.append(base);b.append(base)
            if random.random()<0.5:
                a[-1]=random.randint(0,MAX_VAL)//base*base
                b[-1]=random.randint(0,MAX_VAL)//base*base
        elif r<0.90:
            k=random.randint(1,6)
            av=random.randint(0,MAX_VAL//10**k)*10**k+(10**k-1)
            bv=random.randint(0,9)
            if random.random()<0.5: a.append(av);b.append(bv)
            else: a.append(bv);b.append(av)
        else:
            a.append(random.randint(0,1000));b.append(random.randint(0,1000))
    return a,b

def evaluate(model,n=2000):
    model.eval()
    correct=0
    with torch.no_grad():
        for _ in range(n//BATCH):
            a=[random.randint(0,MAX_VAL) for _ in range(BATCH)]
            b=[random.randint(0,MAX_VAL) for _ in range(BATCH)]
            full=encode_batch(a,b)
            inp=full[:,:INPUT_LEN]; tgt=full[:,INPUT_LEN:]
            seq=inp
            gen=torch.zeros(BATCH,OUTPUT_LEN,dtype=torch.long)
            for step in range(OUTPUT_LEN):
                logits=model(seq)
                nxt=logits[:,-1,:].argmax(dim=-1)
                gen[:,step]=nxt
                seq=torch.cat([seq,nxt.unsqueeze(1)],dim=1)
            correct+=(gen==tgt).all(dim=1).sum().item()
    model.train()
    return correct/(n//BATCH*BATCH)

model=TinyTransformer()
opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=STEPS)
best=0; best_sd=None
for step in range(1,STEPS+1):
    a,b=sample_batch(BATCH)
    full=encode_batch(a,b)
    inp=full[:,:-1]; tgt=full[:,1:]
    logits=model(inp)
    loss=F.cross_entropy(logits.reshape(-1,VOCAB),tgt.reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    opt.step(); sched.step()
    if step%500==0:
        print(f"step {step} loss {loss.item():.4f}",flush=True)
    if step%2000==0:
        acc=evaluate(model,2000)
        print(f"  eval {acc:.4f} best {best:.4f}",flush=True)
        if acc>best:
            best=acc; best_sd={k:v.cpu().clone() for k,v in model.state_dict().items()}
            torch.save(best_sd,"/workspace/best2.pt")
            print(f"  saved best {best:.4f}",flush=True)
    if step%10000==0:
        model.eval()
        with torch.no_grad():
            for av,bv in [(0,0),(0,MAX_VAL),(MAX_VAL,MAX_VAL),(50000000000000,50000000000000)]:
                full2=encode_batch([av],[bv])
                seq=full2[:,:INPUT_LEN]
                gen=[]
                for s in range(OUTPUT_LEN):
                    lg=model(seq); nxt=lg[0,-1].argmax().item(); gen.append(nxt)
                    seq=torch.cat([seq,torch.tensor([[nxt]])],dim=1)
                val=sum(d*10**i for i,d in enumerate(gen))
                print(f"    {av}+{bv}={val} exp {av+bv} {'OK' if val==av+bv else 'FAIL'}",flush=True)
        model.train()

print(f"done best {best}")
if best_sd is not None:
    model.load_state_dict(best_sd)
acc=evaluate(model,5000)
print(f"final {acc:.4f}")

# write submission
sd=model.state_dict()
code=f'''import torch
import torch.nn as nn
import torch.nn.functional as F
VOCAB=12;SEQ_LEN=44;INPUT_LEN=29;OUTPUT_LEN=15;D_MODEL={D_MODEL};NHEAD={NHEAD};D_FF={D_FF}
def _sinusoidal_pe(seq_len,d_model):
    pe=torch.zeros(seq_len,d_model)
    pos=torch.arange(seq_len,dtype=torch.float).unsqueeze(1)
    div=torch.exp(torch.arange(0,d_model,2,dtype=torch.float)*(-torch.log(torch.tensor(10000.0))/d_model))
    pe[:,0::2]=torch.sin(pos*div)
    pe[:,1::2]=torch.cos(pos*div[:d_model//2])
    return pe
class TinyTransformer(nn.Module):
    def __init__(self,d_model=D_MODEL,nhead=NHEAD,d_ff=D_FF,vocab=VOCAB,seq_len=SEQ_LEN):
        super().__init__()
        self.tok_emb=nn.Embedding(vocab,d_model)
        pe=_sinusoidal_pe(seq_len,d_model)
        self.register_buffer('pos_emb',pe)
        self.attn=nn.MultiheadAttention(d_model,nhead,batch_first=True)
        self.ln1=nn.LayerNorm(d_model)
        self.ln2=nn.LayerNorm(d_model)
        self.ln_f=nn.LayerNorm(d_model)
        self.ff1=nn.Linear(d_model,d_ff)
        self.ff2=nn.Linear(d_ff,d_model)
        self.head=nn.Linear(d_model,vocab)
        mask=torch.triu(torch.full((seq_len,seq_len),float('-inf')),diagonal=1)
        self.register_buffer('causal_mask',mask)
    def forward(self,x):
        B,S=x.shape
        h=self.tok_emb(x)+self.pos_emb[:S].unsqueeze(0)
        h_norm=self.ln1(h)
        attn_out,_=self.attn(h_norm,h_norm,h_norm,attn_mask=self.causal_mask[:S,:S])
        h=h+attn_out
        h_norm2=self.ln2(h)
        ff=self.ff2(F.gelu(self.ff1(h_norm2)))
        h=h+ff
        h=self.ln_f(h)
        return self.head(h)
_WEIGHTS={{}}
'''
for k,v in sd.items():
    if k in ('pos_emb','causal_mask'): continue
    code+=f"_WEIGHTS['{k}']={repr(v.tolist())}\n"
code+='''
def build_model():
    model=TinyTransformer()
    sd=model.state_dict()
    for k,lst in _WEIGHTS.items():
        sd[k]=torch.tensor(lst,dtype=sd[k].dtype)
    model.load_state_dict(sd)
    return model,{"vocab":VOCAB,"seq_len":SEQ_LEN,"d_model":D_MODEL,"nhead":NHEAD,"d_ff":D_FF}
def _encode(a,b):
    t=[]
    for i in range(14): t.append((a//(10**i))%10)
    t.append(10)
    for i in range(14): t.append((b//(10**i))%10)
    return t
def _decode(tokens):
    v=0
    for i,d in enumerate(tokens): v+=int(d)*(10**i)
    return v
def add(model,a,b):
    model.eval()
    with torch.no_grad():
        inp=_encode(a,b)
        seq=torch.tensor([inp],dtype=torch.long)
        gen=[]
        for step in range(OUTPUT_LEN):
            logits=model(seq)
            nxt_logit=logits[0,-1,:]
            nxt_logit[10]=float('-inf'); nxt_logit[11]=float('-inf')
            nxt=int(torch.argmax(nxt_logit).item())
            gen.append(nxt)
            seq=torch.cat([seq,torch.tensor([[nxt]],dtype=torch.long)],dim=1)
        return _decode(gen)
'''
open('/workspace/submission.py','w').write(code)
print("wrote submission.py")
import importlib.util
spec=importlib.util.spec_from_file_location("sub","/workspace/submission.py")
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
m,_=mod.build_model()
for av,bv in [(0,0),(0,MAX_VAL),(MAX_VAL,MAX_VAL),(12345678901234,98765432109876)]:
    print(av,bv,mod.add(m,av,bv),av+bv,mod.add(m,av,bv)==av+bv)
print(f"params {sum(p.numel() for p in m.parameters())}")
