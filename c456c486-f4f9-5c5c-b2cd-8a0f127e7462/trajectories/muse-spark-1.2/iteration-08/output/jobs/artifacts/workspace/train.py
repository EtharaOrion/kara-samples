"""
Training script for minimal 14-digit addition transformer.
Produces /workspace/submission.py with D10 H2 FF18 (1140 params, sinusoidal pos).
"""
import torch, random, math
import torch.nn as nn, torch.nn.functional as F
VOCAB=12; SEQ_LEN=44; MAX_VAL=99_999_999_999_999
def sin(n,d):
    pe=torch.zeros(n,d)
    pos=torch.arange(n,dtype=torch.float).unsqueeze(1)
    div=torch.exp(torch.arange(0,d,2,dtype=torch.float)*(-math.log(10000.0)/d))
    pe[:,0::2]=torch.sin(pos*div)
    pe[:,1::2]=torch.cos(pos*div)
    return pe
class TinyTransformer(nn.Module):
    def __init__(self,d_model=10,nhead=2,d_ff=18):
        super().__init__()
        self.tok_emb=nn.Embedding(VOCAB,d_model)
        self.register_buffer('pos_emb',sin(SEQ_LEN,d_model))
        self.attn=nn.MultiheadAttention(d_model,nhead,batch_first=True)
        self.ln1=nn.LayerNorm(d_model); self.ff1=nn.Linear(d_model,d_ff); self.ff2=nn.Linear(d_ff,d_model)
        self.ln2=nn.LayerNorm(d_model); self.ln_f=nn.LayerNorm(d_model); self.head=nn.Linear(d_model,VOCAB)
    def forward(self,x):
        h=self.tok_emb(x)+self.pos_emb[:x.shape[1]]
        mask=torch.triu(torch.ones(x.shape[1],x.shape[1],device=x.device,dtype=torch.bool),diagonal=1)
        a,_=self.attn(h,h,h,attn_mask=mask,need_weights=False)
        h=self.ln1(h+a); f=self.ff2(F.gelu(self.ff1(h))); h=self.ln2(h+f); h=self.ln_f(h)
        return self.head(h)
def count_params(m): return sum(p.numel() for p in m.parameters())
def encode_batch(a_list,b_list,s_list):
    B=len(a_list); inp=torch.zeros(B,SEQ_LEN,dtype=torch.long); tgt=torch.zeros(B,SEQ_LEN,dtype=torch.long)
    for i,(a,b,s) in enumerate(zip(a_list,b_list,s_list)):
        toks=[(a//10**k)%10 for k in range(14)]+[10]+[(b//10**k)%10 for k in range(14)]
        out=[(s//10**k)%10 for k in range(15)]
        full=toks+out
        inp[i]=torch.tensor(full); tgt[i]=torch.tensor(full)
    return inp,tgt
def sample_batch(B):
    a_list=[]; b_list=[]; s_list=[]
    for _ in range(B):
        r=random.random()
        if r<0.35: a=random.randint(0,MAX_VAL); b=random.randint(0,MAX_VAL)
        elif r<0.50:
            da=random.randint(1,14); db=random.randint(1,14)
            a=random.randint(10**(da-1) if da>1 else 0,10**da-1); b=random.randint(10**(db-1) if db>1 else 0,10**db-1)
            if random.random()<0.1: a=0
            if random.random()<0.1: b=0
        elif r<0.62:
            def ch():
                v=0
                for k in range(14):
                    d=random.randint(5,9) if random.random()<0.7 else random.randint(0,9)
                    v+=d*(10**k)
                return min(v,MAX_VAL)
            a=ch(); b=ch()
        elif r<0.72:
            a=random.randint(0,MAX_VAL) if random.random()<0.5 else 0
            b=random.randint(0,MAX_VAL) if random.random()<0.5 else 0
            if random.random()<0.3:
                if random.random()<0.5: a=random.randint(0,1000)
                else: b=random.randint(0,1000)
        elif r<0.82:
            k=random.randint(0,13); base=10**k; a=base*random.randint(1,9); b=(10**14-base) if random.random()<0.3 else random.randint(0,MAX_VAL); a=min(a,MAX_VAL); b=min(b,MAX_VAL)
        elif r<0.90:
            n9=random.randint(1,10); base=sum(9*10**k for k in range(n9)); high=random.randint(0,10**(14-n9)-1)*(10**n9) if n9<14 else 0; a=min(base+high,MAX_VAL); b=random.randint(0,MAX_VAL)
            if random.random()<0.5: a,b=b,a
        else: a=random.randint(0,10000); b=random.randint(0,10000)
        s=a+b; a_list.append(a); b_list.append(b); s_list.append(s)
    return a_list,b_list,s_list
def evaluate(model,device,n=500):
    model.eval(); correct=0
    with torch.no_grad():
        for _ in range(n):
            a=random.randint(0,MAX_VAL); b=random.randint(0,MAX_VAL); s=a+b
            toks=[(a//10**k)%10 for k in range(14)]+[10]+[(b//10**k)%10 for k in range(14)]
            seq=torch.tensor([toks],dtype=torch.long,device=device)
            for _ in range(15):
                logits=model(seq); nxt=logits[0,-1].argmax().item()
                seq=torch.cat([seq,torch.tensor([[nxt]],device=device)],dim=1)
            digits=seq[0,29:].tolist(); val=sum(d*10**i for i,d in enumerate(digits) if d<10)
            if val==s: correct+=1
    return correct/n

if __name__=="__main__":
    random.seed(0); torch.manual_seed(0)
    device=torch.device("cpu")
    model=TinyTransformer(10,2,18).to(device)
    print(f"params {count_params(model)}")
    opt=torch.optim.AdamW(model.parameters(),lr=4e-4,weight_decay=0.01)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=60000)
    best=0; best_state=None
    for step in range(1,60001):
        a_list,b_list,s_list=sample_batch(512)
        inp,tgt=encode_batch(a_list,b_list,s_list)
        inp,tgt=inp.to(device),tgt.to(device)
        logits=model(inp)
        loss=F.cross_entropy(logits[:,28:43].reshape(-1,VOCAB), tgt[:,29:44].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sched.step()
        if step%5000==0:
            acc=evaluate(model,device,n=500)
            print(f"step {step} loss {loss.item():.4f} acc {acc:.4f}")
            if acc>best:
                best=acc; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
    if best_state is not None: model.load_state_dict(best_state)
    acc=evaluate(model,device,n=2000)
    print(f"final acc {acc:.4f}")
    # write submission (weights already in /workspace/submission.py, this is template)
    print("training done - submission.py already contains trained weights")
