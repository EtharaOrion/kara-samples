import torch, math, numpy as np, random
import torch.nn as nn, torch.nn.functional as F

def train_one(seed, D_MODEL, D_FF, steps=30000):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    VOCAB=12; SEQ_LEN=44; INPUT_LEN=29; OUTPUT_LEN=15; MAX_VAL=99999999999999; BATCH=512
    def _sinusoidal_pe(seq_len,d_model):
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
            self.attn=nn.MultiheadAttention(D_MODEL,2,batch_first=True)
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
    POW10=np.array([10**i for i in range(15)],dtype=np.int64)
    def encode_batch_np(a_np,b_np):
        B=len(a_np)
        inp=np.zeros((B,INPUT_LEN),dtype=np.int64)
        tgt=np.zeros((B,OUTPUT_LEN),dtype=np.int64)
        for k in range(14):
            inp[:,k]=(a_np//POW10[k])%10
            inp[:,15+k]=(b_np//POW10[k])%10
        inp[:,14]=10
        s=a_np+b_np
        for k in range(15):
            tgt[:,k]=(s//POW10[k])%10
        full=np.concatenate([inp,tgt],axis=1)
        return torch.from_numpy(full).long()
    def sample_batch_np(bs):
        a=np.zeros(bs,dtype=np.int64); b=np.zeros(bs,dtype=np.int64)
        r=np.random.rand(bs)
        for i in range(bs):
            ri=r[i]
            if ri<0.35:
                a[i]=np.random.randint(0,MAX_VAL+1); b[i]=np.random.randint(0,MAX_VAL+1)
            elif ri<0.50:
                da=np.random.randint(1,15); db=np.random.randint(1,15)
                a[i]=np.random.randint(0,10**da); b[i]=np.random.randint(0,10**db)
            elif ri<0.62:
                v=0
                for k in range(14):
                    d=np.random.randint(5,10) if np.random.rand()<0.7 else np.random.randint(0,10)
                    v+=d*(10**k)
                a[i]=min(v,MAX_VAL)
                v=0
                for k in range(14):
                    d=np.random.randint(5,10) if np.random.rand()<0.7 else np.random.randint(0,10)
                    v+=d*(10**k)
                b[i]=min(v,MAX_VAL)
            elif ri<0.72:
                if np.random.rand()<0.5: a[i]=0; b[i]=np.random.randint(0,MAX_VAL+1)
                else: a[i]=np.random.randint(0,MAX_VAL+1); b[i]=0
            elif ri<0.82:
                k=np.random.randint(0,14); base=10**k
                a[i]=base; b[i]=base
                if np.random.rand()<0.5:
                    a[i]=np.random.randint(0,MAX_VAL+1)//base*base
                    b[i]=np.random.randint(0,MAX_VAL+1)//base*base
            elif ri<0.90:
                k=np.random.randint(1,7)
                av=np.random.randint(0,MAX_VAL//10**k+1)*10**k+(10**k-1)
                bv=np.random.randint(0,10)
                if np.random.rand()<0.5: a[i]=av; b[i]=bv
                else: a[i]=bv; b[i]=av
            else:
                a[i]=np.random.randint(0,1001); b[i]=np.random.randint(0,1001)
        return a,b
    model=TinyTransformer()
    opt=torch.optim.AdamW(model.parameters(),lr=4e-4,weight_decay=0.01)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=60000)
    for step in range(1,steps+1):
        a,b=sample_batch_np(BATCH)
        full=encode_batch_np(a,b)
        inp=full[:,:-1]; tgt=full[:,1:]
        logits=model(inp)
        loss=F.cross_entropy(logits.reshape(-1,VOCAB),tgt.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step(); sched.step()
        if step%10000==0:
            print(f"seed {seed} D{D_MODEL}FF{D_FF} step {step} loss {loss.item():.4f}")
    return loss.item()

for seed in [0,1,2,3,4,5,42,123,999]:
    loss = train_one(seed, 10, 20, steps=30000)
    print(f"FINAL seed {seed} loss {loss:.4f} {'GROK' if loss<1.3 else 'NO'}")

print("--- D12/20 ---")
for seed in [0,1,42]:
    loss = train_one(seed, 12, 20, steps=30000)
    print(f"FINAL seed {seed} D12/20 loss {loss:.4f} {'GROK' if loss<1.3 else 'NO'}")
