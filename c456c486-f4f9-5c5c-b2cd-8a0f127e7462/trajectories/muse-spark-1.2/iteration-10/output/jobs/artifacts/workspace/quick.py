import torch, random, math
import torch.nn as nn, torch.nn.functional as F
D_MODEL=10; NHEAD=2; D_FF=19; VOCAB=12; SEQ_LEN=44; INPUT_LEN=29; OUTPUT_LEN=15
MAX_VAL=99999999999999; BATCH=512
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

# vectorized encode
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
        if r<0.35: a.append(random.randint(0,MAX_VAL));b.append(random.randint(0,MAX_VAL))
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
        else: a.append(random.randint(0,1000));b.append(random.randint(0,1000))
    return a,b

for seed in [0,1,42,123]:
    torch.manual_seed(seed); random.seed(seed)
    model=TinyTransformer()
    opt=torch.optim.AdamW(model.parameters(),lr=4e-4,weight_decay=0.01)
    for step in range(1,3001):
        a,b=sample_batch(BATCH)
        full=encode_batch(a,b)
        inp=full[:,:-1]; tgt=full[:,1:]
        logits=model(inp)
        loss=F.cross_entropy(logits.reshape(-1,VOCAB),tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step%1000==0:
            print(f"seed {seed} step {step} loss {loss.item():.4f}")
