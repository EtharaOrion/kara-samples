import torch, random
import torch.nn as nn
from submission import TinyTransformer, VOCAB_SIZE, D_MODEL, NHEAD, D_FF, MAX_LEN, INPUT_LEN, OUTPUT_LEN, PLUS_TOKEN
DEVICE=torch.device("cpu")
MAX_VAL=99_999_999_999_999
def encode_pair(a,b):
    toks=[]
    for _ in range(14):
        toks.append(a%10); a//=10
    toks.append(PLUS_TOKEN)
    for _ in range(14):
        toks.append(b%10); b//=10
    return toks
def encode_sum(s):
    toks=[]
    for _ in range(15):
        toks.append(s%10); s//=10
    return toks
def sample_batch(bs):
    import random
    inp=torch.zeros(bs, INPUT_LEN+OUTPUT_LEN, dtype=torch.long)
    tgt=torch.zeros(bs, INPUT_LEN+OUTPUT_LEN, dtype=torch.long)
    for i in range(bs):
        r=random.random()
        if r<0.35:
            a=random.randint(0,MAX_VAL); b=random.randint(0,MAX_VAL)
        elif r<0.50:
            da=random.randint(1,14); db=random.randint(1,14)
            a=random.randint(10**(da-1) if da>1 else 0, 10**da-1)
            b=random.randint(10**(db-1) if db>1 else 0, 10**db-1)
        elif r<0.62:
            a=0; b=0
            for pos in range(14):
                a+=random.randint(5,9)*(10**pos)
                b+=random.randint(5,9)*(10**pos)
            if random.random()<0.5: a=random.randint(0,MAX_VAL)
            if random.random()<0.5: b=random.randint(0,MAX_VAL)
        elif r<0.72:
            if random.random()<0.5: a=0; b=random.randint(0,MAX_VAL)
            else: a=random.randint(0,MAX_VAL); b=0
            if random.random()<0.3: a=0; b=0
        elif r<0.82:
            k=random.randint(0,13); base=10**k
            choices=[base,5*base,10**14-1,10**k-1]
            a=random.choice(choices)%(MAX_VAL+1); b=random.choice(choices)%(MAX_VAL+1)
            if random.random()<0.3:
                a=int("9"*random.randint(1,14)) if random.random()<0.5 else a
                b=int("9"*random.randint(1,14)) if random.random()<0.5 else b
        elif r<0.90:
            n9a=random.randint(1,7); n9b=random.randint(1,7)
            a=random.randint(0,MAX_VAL//(10**n9a))*(10**n9a)+int("9"*n9a)
            b=random.randint(0,MAX_VAL//(10**n9b))*(10**n9b)+int("9"*n9b)
            a=min(a,MAX_VAL); b=min(b,MAX_VAL)
        else:
            a=random.randint(0,10000); b=random.randint(0,10000)
        s=a+b
        full=encode_pair(a,b)+encode_sum(s)
        inp[i]=torch.tensor(full); tgt[i]=torch.tensor(full)
    return inp,tgt

# test D10 FF18
import importlib, sys
# monkey patch submission to D10 FF18
import submission
submission.D_MODEL=10; submission.D_FF=18; submission.NHEAD=2
# need to reload class? create directly
from submission import _sinusoidal_pos
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        d_model=10; nhead=2; d_ff=18; vocab=12; max_len=44
        self.tok_emb=nn.Embedding(vocab,d_model)
        pe=_sinusoidal_pos(max_len,d_model)
        self.register_buffer('pos_emb',pe)
        self.attn=nn.MultiheadAttention(d_model,nhead,batch_first=True)
        self.ln1=nn.LayerNorm(d_model)
        self.ffn=nn.Sequential(nn.Linear(d_model,d_ff),nn.GELU(),nn.Linear(d_ff,d_model))
        self.ln2=nn.LayerNorm(d_model)
        self.ln_f=nn.LayerNorm(d_model)
        self.head=nn.Linear(d_model,vocab)
    def forward(self,x):
        B,T=x.shape
        h=self.tok_emb(x)+self.pos_emb[:T].unsqueeze(0)
        h_res=h; h_norm=self.ln1(h)
        causal=torch.triu(torch.ones(T,T,device=x.device,dtype=torch.bool),diagonal=1)
        attn_out,_=self.attn(h_norm,h_norm,h_norm,attn_mask=causal)
        h=h_res+attn_out
        h_res2=h; h_norm2=self.ln2(h)
        h=h_res2+self.ffn(h_norm2)
        h=self.ln_f(h)
        return self.head(h)

for lr in [4e-4,6e-4,8e-4]:
    for seed in [0,1,2]:
        torch.manual_seed(seed); random.seed(seed)
        model=TestModel()
        opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.01)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=20000)
        ce=nn.CrossEntropyLoss()
        for step in range(1,20001):
            x,y=sample_batch(512)
            logits=model(x[:,:-1])
            targets=y[:,1:]
            mask=torch.zeros_like(targets,dtype=torch.bool); mask[:,28:]=True
            loss=ce(logits[mask],targets[mask])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sched.step()
            if step%5000==0:
                with torch.no_grad():
                    x_e,y_e=sample_batch(512)
                    logits_e=model(x_e[:,:-1])
                    preds=logits_e.argmax(-1); targets_e=y_e[:,1:]
                    mask_e=torch.zeros_like(targets_e,dtype=torch.bool); mask_e[:,28:]=True
                    acc=(preds[mask_e]==targets_e[mask_e]).float().mean().item()
                    seq_acc=(preds[mask_e].view(-1,15)==targets_e[mask_e].view(-1,15)).all(dim=1).float().mean().item()
                    print(f"lr {lr} seed {seed} step {step} loss {loss.item():.3f} tok {acc:.3f} seq {seq_acc:.3f}")
        print(f"--- done lr {lr} seed {seed} ---")
