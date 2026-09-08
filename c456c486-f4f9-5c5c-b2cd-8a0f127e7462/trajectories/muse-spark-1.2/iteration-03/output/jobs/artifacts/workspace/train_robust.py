import torch, torch.nn as nn, torch.nn.functional as F, random, math, sys
D_MODEL=12; NHEAD=2; N_LAYERS=2; D_FF=16; VOCAB=12; SEQ_INPUT=30; SEQ_OUTPUT=15; SEQ_TOTAL=45; MAX_VAL=99_999_999_999_999
DEVICE="cpu"
def encode_number(n, length=14): return [int(c) for c in reversed(str(n).zfill(length))]
def encode_input(a,b): return encode_number(a)+[10]+encode_number(b)+[11]
def encode_output(s): return encode_number(s,15)
def decode_output(t): return int(''.join(str(x) for x in reversed(t)).lstrip('0') or '0')
class TinyTransformer(nn.Module):
    def __init__(self, d_model=12, nhead=2, n_layers=2, d_ff=16, vocab=12, seq_len=45, tie=False):
        super().__init__()
        self.d_model=d_model; self.vocab=vocab; self.seq_len=seq_len; self.tie_weights=tie
        self.token_emb=nn.Embedding(vocab,d_model); self.pos_emb=nn.Embedding(seq_len,d_model)
        self.layers=nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({'ln1':nn.LayerNorm(d_model),'attn':nn.MultiheadAttention(d_model,nhead,batch_first=True),'ln2':nn.LayerNorm(d_model),'ff1':nn.Linear(d_model,d_ff),'ff2':nn.Linear(d_ff,d_model)}))
        self.ln_f=nn.LayerNorm(d_model); self.head=nn.Linear(d_model,vocab)
        if tie: self.head.weight=self.token_emb.weight
    def forward(self,x):
        B,T=x.shape; pos=torch.arange(T,device=x.device)
        h=self.token_emb(x)+self.pos_emb(pos)
        mask=torch.triu(torch.ones(T,T,device=x.device,dtype=torch.bool),diagonal=1)
        for layer in self.layers:
            h_norm=layer['ln1'](h); attn_out,_=layer['attn'](h_norm,h_norm,h_norm,attn_mask=mask); h=h+attn_out
            h_norm2=layer['ln2'](h); ff=layer['ff2'](F.gelu(layer['ff1'](h_norm2))); h=h+ff
        return self.head(self.ln_f(h))

def sample_pair():
    r=random.random()
    if r<0.35: a=random.randint(0,MAX_VAL); b=random.randint(0,MAX_VAL)
    elif r<0.50:
        la=random.randint(1,14); lb=random.randint(1,14)
        a=random.randint(0 if la==1 else 10**(la-1),10**la-1); b=random.randint(0 if lb==1 else 10**(lb-1),10**lb-1)
        if random.random()<0.1: a=0
        if random.random()<0.1: b=0
    elif r<0.65: a=int(''.join(str(random.randint(5,9)) for _ in range(14))); b=int(''.join(str(random.randint(5,9)) for _ in range(14)))
    elif r<0.80:
        k=random.randint(3,14); high=random.randint(0,10**(14-k)-1) if k<14 else 0
        a=high*(10**k)+int('9'*k); b=random.randint(1,10**k-1)
        if b>=10**6: b=random.randint(1,999999)
    else: a=random.randint(0,99999); b=random.randint(0,99999)
    return a,b

def make_batch(bs):
    inp=torch.zeros(bs,SEQ_TOTAL,dtype=torch.long)
    for i in range(bs):
        a,b=sample_pair(); s=a+b; full=encode_input(a,b)+encode_output(s); inp[i]=torch.tensor(full)
    return inp

@torch.no_grad()
def greedy_decode(model,a,b):
    enc=encode_input(a,b); toks=enc[:]
    for _ in range(SEQ_OUTPUT):
        x=torch.tensor([toks],dtype=torch.long); logits=model(x); toks.append(int(logits[0,-1].argmax().item()))
    return decode_output(toks[SEQ_INPUT:])

@torch.no_grad()
def evaluate(model,n=1000):
    model.eval(); c=0
    for _ in range(n):
        a=random.randint(0,MAX_VAL); b=random.randint(0,MAX_VAL)
        if greedy_decode(model,a,b)==a+b: c+=1
    model.train(); return c/n

def train_one(seed, d_model, nhead, n_layers, d_ff, steps=35000, lr=0.001, tie=False):
    random.seed(seed); torch.manual_seed(seed)
    model=TinyTransformer(d_model,nhead,n_layers,d_ff,tie_weights=tie).to(DEVICE)
    params=sum(p.numel() for p in model.parameters())
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.01)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=steps)
    for step in range(1,steps+1):
        inp=make_batch(512).to(DEVICE)
        logits=model(inp[:,:-1])
        targets=inp[:,1:]
        loss=F.cross_entropy(logits[:,29:].reshape(-1,VOCAB),targets[:,29:].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sched.step()
        if step%7000==0:
            acc=evaluate(model,500)
            print(f"  seed {seed} step {step} loss {loss.item():.4f} acc {acc:.3f}")
    acc=evaluate(model,2000)
    print(f"SEED {seed} d={d_model} ff={d_ff} tie={tie} params={params} FINAL acc={acc:.4f} {'PASS' if acc>=0.99 else 'FAIL'}")
    return acc, model, params

# quick sweep
import itertools
configs=[(12,2,2,16,False),(12,2,2,20,False),(11,2,2,16,False),(10,2,2,16,False),(12,1,2,16,False),(12,2,2,16,True)]
for d,h,l,ff,tie in configs:
    for seed in [0,1,42]:
        acc,_,p=train_one(seed,d,h,l,ff,steps=30000,tie=tie)
        if acc>=0.99:
            print(f"  *** FOUND {d}/{h}/{l}/{ff} tie={tie} seed={seed} acc={acc:.4f} p={p}")
