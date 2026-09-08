import torch, torch.nn as nn, torch.nn.functional as F, random
VOCAB=12; SEQ_TOTAL=45; SEQ_INPUT=30; SEQ_OUTPUT=15; MAX_VAL=99_999_999_999_999
def enc_num(n,l=14): return [int(c) for c in reversed(str(n).zfill(l))]
def enc_in(a,b): return enc_num(a)+[10]+enc_num(b)+[11]
def enc_out(s): return enc_num(s,15)
def dec_out(t): return int(''.join(str(x) for x in reversed(t)).lstrip('0') or '0')
class M(nn.Module):
    def __init__(self,d,h,l,ff,tie=False):
        super().__init__()
        self.d_model=d; self.vocab=12; self.tie=tie
        self.token_emb=nn.Embedding(12,d); self.pos_emb=nn.Embedding(45,d)
        self.layers=nn.ModuleList()
        for _ in range(l):
            self.layers.append(nn.ModuleDict({'ln1':nn.LayerNorm(d),'attn':nn.MultiheadAttention(d,h,batch_first=True),'ln2':nn.LayerNorm(d),'ff1':nn.Linear(d,ff),'ff2':nn.Linear(ff,d)}))
        self.ln_f=nn.LayerNorm(d); self.head=nn.Linear(d,12)
        if tie: self.head.weight=self.token_emb.weight
    def forward(self,x):
        h=self.token_emb(x)+self.pos_emb(torch.arange(x.shape[1],device=x.device))
        mask=torch.triu(torch.ones(x.shape[1],x.shape[1],device=x.device,dtype=torch.bool),diagonal=1)
        for ly in self.layers:
            hn=ly['ln1'](h); a,_=ly['attn'](hn,hn,hn,attn_mask=mask); h=h+a
            hn2=ly['ln2'](h); h=h+ly['ff2'](F.gelu(ly['ff1'](hn2)))
        return self.head(self.ln_f(h))
def sample():
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
    inp=torch.zeros(bs,45,dtype=torch.long)
    for i in range(bs):
        a,b=sample(); s=a+b; inp[i]=torch.tensor(enc_in(a,b)+enc_out(s))
    return inp
@torch.no_grad()
def decode(m,a,b):
    toks=enc_in(a,b)[:]
    for _ in range(15):
        x=torch.tensor([toks],dtype=torch.long); toks.append(int(m(x)[0,-1].argmax().item()))
    return dec_out(toks[30:])
@torch.no_grad()
def eval_acc(m,n=2000):
    m.eval(); c=0
    for _ in range(n):
        a=random.randint(0,MAX_VAL); b=random.randint(0,MAX_VAL)
        if decode(m,a,b)==a+b: c+=1
    m.train(); return c/n
def train_one(d,h,l,ff,tie,steps,seed):
    random.seed(seed); torch.manual_seed(seed)
    m=M(d,h,l,ff,tie)
    params=sum(p.numel() for p in m.parameters())
    opt=torch.optim.AdamW(m.parameters(),lr=0.001,weight_decay=0.01)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=steps)
    for step in range(1,steps+1):
        inp=make_batch(512)
        logits=m(inp[:,:-1])
        loss=F.cross_entropy(logits[:,29:].reshape(-1,12), inp[:,1:][:,29:].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); sched.step()
    acc=eval_acc(m,2000)
    print(f"RESULT d={d} h={h} L={l} ff={ff} tie={tie} seed={seed} steps={steps} params={params} acc={acc:.4f} {'PASS' if acc>=0.99 else 'FAIL'}", flush=True)
    return acc, m, params

configs = [
    (10,2,2,14,False,40000),
    (10,2,2,12,False,40000),
    (9,1,2,16,False,40000),
    (9,1,2,14,False,40000),
    (10,2,2,16,True,40000),
    (9,1,2,16,True,40000),
    (8,2,2,16,True,40000),
]
for d,h,l,ff,tie,steps in configs:
    for seed in [0,1]:
        train_one(d,h,l,ff,tie,steps,seed)
