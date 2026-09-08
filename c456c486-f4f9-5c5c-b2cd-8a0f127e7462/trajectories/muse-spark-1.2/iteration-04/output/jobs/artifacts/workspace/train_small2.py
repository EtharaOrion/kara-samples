import torch, random
import torch.nn as nn
import torch.nn.functional as F

DEVICE="cpu"

def run_config(D_MODEL, NHEAD, D_FF, STEPS=50000, label=""):
    VOCAB=12; SEQ_IN=29; SEQ_OUT=15; SEQ_TOTAL=44; MAX_VAL=99_999_999_999_999; BATCH=512; LR=4e-4; WD=0.01
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb=nn.Embedding(VOCAB,D_MODEL)
            self.pos_emb=nn.Embedding(SEQ_TOTAL,D_MODEL)
            self.layers=nn.ModuleList()
            for _ in range(1):
                self.layers.append(nn.ModuleDict({
                    'attn': nn.MultiheadAttention(D_MODEL,NHEAD,batch_first=True),
                    'ln1': nn.LayerNorm(D_MODEL),
                    'ln2': nn.LayerNorm(D_MODEL),
                    'ff1': nn.Linear(D_MODEL,D_FF),
                    'ff2': nn.Linear(D_FF,D_MODEL),
                }))
            self.ln_f=nn.LayerNorm(D_MODEL)
            self.head=nn.Linear(D_MODEL,VOCAB,bias=False)
        def forward(self,x):
            B,T=x.shape
            h=self.tok_emb(x)+self.pos_emb(torch.arange(T,device=x.device))
            causal=torch.triu(torch.ones(T,T,device=x.device,dtype=torch.bool),diagonal=1)
            for layer in self.layers:
                h_norm=layer['ln1'](h)
                attn_out,_=layer['attn'](h_norm,h_norm,h_norm,attn_mask=causal)
                h=h+attn_out
                h_norm2=layer['ln2'](h)
                ff=layer['ff2'](F.gelu(layer['ff1'](h_norm2)))
                h=h+ff
            return self.head(self.ln_f(h))
    def cnt(m): return sum(p.numel() for p in m.parameters())
    def enc_pair(a,b):
        t=[]
        for i in range(14): t.append((a//(10**i))%10)
        t.append(10)
        for i in range(14): t.append((b//(10**i))%10)
        return t
    def enc_sum(s): return [(s//(10**i))%10 for i in range(15)]
    def dec_sum(t): return sum(v*(10**i) for i,v in enumerate(t))
    def sample(bs):
        al=[]; bl=[]
        for _ in range(bs):
            r=random.random()
            if r<0.35: a=random.randint(0,MAX_VAL); b=random.randint(0,MAX_VAL)
            elif r<0.50:
                da=random.randint(1,14); db=random.randint(1,14)
                a=random.randint(0,10**da-1); b=random.randint(0,10**db-1)
            elif r<0.62:
                a=sum(random.randint(5,9)*(10**i) for i in range(14))
                b=sum(random.randint(5,9)*(10**i) for i in range(14))
                a=min(a,MAX_VAL); b=min(b,MAX_VAL)
            elif r<0.72:
                if random.random()<0.5: a=0; b=random.randint(0,MAX_VAL)
                else: a=random.randint(0,MAX_VAL); b=0
                if random.random()<0.2: a=0; b=0
            elif r<0.82:
                k=random.randint(0,13)
                a=5*(10**k) if random.random()<0.5 else (10**k)
                b=5*(10**k) if random.random()<0.5 else (10**k)
                if random.random()<0.3: a=random.randint(0,MAX_VAL); b=random.randint(0,MAX_VAL)
            elif r<0.90:
                n=random.randint(1,14)
                a=int("9"*n) if random.random()<0.5 else random.randint(0,MAX_VAL)
                b=int("9"*n) if random.random()<0.5 else random.randint(0,MAX_VAL)
            else: a=random.randint(0,9999); b=random.randint(0,9999)
            al.append(a); bl.append(b)
        return al,bl
    def make_batch(al,bl,device):
        B=len(al)
        x=torch.zeros(B,SEQ_TOTAL,dtype=torch.long,device=device)
        y=torch.zeros(B,SEQ_TOTAL,dtype=torch.long,device=device)
        for i,(a,b) in enumerate(zip(al,bl)):
            full=enc_pair(a,b)+enc_sum(a+b)
            x[i]=torch.tensor(full,dtype=torch.long)
            y[i,:-1]=x[i,1:]; y[i,-1]=full[-1]
        return x,y

    model=M().to(DEVICE)
    print(f"\n=== {label} D={D_MODEL} H={NHEAD} FF={D_FF} params={cnt(model)} ===")
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=STEPS)
    best=0
    best_sd=None
    for step in range(1,STEPS+1):
        model.train()
        al,bl=sample(BATCH)
        x,y=make_batch(al,bl,DEVICE)
        logits=model(x)
        loss=F.cross_entropy(logits[:,28:43].reshape(-1,VOCAB), y[:,28:43].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step(); sched.step()
        if step%5000==0 or step==1:
            model.eval()
            correct=0; total=1000
            with torch.no_grad():
                for _ in range(total//100):
                    ae,be=sample(100)
                    for a,b in zip(ae,be):
                        toks=enc_pair(a,b)
                        for _ in range(SEQ_OUT):
                            inp=torch.tensor([toks],dtype=torch.long,device=DEVICE)
                            toks.append(model(inp)[0,-1].argmax().item())
                        if dec_sum(toks[29:44])==a+b: correct+=1
            acc=correct/total
            print(f"  step {step:5d} loss {loss.item():.4f} acc {acc:.4f}")
            if acc>best:
                best=acc
                best_sd={k:v.cpu().clone() for k,v in model.state_dict().items()}
    print(f"  BEST {best:.4f}")
    if best_sd is not None:
        model.load_state_dict(best_sd)
    model.eval()
    correct=0; total=2000
    with torch.no_grad():
        for _ in range(total//100):
            ae,be=sample(100)
            for a,b in zip(ae,be):
                toks=enc_pair(a,b)
                for _ in range(SEQ_OUT):
                    inp=torch.tensor([toks],dtype=torch.long,device=DEVICE)
                    toks.append(model(inp)[0,-1].argmax().item())
                if dec_sum(toks[29:44])==a+b: correct+=1
    acc=correct/total
    edges=[(0,0),(0,MAX_VAL),(MAX_VAL,0),(MAX_VAL,MAX_VAL),(50000000000000,50000000000000),(12345678901234,98765432109876)]
    eok=0
    with torch.no_grad():
        for a,b in edges:
            toks=enc_pair(a,b)
            for _ in range(SEQ_OUT):
                inp=torch.tensor([toks],dtype=torch.long,device=DEVICE)
                toks.append(model(inp)[0,-1].argmax().item())
            if dec_sum(toks[29:44])==a+b: eok+=1
    print(f"  FINAL 2k acc {acc:.4f} edges {eok}/{len(edges)}")
    return best, acc, eok, best_sd, cnt(model)

configs = [
    (16, 2, 24),
    (16, 2, 26),
    (16, 2, 20),
    (14, 2, 28),
    (15, 1, 28),
]

results=[]
for d,h,ff in configs:
    if d % h != 0:
        print(f"skip {d}/{h}")
        continue
    best, acc, eok, sd, pc = run_config(d,h,ff, STEPS=50000, label=f"try {d}/{h}/{ff}")
    results.append((d,h,ff,pc,best,acc,eok))
    print(f"  -> {d}/{h}/{ff} pc={pc} best={best:.4f} final={acc:.4f} edges={eok}")

print("\nSummary:")
for r in results:
    print(r)
