import torch, torch.nn as nn, torch.nn.functional as F, random, math, time
MAX_VAL=99_999_999_999_999; VOCAB=12; PAD=10; BOS=11; SEQ_TOTAL=44
D_MODEL=16; NHEAD=2; N_LAYERS=1; D_FF=19; USE_SINUSOIDAL=True
DEVICE=torch.device("cpu")
def encode_pair(a,b):
    s=f"{a:014d}+{b:014d}"; return [10 if ch=='+' else int(ch) for ch in s][::-1]
def encode_sum(c): return [int(ch) for ch in f"{c:015d}"[::-1]]
def build_model():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb=nn.Embedding(VOCAB,D_MODEL)
            pe=torch.zeros(SEQ_TOTAL,D_MODEL)
            pos=torch.arange(0,SEQ_TOTAL,dtype=torch.float).unsqueeze(1)
            div=torch.exp(torch.arange(0,D_MODEL,2).float()*(-math.log(10000.0)/D_MODEL))
            pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
            self.register_buffer('pos_emb',pe)
            self.layers=nn.ModuleList()
            for _ in range(N_LAYERS):
                self.layers.append(nn.ModuleDict({'attn':nn.MultiheadAttention(D_MODEL,NHEAD,batch_first=True),'ln1':nn.LayerNorm(D_MODEL),'ln2':nn.LayerNorm(D_MODEL),'ff1':nn.Linear(D_MODEL,D_FF),'ff2':nn.Linear(D_FF,D_MODEL)}))
            self.ln_f=nn.LayerNorm(D_MODEL); self.head=nn.Linear(D_MODEL,VOCAB)
        def forward(self,x):
            h=self.tok_emb(x)+self.pos_emb[:x.shape[1]].unsqueeze(0)
            mask=torch.triu(torch.ones(x.shape[1],x.shape[1],device=x.device,dtype=torch.bool),diagonal=1)
            for layer in self.layers:
                a=layer['ln1'](h); attn_out,_=layer['attn'](a,a,a,attn_mask=mask); h=h+attn_out
                f=layer['ln2'](h); f=layer['ff1'](f); f=F.gelu(f); f=layer['ff2'](f); h=h+f
            return self.head(self.ln_f(h))
    m=M(); print(f"Params: {sum(p.numel() for p in m.parameters())} D_FF={D_FF}"); return m
def sample_pair(rng):
    r=rng.random()
    if r<0.35: a=rng.randint(0,MAX_VAL); b=rng.randint(0,MAX_VAL)
    elif r<0.50:
        da=rng.randint(1,14); db=rng.randint(1,14)
        a=rng.randint(10**(da-1) if da>1 else 0,10**da-1); b=rng.randint(10**(db-1) if db>1 else 0,10**db-1)
    elif r<0.62:
        a=int(''.join(str(rng.randint(5,9)) for _ in range(14))); b=int(''.join(str(rng.randint(5,9)) for _ in range(14))); a=min(a,MAX_VAL); b=min(b,MAX_VAL)
    elif r<0.72:
        a=rng.randint(0,MAX_VAL); b=rng.randint(0,MAX_VAL)
        if rng.random()<0.5: a=a//1000*1000
        if rng.random()<0.5: b=b//1000*1000
    elif r<0.82: k=rng.randint(0,13); base=10**k; a=base*rng.randint(1,9); b=base*rng.randint(1,9)
    elif r<0.90:
        a=rng.randint(0,MAX_VAL//10)*10+9
        if rng.random()<0.5: a=a//100*100+99
        if rng.random()<0.3: a=a//1000*1000+999
        b=rng.randint(0,MAX_VAL)
    else: a=rng.randint(0,10000); b=rng.randint(0,10000)
    return a,b
def make_batch(bs,rng):
    inp=torch.zeros(bs,SEQ_TOTAL,dtype=torch.long); tgt=torch.zeros(bs,SEQ_TOTAL,dtype=torch.long)
    for i in range(bs):
        a,b=sample_pair(rng); c=a+b; enc_in=encode_pair(a,b); enc_out=encode_sum(c)
        inp[i]=torch.tensor(enc_in+[BOS]+enc_out[:14]); tgt[i]=torch.tensor([-100]*29+enc_out)
    return inp,tgt
def eval_fast(model,n=1000,seed=0):
    rng=random.Random(seed); model.eval(); correct=0
    with torch.no_grad():
        for _ in range(n):
            a=rng.randint(0,MAX_VAL); b=rng.randint(0,MAX_VAL); c=a+b
            enc_in=encode_pair(a,b); seq=enc_in+[BOS]+[PAD]*14; x=torch.tensor([seq],dtype=torch.long); preds=[]
            for step in range(15):
                out=model(x); pred=int(out[0,29+step].argmax().item()); preds.append(pred)
                if step+1<15: x[0,29+1+step]=pred
            if sum(d*10**i for i,d in enumerate(preds))==c: correct+=1
    return correct/n
torch.manual_seed(0); random.seed(0)
model=build_model(); model.to(DEVICE)
opt=torch.optim.AdamW(model.parameters(),lr=4e-4,weight_decay=0.01)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=60000)
rng=random.Random(0); BATCH=512; STEPS=60000; best=0; t0=time.time()
for step in range(1,STEPS+1):
    model.train(); inp,tgt=make_batch(BATCH,rng); logits=model(inp)
    loss=F.cross_entropy(logits.view(-1,VOCAB),tgt.view(-1),ignore_index=-100)
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sched.step()
    if step%500==0: print(f"step {step:5d} loss {loss.item():.4f}",flush=True)
    if step%2000==0:
        acc=eval_fast(model,n=1000,seed=step); print(f"  >> acc {acc:.4f} time {time.time()-t0:.0f}s",flush=True)
        if acc>best: best=acc; torch.save(model.state_dict(),"/workspace/best_19.pt"); print(f"  ** best {best:.4f}",flush=True)
print(f"Best {best}")
try:
    model.load_state_dict(torch.load("/workspace/best_19.pt",map_location=DEVICE))
    acc=eval_fast(model,n=2000,seed=999); print(f"final 2000 acc {acc:.4f}")
    if acc>=0.99:
        sd=torch.load("/workspace/best_19.pt",map_location="cpu")
        if 'pos_emb' in sd: del sd['pos_emb']
        with open("/workspace/submission.py","w") as f:
            f.write("import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nimport math\n")
            f.write(f"D_MODEL={D_MODEL}\nNHEAD={NHEAD}\nN_LAYERS={N_LAYERS}\nD_FF={D_FF}\nUSE_SINUSOIDAL=True\nVOCAB=12\nPAD=10\nBOS=11\nSEQ_TOTAL=44\n")
            f.write("class AddTransformer(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.tok_emb=nn.Embedding(VOCAB,D_MODEL)\n        pe=torch.zeros(SEQ_TOTAL,D_MODEL)\n        pos=torch.arange(0,SEQ_TOTAL,dtype=torch.float).unsqueeze(1)\n        div=torch.exp(torch.arange(0,D_MODEL,2).float()*(-math.log(10000.0)/D_MODEL))\n        pe[:,0::2]=torch.sin(pos*div)\n        pe[:,1::2]=torch.cos(pos*div)\n        self.register_buffer('pos_emb',pe)\n        self.layers=nn.ModuleList()\n        for _ in range(N_LAYERS):\n            self.layers.append(nn.ModuleDict({'attn':nn.MultiheadAttention(D_MODEL,NHEAD,batch_first=True),'ln1':nn.LayerNorm(D_MODEL),'ln2':nn.LayerNorm(D_MODEL),'ff1':nn.Linear(D_MODEL,D_FF),'ff2':nn.Linear(D_FF,D_MODEL)}))\n        self.ln_f=nn.LayerNorm(D_MODEL)\n        self.head=nn.Linear(D_MODEL,VOCAB)\n    def forward(self,x):\n        h=self.tok_emb(x)+self.pos_emb[:x.shape[1]].unsqueeze(0)\n        mask=torch.triu(torch.ones(x.shape[1],x.shape[1],device=x.device,dtype=torch.bool),diagonal=1)\n        for layer in self.layers:\n            a=layer['ln1'](h); attn_out,_=layer['attn'](a,a,a,attn_mask=mask); h=h+attn_out\n            f=layer['ln2'](h); f=layer['ff1'](f); f=F.gelu(f); f=layer['ff2'](f); h=h+f\n        return self.head(self.ln_f(h))\n")
            f.write("def build_model():\n    m=AddTransformer()\n    sd=m.state_dict()\n")
            for k,v in sd.items():
                f.write(f"    sd['{k}']=torch.tensor({repr(v.cpu().numpy().tolist())}).to(sd['{k}'].dtype).reshape(sd['{k}'].shape)\n")
            f.write("    m.load_state_dict(sd)\n    return m,{}\n")
            f.write("def add(model,a,b):\n    model.eval()\n    s=f\"{a:014d}+{b:014d}\"\n    toks=[10 if ch=='+' else int(ch) for ch in s][::-1]\n    seq=toks+[BOS]+[PAD]*14\n    x=torch.tensor([seq],dtype=torch.long)\n    preds=[]\n    with torch.no_grad():\n        for step in range(15):\n            out=model(x)\n            pred=int(out[0,29+step].argmax().item())\n            preds.append(pred)\n            if step+1<15: x[0,29+1+step]=pred\n    val=0\n    for i,d in enumerate(preds): val+=d*10**i\n    return val\n")
        print(f"Wrote submission {sum(p.numel() for p in model.parameters())} params")
    else: print("NOT overwriting")
except Exception as e: print(f"err {e}")
