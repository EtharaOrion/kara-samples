import torch
import torch.nn as nn
import torch.nn.functional as F
import random, math, time

MAX_VAL = 99_999_999_999_999
VOCAB = 12
PAD = 10
BOS = 11
SEQ_TOTAL = 44
D_MODEL = 16
NHEAD = 2
N_LAYERS = 1
D_FF = 28
USE_SINUSOIDAL = True
DEVICE = torch.device("cpu")

def encode_pair(a, b):
    s = f"{a:014d}+{b:014d}"
    toks = []
    for ch in s:
        toks.append(10 if ch == '+' else int(ch))
    return toks[::-1]

def encode_sum(c):
    return [int(ch) for ch in f"{c:015d}"[::-1]]

def build_model():
    class AddTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(VOCAB, D_MODEL)
            if USE_SINUSOIDAL:
                pe = torch.zeros(SEQ_TOTAL, D_MODEL)
                pos = torch.arange(0, SEQ_TOTAL, dtype=torch.float).unsqueeze(1)
                div = torch.exp(torch.arange(0, D_MODEL, 2).float() * (-math.log(10000.0) / D_MODEL))
                pe[:, 0::2] = torch.sin(pos * div)
                pe[:, 1::2] = torch.cos(pos * div)
                self.register_buffer('pos_emb', pe)
            else:
                self.pos_emb = nn.Parameter(torch.randn(SEQ_TOTAL, D_MODEL) * 0.1)
            self.layers = nn.ModuleList()
            for _ in range(N_LAYERS):
                self.layers.append(nn.ModuleDict({
                    'attn': nn.MultiheadAttention(D_MODEL, NHEAD, batch_first=True),
                    'ln1': nn.LayerNorm(D_MODEL),
                    'ln2': nn.LayerNorm(D_MODEL),
                    'ff1': nn.Linear(D_MODEL, D_FF),
                    'ff2': nn.Linear(D_FF, D_MODEL),
                }))
            self.ln_f = nn.LayerNorm(D_MODEL)
            self.head = nn.Linear(D_MODEL, VOCAB)
        def forward(self, x):
            B, T = x.shape
            h = self.tok_emb(x) + self.pos_emb[:T].unsqueeze(0)
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            for layer in self.layers:
                a = layer['ln1'](h)
                attn_out, _ = layer['attn'](a, a, a, attn_mask=mask)
                h = h + attn_out
                f = layer['ln2'](h)
                f = layer['ff1'](f)
                f = F.gelu(f)
                f = layer['ff2'](f)
                h = h + f
            h = self.ln_f(h)
            return self.head(h)
    m = AddTransformer()
    print(f"Params: {sum(p.numel() for p in m.parameters())} sinusoidal={USE_SINUSOIDAL}")
    return m

def sample_pair(rng):
    r = rng.random()
    if r < 0.35:
        a = rng.randint(0, MAX_VAL); b = rng.randint(0, MAX_VAL)
    elif r < 0.50:
        da = rng.randint(1, 14); db = rng.randint(1, 14)
        a = rng.randint(10**(da-1) if da>1 else 0, 10**da-1)
        b = rng.randint(10**(db-1) if db>1 else 0, 10**db-1)
    elif r < 0.62:
        a = int(''.join(str(rng.randint(5,9)) for _ in range(14)))
        b = int(''.join(str(rng.randint(5,9)) for _ in range(14)))
        a = min(a,MAX_VAL); b=min(b,MAX_VAL)
    elif r < 0.72:
        a = rng.randint(0,MAX_VAL); b=rng.randint(0,MAX_VAL)
        if rng.random()<0.5: a=a//1000*1000
        if rng.random()<0.5: b=b//1000*1000
    elif r < 0.82:
        k=rng.randint(0,13); base=10**k
        a=base*rng.randint(1,9); b=base*rng.randint(1,9)
    elif r < 0.90:
        a=rng.randint(0,MAX_VAL//10)*10+9
        if rng.random()<0.5: a=a//100*100+99
        if rng.random()<0.3: a=a//1000*1000+999
        b=rng.randint(0,MAX_VAL)
    else:
        a=rng.randint(0,10000); b=rng.randint(0,10000)
    return a,b

def make_batch(bs, rng):
    inp = torch.zeros(bs, SEQ_TOTAL, dtype=torch.long)
    tgt = torch.zeros(bs, SEQ_TOTAL, dtype=torch.long)
    for i in range(bs):
        a,b = sample_pair(rng); c=a+b
        enc_in = encode_pair(a,b); enc_out = encode_sum(c)
        full_in = enc_in + [BOS] + enc_out[:14]
        full_tgt = [-100]*29 + enc_out
        inp[i]=torch.tensor(full_in)
        tgt[i]=torch.tensor(full_tgt)
    return inp, tgt

def evaluate_fast(model, n=1000, seed=0):
    rng=random.Random(seed)
    model.eval(); correct=0
    with torch.no_grad():
        for _ in range(n):
            a=rng.randint(0,MAX_VAL); b=rng.randint(0,MAX_VAL); c=a+b
            enc_in=encode_pair(a,b)
            seq=enc_in+[BOS]+[PAD]*14
            x=torch.tensor([seq],dtype=torch.long)
            preds=[]
            for step in range(15):
                out=model(x)
                pred=int(out[0,29+step].argmax().item())
                preds.append(pred)
                if step+1<15: x[0,29+1+step]=pred
            val=sum(d*10**i for i,d in enumerate(preds))
            if val==c: correct+=1
    return correct/n

torch.manual_seed(0); random.seed(0)
model=build_model(); model.to(DEVICE)
opt=torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.01)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60000)
rng=random.Random(0)
BATCH=512; STEPS=60000; best=0; t0=time.time()
for step in range(1, STEPS+1):
    model.train()
    inp,tgt=make_batch(BATCH,rng)
    logits=model(inp)
    loss=F.cross_entropy(logits.view(-1,VOCAB), tgt.view(-1), ignore_index=-100)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    opt.step(); sched.step()
    if step%500==0:
        print(f"step {step:5d} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e}", flush=True)
    if step%2000==0:
        acc=evaluate_fast(model, n=1000, seed=step)
        elapsed=time.time()-t0
        print(f"  >> step {step} acc {acc:.4f} time {elapsed:.0f}s", flush=True)
        if acc>best:
            best=acc
            torch.save(model.state_dict(), "/workspace/best.pt")
            print(f"  ** new best {best:.4f} saved", flush=True)

# final
print("Loading best...")
model.load_state_dict(torch.load("/workspace/best.pt", map_location=DEVICE))
for n in [1000,5000]:
    acc=evaluate_fast(model,n=n,seed=123)
    print(f"FINAL n={n} acc={acc:.4f}")
edges=[(0,0),(0,MAX_VAL),(MAX_VAL,0),(MAX_VAL,MAX_VAL),(MAX_VAL,1),(12345678901234,98765432109876),(50000000000000,50000000000000)]
model.eval()
with torch.no_grad():
    for a,b in edges:
        c=a+b; enc_in=encode_pair(a,b); seq=enc_in+[BOS]+[PAD]*14
        x=torch.tensor([seq],dtype=torch.long); preds=[]
        for step in range(15):
            out=model(x); pred=int(out[0,29+step].argmax().item()); preds.append(pred)
            if step+1<15: x[0,29+1+step]=pred
        val=sum(d*10**i for i,d in enumerate(preds))
        print(f"edge {a}+{b}={c} pred={val} {'OK' if val==c else 'FAIL'}")

# write submission
sd=torch.load("/workspace/best.pt", map_location="cpu")
# filter out pos_emb buffer if sinusoidal (it's recomputed)
if USE_SINUSOIDAL and 'pos_emb' in sd:
    del sd['pos_emb']

with open("/workspace/submission.py","w") as f:
    f.write("import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nimport math\n")
    f.write(f"D_MODEL={D_MODEL}\nNHEAD={NHEAD}\nN_LAYERS={N_LAYERS}\nD_FF={D_FF}\nUSE_SINUSOIDAL={USE_SINUSOIDAL}\nVOCAB=12\nPAD=10\nBOS=11\nSEQ_TOTAL=44\n")
    f.write("class AddTransformer(nn.Module):\n")
    f.write("    def __init__(self):\n")
    f.write("        super().__init__()\n")
    f.write("        self.tok_emb=nn.Embedding(VOCAB,D_MODEL)\n")
    f.write("        if USE_SINUSOIDAL:\n")
    f.write("            pe=torch.zeros(SEQ_TOTAL,D_MODEL)\n")
    f.write("            pos=torch.arange(0,SEQ_TOTAL,dtype=torch.float).unsqueeze(1)\n")
    f.write("            div=torch.exp(torch.arange(0,D_MODEL,2).float()*(-math.log(10000.0)/D_MODEL))\n")
    f.write("            pe[:,0::2]=torch.sin(pos*div)\n")
    f.write("            pe[:,1::2]=torch.cos(pos*div)\n")
    f.write("            self.register_buffer('pos_emb',pe)\n")
    f.write("        else:\n")
    f.write("            self.pos_emb=nn.Parameter(torch.randn(SEQ_TOTAL,D_MODEL)*0.1)\n")
    f.write("        self.layers=nn.ModuleList()\n")
    f.write("        for _ in range(N_LAYERS):\n")
    f.write("            self.layers.append(nn.ModuleDict({'attn':nn.MultiheadAttention(D_MODEL,NHEAD,batch_first=True),'ln1':nn.LayerNorm(D_MODEL),'ln2':nn.LayerNorm(D_MODEL),'ff1':nn.Linear(D_MODEL,D_FF),'ff2':nn.Linear(D_FF,D_MODEL)}))\n")
    f.write("        self.ln_f=nn.LayerNorm(D_MODEL)\n")
    f.write("        self.head=nn.Linear(D_MODEL,VOCAB)\n")
    f.write("    def forward(self,x):\n")
    f.write("        B,T=x.shape\n")
    f.write("        h=self.tok_emb(x)+self.pos_emb[:T].unsqueeze(0)\n")
    f.write("        mask=torch.triu(torch.ones(T,T,device=x.device,dtype=torch.bool),diagonal=1)\n")
    f.write("        for layer in self.layers:\n")
    f.write("            a=layer['ln1'](h)\n")
    f.write("            attn_out,_=layer['attn'](a,a,a,attn_mask=mask)\n")
    f.write("            h=h+attn_out\n")
    f.write("            f=layer['ln2'](h)\n")
    f.write("            f=layer['ff1'](f)\n")
    f.write("            f=F.gelu(f)\n")
    f.write("            f=layer['ff2'](f)\n")
    f.write("            h=h+f\n")
    f.write("        h=self.ln_f(h)\n")
    f.write("        return self.head(h)\n")
    f.write("def build_model():\n")
    f.write("    m=AddTransformer()\n")
    f.write("    sd=m.state_dict()\n")
    for k,v in sd.items():
        lst=v.cpu().numpy().tolist()
        f.write(f"    sd['{k}']=torch.tensor({repr(lst)}).to(sd['{k}'].dtype).reshape(sd['{k}'].shape)\n")
    f.write("    m.load_state_dict(sd)\n")
    f.write("    return m,{}\n")
    f.write("def add(model,a,b):\n")
    f.write("    model.eval()\n")
    f.write("    s=f\"{a:014d}+{b:014d}\"\n")
    f.write("    toks=[10 if ch=='+' else int(ch) for ch in s][::-1]\n")
    f.write("    seq=toks+[BOS]+[PAD]*14\n")
    f.write("    x=torch.tensor([seq],dtype=torch.long)\n")
    f.write("    preds=[]\n")
    f.write("    with torch.no_grad():\n")
    f.write("        for step in range(15):\n")
    f.write("            out=model(x)\n")
    f.write("            pred=int(out[0,29+step].argmax().item())\n")
    f.write("            preds.append(pred)\n")
    f.write("            if step+1<15:\n")
    f.write("                x[0,29+1+step]=pred\n")
    f.write("    val=0\n")
    f.write("    for i,d in enumerate(preds): val+=d*10**i\n")
    f.write("    return val\n")

print("Wrote submission.py")
# verify
import importlib.util, sys
spec=importlib.util.spec_from_file_location("submission","/workspace/submission.py")
mod=importlib.util.module_from_spec(spec)
sys.modules["submission"]=mod
spec.loader.exec_module(mod)
m2,_=mod.build_model()
print(f"Submission params: {sum(p.numel() for p in m2.parameters())}")
for a,b in [(0,0),(MAX_VAL,MAX_VAL),(12345678901234,98765432109876),(50000000000000,50000000000000)]:
    print(f"  {a}+{b}={mod.add(m2,a,b)} expected {a+b} {'OK' if mod.add(m2,a,b)==a+b else 'FAIL'}")
# quick acc
acc=evaluate_fast(m2,n=500,seed=999)
print(f"Submission acc 500: {acc:.4f}")
