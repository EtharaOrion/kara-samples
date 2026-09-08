import sys
sys.path = [p for p in sys.path if 'openhands-venv' not in p and 'python3.13' not in p]
import torch, torch.nn as nn, torch.nn.functional as F, math, random
VOCAB=12; SEQ_IN=29; SEQ_OUT=15; SEQ_TOTAL=44; MAX_VAL=99_999_999_999_999
DEVICE='cuda'
print(f"DEVICE {DEVICE} torch {torch.__version__}")

D_MODEL=12; NHEAD=2; D_FF=24
print(f"Config D{D_MODEL} H{NHEAD} FF{D_FF}")

class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB, D_MODEL)
        self.pos_emb = nn.Parameter(torch.randn(SEQ_TOTAL, D_MODEL)*0.1)
        self.attn = nn.MultiheadAttention(D_MODEL, NHEAD, batch_first=True)
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.ff1 = nn.Linear(D_MODEL, D_FF)
        self.ff2 = nn.Linear(D_FF, D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB)
    def forward(self, x):
        B,T=x.shape
        h=self.tok_emb(x)+self.pos_emb[:T].unsqueeze(0)
        mask=torch.triu(torch.ones(T,T,device=x.device,dtype=torch.bool),diagonal=1)
        a_out,_=self.attn(h,h,h,attn_mask=mask,need_weights=False)
        h=self.ln1(h+a_out)
        ff=self.ff2(F.gelu(self.ff1(h)))
        h=self.ln2(h+ff)
        h=self.ln_f(h)
        return self.head(h)

def encode_pair(a,b):
    toks=[]
    for _ in range(14):
        toks.append(a%10); a//=10
    toks.append(10)
    for _ in range(14):
        toks.append(b%10); b//=10
    return toks

def make_batch(bs):
    inp=torch.zeros(bs,SEQ_TOTAL,dtype=torch.long)
    for i in range(bs):
        r=random.random()
        if r<0.35:
            a=random.randint(0,MAX_VAL); b=random.randint(0,MAX_VAL)
        elif r<0.50:
            da=random.randint(1,14); db=random.randint(1,14)
            a=random.randint(10**(da-1) if da>1 else 0, 10**da-1)
            b=random.randint(10**(db-1) if db>1 else 0, 10**db-1)
            a=min(a,MAX_VAL); b=min(b,MAX_VAL)
        elif r<0.62:
            a=0; b=0
            for pos in range(14):
                da=random.randint(5,9); db=random.randint(5,9)
                if random.random()<0.3:
                    da=random.randint(0,9); db=random.randint(0,9)
                a+=da*(10**pos); b+=db*(10**pos)
            a=min(a,MAX_VAL); b=min(b,MAX_VAL)
        elif r<0.72:
            if random.random()<0.5:
                a=0; b=random.randint(0,MAX_VAL)
            else:
                a=random.randint(0,MAX_VAL); b=0
            if random.random()<0.3:
                a=random.randint(0,1000); b=random.randint(0,1000)
        elif r<0.82:
            if random.random()<0.5:
                k=random.randint(0,13)
                a=10**k; b=10**k
                if random.random()<0.5:
                    b=MAX_VAL-random.randint(0,1000)
                    a=random.randint(0,1000)
            else:
                a=random.randint(0,MAX_VAL)
                b=(10**14-1)-a+random.randint(-5,5)
                b=max(0,min(MAX_VAL,b))
        elif r<0.90:
            nines=random.randint(1,10)
            base_a=random.randint(0,10**(14-nines)-1) if nines<14 else 0
            a=base_a*(10**nines)+(10**nines-1)
            b=random.randint(0,MAX_VAL)
            if random.random()<0.5:
                b,a=a,b
            a=min(a,MAX_VAL); b=min(b,MAX_VAL)
        else:
            a=random.randint(0,10000); b=random.randint(0,10000)
        enc=encode_pair(a,b)
        s=a+b
        out=[(s//(10**i))%10 for i in range(15)]
        full=enc+out
        inp[i]=torch.tensor(full)
    return inp

def evaluate(model, n=500):
    model.eval()
    correct=0
    with torch.no_grad():
        for _ in range(n):
            a=random.randint(0,MAX_VAL); b=random.randint(0,MAX_VAL)
            enc=encode_pair(a,b)
            seq=torch.tensor([enc],dtype=torch.long,device=DEVICE)
            for _ in range(SEQ_OUT):
                logits=model(seq)
                nxt=logits[0,-1].argmax().item()
                seq=torch.cat([seq,torch.tensor([[nxt]],device=DEVICE)],dim=1)
            pred=sum(d*(10**i) for i,d in enumerate(seq[0,SEQ_IN:].tolist()))
            if pred==a+b:
                correct+=1
    return correct/n

model=TinyTransformer().to(DEVICE)
n_params=sum(p.numel() for p in model.parameters())
print(f"Params: {n_params}")
opt=torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.01)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60000)
crit=nn.CrossEntropyLoss()
best=0
BATCH=512
TOTAL=60000
for step in range(1, TOTAL+1):
    model.train()
    inp=make_batch(BATCH).to(DEVICE)
    x=inp[:,:-1]; y=inp[:,1:]
    logits=model(x)
    loss=crit(logits[:,SEQ_IN-1:].reshape(-1,VOCAB), y[:,SEQ_IN-1:].reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    opt.step(); sched.step()
    if step%2000==0:
        print(f"step {step} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.6f}", flush=True)
    if step%5000==0:
        acc=evaluate(model, n=500)
        print(f"  eval 500: {acc*100:.2f}%", flush=True)
        if acc>best:
            best=acc
            torch.save(model.state_dict(), "/workspace/best.pt")
            print(f"  new best {best*100:.2f}% saved", flush=True)
        if acc>=0.99:
            print(f"  reached 99% at step {step}", flush=True)

# Final eval
print("Loading best...")
model.load_state_dict(torch.load("/workspace/best.pt", map_location=DEVICE))
acc=evaluate(model, n=2000)
print(f"FINAL 2000: {acc*100:.2f}%")
acc2=evaluate(model, n=5000)
print(f"FINAL 5000: {acc2*100:.2f}%")
edges=[(0,0),(0,MAX_VAL),(MAX_VAL,0),(MAX_VAL,MAX_VAL),(MAX_VAL,1),(12345678901234,98765432109876),(50000000000000,50000000000000),(99999999999999,1),(10000000000000,10000000000000)]
print("Edges:")
model.eval()
with torch.no_grad():
    for a,b in edges:
        enc=encode_pair(a,b)
        seq=torch.tensor([enc],dtype=torch.long,device=DEVICE)
        for _ in range(SEQ_OUT):
            logits=model(seq)
            nxt=logits[0,-1].argmax().item()
            seq=torch.cat([seq,torch.tensor([[nxt]],device=DEVICE)],dim=1)
        pred=sum(d*(10**i) for i,d in enumerate(seq[0,SEQ_IN:].tolist()))
        ok="OK" if pred==a+b else "FAIL"
        print(f"  {a}+{b}={a+b} pred {pred} {ok}")

# Write submission.py
sd=torch.load("/workspace/best.pt", map_location="cpu")
def t2l(t): return t.detach().cpu().numpy().tolist()
weights_repr = repr({k: t2l(v) for k,v in sd.items()})
code = f'''import torch
import torch.nn as nn
import torch.nn.functional as F
import math
VOCAB=12; SEQ_IN=29; SEQ_OUT=15; SEQ_TOTAL=44; D_MODEL={D_MODEL}; NHEAD={NHEAD}; D_FF={D_FF}
class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb=nn.Embedding(VOCAB, D_MODEL)
        self.pos_emb=nn.Parameter(torch.randn(SEQ_TOTAL, D_MODEL)*0.1)
        self.attn=nn.MultiheadAttention(D_MODEL, NHEAD, batch_first=True)
        self.ln1=nn.LayerNorm(D_MODEL)
        self.ff1=nn.Linear(D_MODEL, D_FF)
        self.ff2=nn.Linear(D_FF, D_MODEL)
        self.ln2=nn.LayerNorm(D_MODEL)
        self.ln_f=nn.LayerNorm(D_MODEL)
        self.head=nn.Linear(D_MODEL, VOCAB)
    def forward(self, x):
        B,T=x.shape
        h=self.tok_emb(x)+self.pos_emb[:T].unsqueeze(0)
        mask=torch.triu(torch.ones(T,T,device=x.device,dtype=torch.bool),diagonal=1)
        a_out,_=self.attn(h,h,h,attn_mask=mask,need_weights=False)
        h=self.ln1(h+a_out)
        ff=self.ff2(F.gelu(self.ff1(h)))
        h=self.ln2(h+ff)
        h=self.ln_f(h)
        return self.head(h)
_weights={weights_repr}
def build_model():
    model=TinyTransformer()
    sd={{k: torch.tensor(v) for k,v in _weights.items()}}
    model.load_state_dict(sd)
    model.eval()
    return model, {{"vocab": VOCAB, "seq_in": SEQ_IN, "seq_out": SEQ_OUT}}
def add(model, a: int, b: int) -> int:
    model.eval()
    device=next(model.parameters()).device
    toks=[]
    aa,bb=a,b
    for _ in range(14):
        toks.append(aa%10); aa//=10
    toks.append(10)
    for _ in range(14):
        toks.append(bb%10); bb//=10
    seq=torch.tensor([toks], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(SEQ_OUT):
            logits=model(seq)
            nxt=int(logits[0,-1].argmax().item())
            seq=torch.cat([seq, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
    digits=seq[0, SEQ_IN:].tolist()
    s=0
    for i,d in enumerate(digits):
        s+=d*(10**i)
    return int(s)
'''
open("/workspace/submission.py","w").write(code)
print("Wrote /workspace/submission.py")
# verify
import importlib.util
spec=importlib.util.spec_from_file_location("submission","/workspace/submission.py")
mod=importlib.util.module_from_spec(spec)
import sys; sys.modules["submission"]=mod
spec.loader.exec_module(mod)
m,meta=mod.build_model()
print("verify add 123+456=",mod.add(m,123,456))
print("verify max+max=",mod.add(m,MAX_VAL,MAX_VAL), "expected",MAX_VAL*2)
print("params",sum(p.numel() for p in m.parameters()))
# check imports
import ast
tree=ast.parse(open("/workspace/submission.py").read())
imports=[n.names[0].name if isinstance(n, ast.Import) else n.module for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
print("imports:",imports)
# corruption test
orig=mod.add(m, 12345, 67890)
# corrupt weights
with torch.no_grad():
    for p in m.parameters():
        p.add_(torch.randn_like(p)*0.5)
        break
corrupted=mod.add(m, 12345, 67890)
print(f"corruption test: orig {orig} corrupted {corrupted} changed={orig!=corrupted}")
