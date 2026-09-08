import torch, random
import torch.nn as nn
import torch.nn.functional as F

DEVICE="cpu"
D_MODEL=16; NHEAD=2; N_LAYERS=1; D_FF=28; VOCAB=12; SEQ_IN=29; SEQ_OUT=15; SEQ_TOTAL=44
MAX_VAL=99_999_999_999_999; BATCH=512; STEPS=60000; LR=4e-4; WD=0.01

class AddTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb=nn.Embedding(VOCAB,D_MODEL)
        self.pos_emb=nn.Embedding(SEQ_TOTAL,D_MODEL)
        self.layers=nn.ModuleList()
        for _ in range(N_LAYERS):
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

model=AddTransformer().to(DEVICE)
print(f"Params: {cnt(model)}")
opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=STEPS)
best_acc=0
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
    if step%2000==0 or step==1:
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
        edges=[(0,0),(0,MAX_VAL),(MAX_VAL,0),(MAX_VAL,MAX_VAL),(50000000000000,50000000000000),(12345678901234,98765432109876),(0,1),(1,0),(999,1),(10000000000000,10000000000000)]
        eok=0
        with torch.no_grad():
            for a,b in edges:
                toks=enc_pair(a,b)
                for _ in range(SEQ_OUT):
                    inp=torch.tensor([toks],dtype=torch.long,device=DEVICE)
                    toks.append(model(inp)[0,-1].argmax().item())
                if dec_sum(toks[29:44])==a+b: eok+=1
        print(f"Step {step:5d} loss {loss.item():.5f} acc {acc:.4f} edges {eok}/{len(edges)}")
        if acc>=0.99 and eok==len(edges):
            if acc>best_acc:
                best_acc=acc
                best_sd={k:v.cpu().clone() for k,v in model.state_dict().items()}
                torch.save(best_sd,"/workspace/best_final.pt")
                print(f"  -> saved best {acc:.4f}")

print(f"Best {best_acc:.4f}")
if best_sd is not None:
    model.load_state_dict(best_sd)
else:
    # save current if no best
    torch.save({k:v.cpu().clone() for k,v in model.state_dict().items()},"/workspace/best_final.pt")
    print("No best found, saved current")

model.eval()
correct=0; total=5000
with torch.no_grad():
    for _ in range(total//100):
        ae,be=sample(100)
        for a,b in zip(ae,be):
            toks=enc_pair(a,b)
            for _ in range(SEQ_OUT):
                inp=torch.tensor([toks],dtype=torch.long,device=DEVICE)
                toks.append(model(inp)[0,-1].argmax().item())
            if dec_sum(toks[29:44])==a+b: correct+=1
print(f"Final 5k: {correct}/{total}={correct/total:.4f}")
for a,b in [(0,0),(0,MAX_VAL),(MAX_VAL,0),(MAX_VAL,MAX_VAL),(50000000000000,50000000000000),(12345678901234,98765432109876),(0,1),(1,0)]:
    toks=enc_pair(a,b)
    with torch.no_grad():
        for _ in range(SEQ_OUT):
            inp=torch.tensor([toks],dtype=torch.long,device=DEVICE)
            toks.append(model(inp)[0,-1].argmax().item())
    pred=dec_sum(toks[29:44])
    print(f"  {a}+{b}={a+b} pred={pred} {'OK' if pred==a+b else 'FAIL'}")

# write submission
sd=torch.load("/workspace/best_final.pt",map_location="cpu")
lines=[]
lines.append("import torch")
lines.append("import torch.nn as nn")
lines.append("import torch.nn.functional as F")
lines.append("")
lines.append(f"D_MODEL={D_MODEL}")
lines.append(f"NHEAD={NHEAD}")
lines.append(f"N_LAYERS={N_LAYERS}")
lines.append(f"D_FF={D_FF}")
lines.append(f"VOCAB={VOCAB}")
lines.append(f"SEQ_TOTAL={SEQ_TOTAL}")
lines.append(f"SEQ_IN={SEQ_IN}")
lines.append(f"SEQ_OUT={SEQ_OUT}")
lines.append("")
lines.append("class AddTransformer(nn.Module):")
lines.append("    def __init__(self):")
lines.append("        super().__init__()")
lines.append("        self.tok_emb = nn.Embedding(VOCAB, D_MODEL)")
lines.append("        self.pos_emb = nn.Embedding(SEQ_TOTAL, D_MODEL)")
lines.append("        self.layers = nn.ModuleList()")
lines.append("        for _ in range(N_LAYERS):")
lines.append("            self.layers.append(nn.ModuleDict({")
lines.append("                'attn': nn.MultiheadAttention(D_MODEL, NHEAD, batch_first=True),")
lines.append("                'ln1': nn.LayerNorm(D_MODEL),")
lines.append("                'ln2': nn.LayerNorm(D_MODEL),")
lines.append("                'ff1': nn.Linear(D_MODEL, D_FF),")
lines.append("                'ff2': nn.Linear(D_FF, D_MODEL),")
lines.append("            }))")
lines.append("        self.ln_f = nn.LayerNorm(D_MODEL)")
lines.append("        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)")
lines.append("    def forward(self, x):")
lines.append("        B, T = x.shape")
lines.append("        h = self.tok_emb(x) + self.pos_emb(torch.arange(T, device=x.device))")
lines.append("        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)")
lines.append("        for layer in self.layers:")
lines.append("            h_norm = layer['ln1'](h)")
lines.append("            attn_out, _ = layer['attn'](h_norm, h_norm, h_norm, attn_mask=causal)")
lines.append("            h = h + attn_out")
lines.append("            h_norm2 = layer['ln2'](h)")
lines.append("            ff = layer['ff2'](F.gelu(layer['ff1'](h_norm2)))")
lines.append("            h = h + ff")
lines.append("        h = self.ln_f(h)")
lines.append("        return self.head(h)")
lines.append("")
lines.append("_model = None")
lines.append("def _get_model():")
lines.append("    global _model")
lines.append("    if _model is not None: return _model")
lines.append("    _model = AddTransformer()")
lines.append("    _sd = {")
for k,v in sd.items():
    lines.append(f"        {repr(k)}: torch.tensor({repr(v.numpy().tolist())}),")
lines.append("    }")
lines.append("    _model.load_state_dict(_sd)")
lines.append("    _model.eval()")
lines.append("    return _model")
lines.append("def build_model():")
lines.append("    return _get_model(), {}")
lines.append("def add(model, a, b):")
lines.append("    model.eval()")
lines.append("    with torch.no_grad():")
lines.append("        toks = []")
lines.append("        for i in range(14): toks.append((a // (10**i)) % 10)")
lines.append("        toks.append(10)")
lines.append("        for i in range(14): toks.append((b // (10**i)) % 10)")
lines.append("        for _ in range(SEQ_OUT):")
lines.append("            inp = torch.tensor([toks], dtype=torch.long)")
lines.append("            toks.append(int(model(inp)[0, -1].argmax().item()))")
lines.append("        return sum(t * (10**i) for i, t in enumerate(toks[SEQ_IN:SEQ_IN+SEQ_OUT]))")
with open("/workspace/submission.py","w") as f: f.write("\n".join(lines))
print("Wrote submission.py")
import sys
if 'submission' in sys.modules: del sys.modules['submission']
import submission as sub2
m2,_=sub2.build_model()
print(f"params: {sum(p.numel() for p in m2.parameters())}")
for a,b in [(0,0),(MAX_VAL,MAX_VAL),(50000000000000,50000000000000),(12345678901234,98765432109876)]:
    r=sub2.add(m2,a,b)
    print(f"verify {a}+{b}={a+b} got {r} {'OK' if r==a+b else 'FAIL'}")
# thorough random test
import random as rnd
rnd.seed(1)
fail=0
for _ in range(5000):
    a=rnd.randint(0,MAX_VAL); b=rnd.randint(0,MAX_VAL)
    r=sub2.add(m2,a,b)
    if r!=a+b: fail+=1
print(f"random 5000: {5000-fail}/5000={ (5000-fail)/5000:.4f}")
