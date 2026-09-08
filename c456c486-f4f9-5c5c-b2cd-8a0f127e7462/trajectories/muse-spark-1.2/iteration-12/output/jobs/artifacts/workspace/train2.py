import torch, torch.nn as nn, torch.nn.functional as F, math, random, time, os
MAX_VAL=99_999_999_999_999
VOCAB=12; SEQ_IN=29; SEQ_OUT=15; SEQ_TOTAL=44
D_MODEL=16; NHEAD=2; D_FF=32
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
print(f"Device {DEVICE} D={D_MODEL} FF={D_FF}")

torch.manual_seed(42)
random.seed(42)

class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb=nn.Embedding(VOCAB,D_MODEL)
        pe=torch.zeros(SEQ_TOTAL,D_MODEL)
        pos=torch.arange(0,SEQ_TOTAL,dtype=torch.float).unsqueeze(1)
        div=torch.exp(torch.arange(0,D_MODEL,2).float()*(-math.log(10000.0)/D_MODEL))
        pe[:,0::2]=torch.sin(pos*div)
        pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer('pos_emb',pe)
        self.attn=nn.MultiheadAttention(D_MODEL,NHEAD,batch_first=True)
        self.ln1=nn.LayerNorm(D_MODEL)
        self.ff1=nn.Linear(D_MODEL,D_FF)
        self.ff2=nn.Linear(D_FF,D_MODEL)
        self.ln2=nn.LayerNorm(D_MODEL)
        self.ln_f=nn.LayerNorm(D_MODEL)
        self.head=nn.Linear(D_MODEL,VOCAB)
    def forward(self,x):
        h=self.tok_emb(x)+self.pos_emb[:x.shape[1]].unsqueeze(0)
        mask=torch.triu(torch.full((x.shape[1],x.shape[1]),float('-inf'),device=x.device),diagonal=1)
        h2,_=self.attn(h,h,h,attn_mask=mask)
        h=self.ln1(h+h2)
        h2=self.ff2(F.gelu(self.ff1(h)))
        h=self.ln2(h+h2)
        h=self.ln_f(h)
        return self.head(h)

def encode_pair(a,b):
    s=[]
    for i in range(14): s.append((a//(10**i))%10)
    s.append(10)
    for i in range(14): s.append((b//(10**i))%10)
    return s
def encode_sum(c): return [(c//(10**i))%10 for i in range(15)]

def make_batch(bs):
    inp=torch.zeros(bs,SEQ_TOTAL,dtype=torch.long)
    tgt=torch.zeros(bs,SEQ_TOTAL,dtype=torch.long)
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
            a=int(''.join(str(random.randint(5,9)) for _ in range(14))[:14])%(MAX_VAL+1)
            b=int(''.join(str(random.randint(5,9)) for _ in range(14))[:14])%(MAX_VAL+1)
        elif r<0.72:
            if random.random()<0.5: a=0; b=random.randint(0,MAX_VAL)
            else: a=random.randint(0,MAX_VAL); b=0
        elif r<0.82:
            k=random.randint(0,13); a=10**k; b=10**k
            if random.random()<0.5:
                a=MAX_VAL-random.randint(0,1000); b=random.randint(1,1000)
        elif r<0.90:
            k=random.randint(1,10); a=int('9'*k) if k<=14 else MAX_VAL; b=random.randint(1,10**k)
            a=min(a,MAX_VAL); b=min(b,MAX_VAL)
        else:
            a=random.randint(0,9999); b=random.randint(0,9999)
        c=a+b
        full=encode_pair(a,b)+encode_sum(c)
        inp[i]=torch.tensor(full); tgt[i]=torch.tensor(full)
    return inp,tgt

def evaluate(model,n=1000):
    model.eval()
    correct=0
    with torch.no_grad():
        for _ in range(n):
            a=random.randint(0,MAX_VAL); b=random.randint(0,MAX_VAL); c=a+b
            inp_seq=encode_pair(a,b)
            gen=[]
            for step in range(15):
                seq=torch.tensor([inp_seq+gen],dtype=torch.long,device=DEVICE)
                logits=model(seq)
                gen.append(int(logits[0,-1].argmax().item()))
            pred=sum(d*(10**i) for i,d in enumerate(gen))
            if pred==c: correct+=1
    return correct/n

model=TinyTransformer().to(DEVICE)
opt=torch.optim.AdamW(model.parameters(),lr=4e-4,weight_decay=0.01)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=60000)
best=0; best_state=None
t0=time.time()
for step in range(1,60001):
    model.train()
    inp,tgt=make_batch(512)
    inp=inp.to(DEVICE); tgt=tgt.to(DEVICE)
    logits=model(inp)
    loss=F.cross_entropy(logits[:,SEQ_IN-1:-1].reshape(-1,VOCAB), tgt[:,SEQ_IN:].reshape(-1))
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sched.step()
    if step%500==0:
        print(f"step {step} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.6f} t {time.time()-t0:.0f}s")
    if step%2000==0:
        acc=evaluate(model,1000)
        print(f" eval {acc:.4f}")
        if acc>best:
            best=acc; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
            torch.save(best_state,"/workspace/best.pt")
            print(f" *** best {best:.4f}")
        if acc>=0.99:
            print(" reached 99%")
print(f"done best {best}")
if best_state is not None:
    torch.save(best_state,"/workspace/best.pt")
    model.load_state_dict({k:v.to(DEVICE) for k,v in best_state.items()})
    print("final 5000:",evaluate(model,5000))
    # write submission
    m_cpu=TinyTransformer(); m_cpu.load_state_dict(best_state)
    total=sum(p.numel() for p in m_cpu.parameters())
    print(f"params {total}")
    def tl(t): return t.detach().cpu().numpy().tolist()
    tok_w=tl(m_cpu.tok_emb.weight)
    attn_in_w=tl(m_cpu.attn.in_proj_weight); attn_in_b=tl(m_cpu.attn.in_proj_bias)
    attn_out_w=tl(m_cpu.attn.out_proj.weight); attn_out_b=tl(m_cpu.attn.out_proj.bias)
    ln1_w=tl(m_cpu.ln1.weight); ln1_b=tl(m_cpu.ln1.bias)
    ff1_w=tl(m_cpu.ff1.weight); ff1_b=tl(m_cpu.ff1.bias)
    ff2_w=tl(m_cpu.ff2.weight); ff2_b=tl(m_cpu.ff2.bias)
    ln2_w=tl(m_cpu.ln2.weight); ln2_b=tl(m_cpu.ln2.bias)
    lnf_w=tl(m_cpu.ln_f.weight); lnf_b=tl(m_cpu.ln_f.bias)
    head_w=tl(m_cpu.head.weight); head_b=tl(m_cpu.head.bias)
    code=f'''import torch
import torch.nn as nn
import torch.nn.functional as F
import math
VOCAB=12; SEQ_IN=29; SEQ_OUT=15; SEQ_TOTAL=44; D_MODEL={D_MODEL}; NHEAD={NHEAD}; D_FF={D_FF}
class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb=nn.Embedding(VOCAB,D_MODEL)
        pe=torch.zeros(SEQ_TOTAL,D_MODEL)
        pos=torch.arange(0,SEQ_TOTAL,dtype=torch.float).unsqueeze(1)
        div=torch.exp(torch.arange(0,D_MODEL,2).float()*(-math.log(10000.0)/D_MODEL))
        pe[:,0::2]=torch.sin(pos*div)
        pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer('pos_emb',pe)
        self.attn=nn.MultiheadAttention(D_MODEL,NHEAD,batch_first=True)
        self.ln1=nn.LayerNorm(D_MODEL)
        self.ff1=nn.Linear(D_MODEL,D_FF)
        self.ff2=nn.Linear(D_FF,D_MODEL)
        self.ln2=nn.LayerNorm(D_MODEL)
        self.ln_f=nn.LayerNorm(D_MODEL)
        self.head=nn.Linear(D_MODEL,VOCAB)
        self._load()
    def _load(self):
        import torch as _t
        self.tok_emb.weight.data=_t.tensor({tok_w},dtype=_t.float32)
        self.attn.in_proj_weight.data=_t.tensor({attn_in_w},dtype=_t.float32)
        self.attn.in_proj_bias.data=_t.tensor({attn_in_b},dtype=_t.float32)
        self.attn.out_proj.weight.data=_t.tensor({attn_out_w},dtype=_t.float32)
        self.attn.out_proj.bias.data=_t.tensor({attn_out_b},dtype=_t.float32)
        self.ln1.weight.data=_t.tensor({ln1_w},dtype=_t.float32)
        self.ln1.bias.data=_t.tensor({ln1_b},dtype=_t.float32)
        self.ff1.weight.data=_t.tensor({ff1_w},dtype=_t.float32)
        self.ff1.bias.data=_t.tensor({ff1_b},dtype=_t.float32)
        self.ff2.weight.data=_t.tensor({ff2_w},dtype=_t.float32)
        self.ff2.bias.data=_t.tensor({ff2_b},dtype=_t.float32)
        self.ln2.weight.data=_t.tensor({ln2_w},dtype=_t.float32)
        self.ln2.bias.data=_t.tensor({ln2_b},dtype=_t.float32)
        self.ln_f.weight.data=_t.tensor({lnf_w},dtype=_t.float32)
        self.ln_f.bias.data=_t.tensor({lnf_b},dtype=_t.float32)
        self.head.weight.data=_t.tensor({head_w},dtype=_t.float32)
        self.head.bias.data=_t.tensor({head_b},dtype=_t.float32)
    def forward(self,x):
        h=self.tok_emb(x)+self.pos_emb[:x.shape[1]].unsqueeze(0)
        mask=torch.triu(torch.full((x.shape[1],x.shape[1]),float('-inf'),device=x.device),diagonal=1)
        h2,_=self.attn(h,h,h,attn_mask=mask)
        h=self.ln1(h+h2)
        h2=self.ff2(F.gelu(self.ff1(h)))
        h=self.ln2(h+h2)
        h=self.ln_f(h)
        return self.head(h)
_model=None
def build_model():
    global _model
    _model=TinyTransformer()
    _model.eval()
    return _model, {{\"params\": sum(p.numel() for p in _model.parameters()), \"d_model\": D_MODEL, \"nhead\": NHEAD, \"d_ff\": D_FF}}
def add(model,a,b):
    model.eval()
    import torch as _t
    with _t.no_grad():
        inp=[]
        for i in range(14): inp.append((a//(10**i))%10)
        inp.append(10)
        for i in range(14): inp.append((b//(10**i))%10)
        gen=[]
        for _ in range(15):
            seq=_t.tensor([inp+gen],dtype=_t.long)
            logits=model(seq)
            gen.append(int(logits[0,-1].argmax().item()))
        return sum(d*(10**i) for i,d in enumerate(gen))
'''
    open("/workspace/submission.py","w").write(code)
    print("wrote submission.py")
    # test
    import importlib.util
    spec=importlib.util.spec_from_file_location("sub","/workspace/submission.py")
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    m2,meta=mod.build_model()
    print(meta)
    for a,b in [(0,0),(99999999999999,99999999999999),(12345678901234,98765432109876)]:
        print(a,b,mod.add(m2,a,b),a+b,mod.add(m2,a,b)==a+b)
    import random as _r
    ok=sum(1 for _ in range(500) if mod.add(m2,_r.randint(0,MAX_VAL),_r.randint(0,MAX_VAL))==_r.randint(0,MAX_VAL)+_r.randint(0,MAX_VAL))
    # correct random test
    ok=0
    for _ in range(500):
        a=_r.randint(0,MAX_VAL); b=_r.randint(0,MAX_VAL)
        if mod.add(m2,a,b)==a+b: ok+=1
    print(f"random 500 {ok}/500 {ok/500:.4f}")
