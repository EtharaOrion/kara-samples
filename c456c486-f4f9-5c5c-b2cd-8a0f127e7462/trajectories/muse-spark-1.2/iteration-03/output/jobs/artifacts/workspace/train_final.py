import torch, torch.nn as nn, torch.nn.functional as F, random
VOCAB=12; SEQ_TOTAL=45; SEQ_INPUT=30; SEQ_OUTPUT=15; MAX_VAL=99_999_999_999_999
def enc_num(n,l=14): return [int(c) for c in reversed(str(n).zfill(l))]
def enc_in(a,b): return enc_num(a)+[10]+enc_num(b)+[11]
def enc_out(s): return enc_num(s,15)
def dec_out(t): return int(''.join(str(x) for x in reversed(t)).lstrip('0') or '0')
class TinyTransformer(nn.Module):
    def __init__(self, d_model, nhead, n_layers, d_ff):
        super().__init__()
        self.d_model=d_model; self.vocab=12; self.seq_len=45
        self.token_emb=nn.Embedding(12,d_model); self.pos_emb=nn.Embedding(45,d_model)
        self.layers=nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({'ln1':nn.LayerNorm(d_model),'attn':nn.MultiheadAttention(d_model,nhead,batch_first=True),'ln2':nn.LayerNorm(d_model),'ff1':nn.Linear(d_model,d_ff),'ff2':nn.Linear(d_ff,d_model)}))
        self.ln_f=nn.LayerNorm(d_model); self.head=nn.Linear(d_model,12)
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

def do_train(d,h,l,ff,seed,steps):
    random.seed(seed); torch.manual_seed(seed)
    m=TinyTransformer(d,h,l,ff)
    print(f"Training d={d} h={h} L={l} ff={ff} seed={seed} steps={steps} params={sum(p.numel() for p in m.parameters())}")
    opt=torch.optim.AdamW(m.parameters(),lr=0.001,weight_decay=0.01)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=steps)
    for step in range(1,steps+1):
        inp=make_batch(512)
        logits=m(inp[:,:-1])
        loss=F.cross_entropy(logits[:,29:].reshape(-1,12), inp[:,1:][:,29:].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); sched.step()
        if step%10000==0:
            print(f" step {step} loss {loss.item():.5f}")
    acc=eval_acc(m,5000)
    print(f" FINAL acc {acc:.4f} {'PASS' if acc>=0.99 else 'FAIL'}")
    return m, acc

# Try d=12,2,2,16 seed 0 - known good
m,acc = do_train(12,2,2,16,0,40000)
if acc>=0.99:
    print("SAVING 12/2/2/16")
    D_MODEL,NHEAD,N_LAYERS,D_FF=12,2,2,16
    sd=m.state_dict()
    lines=["import torch","import torch.nn as nn","import torch.nn.functional as F",f"D_MODEL={D_MODEL}",f"NHEAD={NHEAD}",f"N_LAYERS={N_LAYERS}",f"D_FF={D_FF}","VOCAB=12; SEQ_INPUT=30; SEQ_OUTPUT=15; SEQ_TOTAL=45","class TinyTransformer(nn.Module):","    def __init__(self):","        super().__init__()","        d_model=D_MODEL; nhead=NHEAD; n_layers=N_LAYERS; d_ff=D_FF; vocab=VOCAB; seq_len=SEQ_TOTAL","        self.token_emb=nn.Embedding(vocab,d_model)","        self.pos_emb=nn.Embedding(seq_len,d_model)","        self.layers=nn.ModuleList()","        for _ in range(n_layers):","            self.layers.append(nn.ModuleDict({'ln1':nn.LayerNorm(d_model),'attn':nn.MultiheadAttention(d_model,nhead,batch_first=True),'ln2':nn.LayerNorm(d_model),'ff1':nn.Linear(d_model,d_ff),'ff2':nn.Linear(d_ff,d_model)}))","        self.ln_f=nn.LayerNorm(d_model)","        self.head=nn.Linear(d_model,vocab)","    def forward(self,x):","        B,T=x.shape; pos=torch.arange(T,device=x.device)","        h=self.token_emb(x)+self.pos_emb(pos)","        mask=torch.triu(torch.ones(T,T,device=x.device,dtype=torch.bool),diagonal=1)","        for layer in self.layers:","            h_norm=layer['ln1'](h)","            attn_out,_=layer['attn'](h_norm,h_norm,h_norm,attn_mask=mask)","            h=h+attn_out","            h_norm2=layer['ln2'](h)","            ff=layer['ff2'](F.gelu(layer['ff1'](h_norm2)))","            h=h+ff","        h=self.ln_f(h)","        return self.head(h)","_SD={}"]
    for k,v in sd.items():
        lines.append(f"_SD['{k}']=torch.tensor({v.cpu().float().flatten().tolist()},dtype=torch.float32).reshape({list(v.shape)})")
    lines+=["_MODEL=None","def build_model():","    global _MODEL","    m=TinyTransformer()","    m.load_state_dict(_SD, strict=True)","    m.eval()","    meta={'params': sum(p.numel() for p in m.parameters()), 'd_model': D_MODEL, 'nhead': NHEAD, 'n_layers': N_LAYERS, 'd_ff': D_FF}","    _MODEL=m","    return m, meta","def _encode_input(a,b):","    s=str(a).zfill(14); ad=[int(c) for c in reversed(s)]","    s=str(b).zfill(14); bd=[int(c) for c in reversed(s)]","    return ad+[10]+bd+[11]","def _decode_output(tokens):","    s=''.join(str(t) for t in reversed(tokens))","    return int(s.lstrip('0') or '0')","def add(model,a,b):","    model.eval()","    with torch.no_grad():","        enc=_encode_input(a,b)","        toks=enc[:]","        for _ in range(SEQ_OUTPUT):","            x=torch.tensor([toks],dtype=torch.long)","            logits=model(x)","            nxt=int(logits[0,-1].argmax().item())","            toks.append(nxt)","        out=toks[SEQ_INPUT:SEQ_INPUT+SEQ_OUTPUT]","        return _decode_output(out)"]
    open("/workspace/submission.py","w").write("\n".join(lines))
    print("Saved to /workspace/submission.py")
    # verify
    import importlib, submission
    importlib.reload(submission)
    m2,meta=submission.build_model()
    print(meta)
    for a,b in [(0,0),(0,MAX_VAL),(MAX_VAL,MAX_VAL),(99999999999999,1)]:
        print(a,b,submission.add(m2,a,b), "OK" if submission.add(m2,a,b)==a+b else "FAIL")
    c=0
    for _ in range(5000):
        a=random.randint(0,MAX_VAL); b=random.randint(0,MAX_VAL)
        if submission.add(m2,a,b)==a+b: c+=1
    print(f"verify 5000: {c/5000:.4f}")
else:
    print("FAILED - trying seed 42")
    m,acc = do_train(12,2,2,16,42,40000)
    print(f"seed42 acc {acc}")
