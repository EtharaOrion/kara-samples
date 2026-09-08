import torch, torch.nn as nn, torch.nn.functional as F, math, random, sys
D_MODEL=12
NHEAD=2
D_FF=11
VOCAB=12
SEQ_TOTAL=44
MAX_VAL=99_999_999_999_999
TOK_PLUS=10
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE {DEVICE} D_MODEL={D_MODEL} NHEAD={NHEAD} D_FF={D_FF} TIED")

def make_sinusoidal(n_pos, d_model):
    pe=torch.zeros(n_pos,d_model)
    pos=torch.arange(0,n_pos,dtype=torch.float).unsqueeze(1)
    div=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
    pe[:,0::2]=torch.sin(pos*div)
    pe[:,1::2]=torch.cos(pos*div[:d_model//2])
    return pe

def encode_pair_correct(a,b):
    ta=[(a//(10**i))%10 for i in range(14)]
    tb=[(b//(10**i))%10 for i in range(14)]
    inp=ta+[TOK_PLUS]+tb
    s=a+b
    tout=[(s//(10**i))%10 for i in range(15)]
    return inp,tout

def make_batch(bs):
    inp_batch=torch.zeros(bs,29,dtype=torch.long)
    tgt_batch=torch.zeros(bs,15,dtype=torch.long)
    for i in range(bs):
        r=random.random()
        if r<0.35:
            a=random.randint(0,MAX_VAL); b=random.randint(0,MAX_VAL)
        elif r<0.50:
            da=random.randint(1,14); db=random.randint(1,14)
            a=random.randint(10**(da-1) if da>1 else 0, 10**da-1)
            if a>MAX_VAL: a=random.randint(0,MAX_VAL)
            b=random.randint(10**(db-1) if db>1 else 0, 10**db-1)
            if b>MAX_VAL: b=random.randint(0,MAX_VAL)
        elif r<0.62:
            a=0; b=0
            for k in range(14):
                a+=random.randint(5,9)*(10**k)
                b+=random.randint(5,9)*(10**k)
            a%=(MAX_VAL+1); b%=(MAX_VAL+1)
        elif r<0.72:
            if random.random()<0.5: a=0; b=random.randint(0,MAX_VAL)
            else: a=random.randint(0,MAX_VAL); b=0
        elif r<0.82:
            k=random.randint(0,13); a=10**k; b=random.randint(0,MAX_VAL)
            if random.random()<0.5: a,b=b,a
        elif r<0.90:
            k=random.randint(1,6); base=random.randint(0,MAX_VAL//(10**k))
            a=base*(10**k)+(10**k-1); b=random.randint(0,MAX_VAL)
            if a>MAX_VAL: a=MAX_VAL
        else:
            a=random.randint(0,10000); b=random.randint(0,10000)
        inp,tout=encode_pair_correct(a,b)
        inp_batch[i]=torch.tensor(inp,dtype=torch.long)
        tgt_batch[i]=torch.tensor(tout,dtype=torch.long)
    return inp_batch,tgt_batch

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb=nn.Embedding(VOCAB,D_MODEL)
        pe=make_sinusoidal(SEQ_TOTAL,D_MODEL)
        self.register_buffer('pos_emb',pe)
        self.layers=nn.ModuleList([nn.ModuleDict({'attn':nn.MultiheadAttention(D_MODEL,NHEAD,batch_first=True),'ln1':nn.LayerNorm(D_MODEL),'ln2':nn.LayerNorm(D_MODEL),'ff1':nn.Linear(D_MODEL,D_FF),'ff2':nn.Linear(D_FF,D_MODEL)})])
        self.ln_f=nn.LayerNorm(D_MODEL)
        self.head=nn.Linear(D_MODEL,VOCAB,bias=False)
        # tie weights
        self.head.weight=self.tok_emb.weight
    def forward(self,x):
        B,S=x.shape
        h=self.tok_emb(x)+self.pos_emb[:S].unsqueeze(0)
        mask=torch.triu(torch.ones(S,S,device=x.device,dtype=torch.bool),diagonal=1)
        am=torch.zeros(S,S,device=x.device)
        am.masked_fill_(mask,float('-inf'))
        for layer in self.layers:
            h2=layer['ln1'](h)
            a,_=layer['attn'](h2,h2,h2,attn_mask=am)
            h=h+a
            h2=layer['ln2'](h)
            h=h+layer['ff2'](F.gelu(layer['ff1'](h2)))
        h=self.ln_f(h)
        return self.head(h)

def count_params(m): return sum(p.numel() for p in m.parameters())
def eval_model(model,n=1000):
    model.eval()
    correct=0
    with torch.no_grad():
        for _ in range(n):
            a=random.randint(0,MAX_VAL); b=random.randint(0,MAX_VAL)
            inp,_=encode_pair_correct(a,b)
            s=a+b
            target=[(s//(10**i))%10 for i in range(15)]
            seq=torch.tensor([inp],dtype=torch.long,device=DEVICE)
            gen=[]
            for _ in range(15):
                logits=model(seq)
                nxt=int(logits[0,-1].argmax().item())
                gen.append(nxt)
                seq=torch.cat([seq,torch.tensor([[nxt]],device=DEVICE)],dim=1)
            if gen==target: correct+=1
    return correct/n

if len(sys.argv)>1: D_MODEL=int(sys.argv[1])
if len(sys.argv)>2: D_FF=int(sys.argv[2])
# rebuild with new D_MODEL/D_FF if overridden
# need to recreate model
model=Model()
# if D_MODEL/D_FF overridden, need to adjust - but for now use defaults
# Actually support override by recreating
if len(sys.argv)>1 or len(sys.argv)>2:
    # hack: recreate with new dims
    import importlib
    # just set globals and recreate
    pass

# For now, use the Model as defined with D_MODEL=12 D_FF=11
# If CLI wants different, we need to handle
if len(sys.argv)>1:
    # recreate with CLI dims
    D_MODEL=int(sys.argv[1])
    D_FF=int(sys.argv[2]) if len(sys.argv)>2 else D_FF
    class Model2(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb=nn.Embedding(VOCAB,D_MODEL)
            pe=make_sinusoidal(SEQ_TOTAL,D_MODEL)
            self.register_buffer('pos_emb',pe)
            self.layers=nn.ModuleList([nn.ModuleDict({'attn':nn.MultiheadAttention(D_MODEL,NHEAD,batch_first=True),'ln1':nn.LayerNorm(D_MODEL),'ln2':nn.LayerNorm(D_MODEL),'ff1':nn.Linear(D_MODEL,D_FF),'ff2':nn.Linear(D_FF,D_MODEL)})])
            self.ln_f=nn.LayerNorm(D_MODEL)
            self.head=nn.Linear(D_MODEL,VOCAB,bias=False)
            self.head.weight=self.tok_emb.weight
        def forward(self,x):
            B,S=x.shape
            h=self.tok_emb(x)+self.pos_emb[:S].unsqueeze(0)
            mask=torch.triu(torch.ones(S,S,device=x.device,dtype=torch.bool),diagonal=1)
            am=torch.zeros(S,S,device=x.device)
            am.masked_fill_(mask,float('-inf'))
            for layer in self.layers:
                h2=layer['ln1'](h)
                a,_=layer['attn'](h2,h2,h2,attn_mask=am)
                h=h+a
                h2=layer['ln2'](h)
                h=h+layer['ff2'](F.gelu(layer['ff1'](h2)))
            h=self.ln_f(h)
            return self.head(h)
    model=Model2()

print(f"params {count_params(model)}")
opt=torch.optim.AdamW(model.parameters(),lr=4e-4,weight_decay=0.01)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=60000)
model=model.to(DEVICE)
best=0
best_state=None
for step in range(1,60001):
    model.train()
    inp,tgt=make_batch(512)
    full=torch.cat([inp,tgt],dim=1).to(DEVICE)
    x=full[:,:-1]; y=full[:,1:]
    logits=model(x)
    loss=F.cross_entropy(logits[:,28:].reshape(-1,VOCAB), y[:,28:].reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    opt.step(); opt.zero_grad(); sched.step()
    if step%500==0:
        print(f"step {step} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.6f}")
    if step%2000==0 or step==1000:
        acc=eval_model(model,n=1000)
        print(f"  eval {acc*100:.2f}%")
        if acc>best:
            best=acc
            best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
            print(f"  NEW BEST {best*100:.2f}%")
            torch.save(best_state,f"/workspace/best_tied_D{D_MODEL}_FF{D_FF}.pt")
print(f"BEST {best*100:.2f}%")
if best_state is not None:
    torch.save(best_state,f"/workspace/best_tied_D{D_MODEL}_FF{D_FF}.pt")
    model.load_state_dict(best_state)
    acc=eval_model(model,n=2000)
    print(f"FINAL 2000 {acc*100:.2f}%")
