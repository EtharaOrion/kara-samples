import torch, torch.nn as nn, torch.nn.functional as F, random, os, sys

# Config from args
D_MODEL = int(sys.argv[1]) if len(sys.argv)>1 else 32
N_HEADS = int(sys.argv[2]) if len(sys.argv)>2 else 2
N_LAYERS = int(sys.argv[3]) if len(sys.argv)>3 else 2
D_FF = int(sys.argv[4]) if len(sys.argv)>4 else 64
MAX_STEPS = int(sys.argv[5]) if len(sys.argv)>5 else 60000

VOCAB=12; SEQ_LEN=45; DEVICE="cuda" if torch.cuda.is_available() else "cpu"
print(f"Config: d={D_MODEL} h={N_HEADS} L={N_LAYERS} ff={D_FF} steps={MAX_STEPS} device={DEVICE}")

def encode_pair(a,b):
    inp=[]
    for i in range(14):
        inp.append(a%10); a//=10
    inp.append(10)
    for i in range(14):
        inp.append(b%10); b//=10
    inp.append(11)
    return inp
def encode_sum(s):
    return [(s//(10**i))%10 for i in range(15)]
def decode_sum(t): return sum(t[i]*(10**i) for i in range(len(t)))

class TransformerAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb=nn.Embedding(VOCAB,D_MODEL)
        self.pos_emb=nn.Embedding(SEQ_LEN,D_MODEL)
        self.layers=nn.ModuleList()
        for _ in range(N_LAYERS):
            self.layers.append(nn.ModuleDict({
                'attn': nn.MultiheadAttention(D_MODEL,N_HEADS,batch_first=True),
                'ln1': nn.LayerNorm(D_MODEL),
                'ln2': nn.LayerNorm(D_MODEL),
                'ff1': nn.Linear(D_MODEL,D_FF),
                'ff2': nn.Linear(D_FF,D_MODEL),
            }))
        self.ln_f=nn.LayerNorm(D_MODEL)
        self.head=nn.Linear(D_MODEL,VOCAB)
    def forward(self,x):
        B,T=x.shape
        h=self.token_emb(x)+self.pos_emb(torch.arange(T,device=x.device))
        causal=torch.triu(torch.ones(T,T,device=x.device,dtype=torch.bool),diagonal=1)
        for layer in self.layers:
            h_norm=layer['ln1'](h)
            attn_out,_=layer['attn'](h_norm,h_norm,h_norm,attn_mask=causal,need_weights=False)
            h=h+attn_out
            h_norm2=layer['ln2'](h)
            ff=layer['ff2'](F.gelu(layer['ff1'](h_norm2)))
            h=h+ff
        return self.head(self.ln_f(h))

def gen_batch(bs):
    seqs=[]
    for _ in range(bs):
        r=random.random()
        if r<0.5: a=random.randint(0,99999999999999); b=random.randint(0,99999999999999)
        elif r<0.7: a=random.randint(0,10**random.randint(1,14)-1); b=random.randint(0,10**random.randint(1,14)-1)
        elif r<0.85:
            a=sum(random.randint(5,9)*(10**i) for i in range(14))
            b=sum(random.randint(5,9)*(10**i) for i in range(14))
            if random.random()<0.5: a=random.randint(0,99999999999999)
        else: a=random.randint(0,999999); b=random.randint(0,999999)
        inp=encode_pair(a,b); out=encode_sum(a+b)
        seqs.append(inp+out)
    return torch.tensor(seqs,dtype=torch.long,device=DEVICE)

def evaluate(model, n=500):
    model.eval()
    correct=0
    with torch.no_grad():
        for _ in range(n):
            a=random.randint(0,99999999999999); b=random.randint(0,99999999999999)
            inp=encode_pair(a,b)
            seq=torch.tensor([inp],dtype=torch.long,device=DEVICE)
            for _ in range(15):
                logits=model(seq)
                nxt=logits[0,-1].argmax().item()
                seq=torch.cat([seq,torch.tensor([[nxt]],device=DEVICE)],dim=1)
            pred=sum(seq[0,30+i].item()*(10**i) for i in range(15))
            if pred==a+b: correct+=1
    model.train()
    return correct/n

model=TransformerAdd().to(DEVICE)
print(f"params: {sum(p.numel() for p in model.parameters())}")
opt=torch.optim.AdamW(model.parameters(),lr=4e-4,weight_decay=0.01)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=MAX_STEPS)

best=0
for step in range(1,MAX_STEPS+1):
    x=gen_batch(512)
    inp=x[:,:-1]; tgt=x[:,1:]
    logits=model(inp)
    loss=F.cross_entropy(logits[:,29:44].reshape(-1,VOCAB),tgt[:,29:44].reshape(-1))
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sched.step()
    if step%500==0: print(f"step {step} loss {loss.item():.4f}")
    if step%2000==0:
        acc=evaluate(model,500)
        print(f"  eval {acc:.4f} best {best:.4f}")
        if acc>best:
            best=acc
            torch.save(model.state_dict(),f"/workspace/best_{D_MODEL}_{N_HEADS}_{N_LAYERS}_{D_FF}.pt")
            print(f"  saved best {best:.4f}")
        if acc>=0.99:
            print(f"  *** 99% reached at {step}")

# final eval 2000
ckpt=f"/workspace/best_{D_MODEL}_{N_HEADS}_{N_LAYERS}_{D_FF}.pt"
if os.path.exists(ckpt):
    model.load_state_dict(torch.load(ckpt,map_location=DEVICE))
acc=evaluate(model,2000)
print(f"FINAL {acc:.4f} best {best:.4f} params {sum(p.numel() for p in model.parameters())}")
