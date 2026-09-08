import torch, random, math
import torch.nn as nn
from submission import _sinusoidal_pos, VOCAB_SIZE, MAX_LEN, INPUT_LEN, OUTPUT_LEN, PLUS_TOKEN

DEVICE=torch.device("cpu")
MAX_VAL=99_999_999_999_999
BATCH=512
STEPS=60000
LR=4e-4
WD=0.01
D_MODEL=10
NHEAD=2
D_FF=19

class TiedTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        d_model=D_MODEL; nhead=NHEAD; d_ff=D_FF; vocab=VOCAB_SIZE; max_len=MAX_LEN
        self.tok_emb=nn.Embedding(vocab,d_model)
        pe=_sinusoidal_pos(max_len,d_model)
        self.register_buffer('pos_emb',pe)
        self.attn=nn.MultiheadAttention(d_model,nhead,batch_first=True)
        self.ln1=nn.LayerNorm(d_model)
        self.ffn=nn.Sequential(nn.Linear(d_model,d_ff),nn.GELU(),nn.Linear(d_ff,d_model))
        self.ln2=nn.LayerNorm(d_model)
        self.ln_f=nn.LayerNorm(d_model)
        self.head=nn.Linear(d_model,vocab)
        self.head.weight=self.tok_emb.weight
    def forward(self,x):
        B,T=x.shape
        h=self.tok_emb(x)+self.pos_emb[:T].unsqueeze(0)
        h_res=h; h_norm=self.ln1(h)
        causal=torch.triu(torch.ones(T,T,device=x.device,dtype=torch.bool),diagonal=1)
        attn_out,_=self.attn(h_norm,h_norm,h_norm,attn_mask=causal)
        h=h_res+attn_out
        h_res2=h; h_norm2=self.ln2(h)
        h=h_res2+self.ffn(h_norm2)
        h=self.ln_f(h)
        return self.head(h)

def encode_pair(a,b):
    toks=[]
    for _ in range(14):
        toks.append(a%10); a//=10
    toks.append(PLUS_TOKEN)
    for _ in range(14):
        toks.append(b%10); b//=10
    return toks
def encode_sum(s):
    toks=[]
    for _ in range(15):
        toks.append(s%10); s//=10
    return toks
def sample_batch(bs):
    inp=torch.zeros(bs, INPUT_LEN+OUTPUT_LEN, dtype=torch.long)
    tgt=torch.zeros(bs, INPUT_LEN+OUTPUT_LEN, dtype=torch.long)
    for i in range(bs):
        r=random.random()
        if r<0.35:
            a=random.randint(0,MAX_VAL); b=random.randint(0,MAX_VAL)
        elif r<0.50:
            da=random.randint(1,14); db=random.randint(1,14)
            a=random.randint(10**(da-1) if da>1 else 0, 10**da-1)
            b=random.randint(10**(db-1) if db>1 else 0, 10**db-1)
            if da==1 and random.random()<0.5: a=random.randint(0,9)
            if db==1 and random.random()<0.5: b=random.randint(0,9)
        elif r<0.62:
            a=0; b=0
            for pos in range(14):
                a+=random.randint(5,9)*(10**pos)
                b+=random.randint(5,9)*(10**pos)
            if random.random()<0.5: a=random.randint(0,MAX_VAL)
            if random.random()<0.5: b=random.randint(0,MAX_VAL)
        elif r<0.72:
            if random.random()<0.5: a=0; b=random.randint(0,MAX_VAL)
            else: a=random.randint(0,MAX_VAL); b=0
            if random.random()<0.3: a=0; b=0
        elif r<0.82:
            k=random.randint(0,13); base=10**k
            choices=[base,5*base,10**14-1,10**k-1]
            a=random.choice(choices)%(MAX_VAL+1); b=random.choice(choices)%(MAX_VAL+1)
            if random.random()<0.3:
                a=int("9"*random.randint(1,14)) if random.random()<0.5 else a
                b=int("9"*random.randint(1,14)) if random.random()<0.5 else b
        elif r<0.90:
            n9a=random.randint(1,7); n9b=random.randint(1,7)
            a=random.randint(0,MAX_VAL//(10**n9a))*(10**n9a)+int("9"*n9a)
            b=random.randint(0,MAX_VAL//(10**n9b))*(10**n9b)+int("9"*n9b)
            a=min(a,MAX_VAL); b=min(b,MAX_VAL)
        else:
            a=random.randint(0,10000); b=random.randint(0,10000)
        s=a+b
        full=encode_pair(a,b)+encode_sum(s)
        inp[i]=torch.tensor(full); tgt[i]=torch.tensor(full)
    return inp,tgt

def add_fn(model,a,b):
    model.eval()
    device=next(model.parameters()).device
    def _enc(a,b):
        toks=[]
        for _ in range(14):
            toks.append(a%10); a//=10
        toks.append(PLUS_TOKEN)
        for _ in range(14):
            toks.append(b%10); b//=10
        return toks
    def _dec(toks):
        v=0; m=1
        for t in toks:
            v+=int(t)*m; m*=10
        return v
    inp=_enc(a,b)
    seq=torch.tensor([inp],dtype=torch.long,device=device)
    gen=[]
    with torch.no_grad():
        for _ in range(OUTPUT_LEN):
            logits=model(seq)
            nxt=int(torch.argmax(logits[0,-1,:]).item())
            if nxt>=10: nxt=int(torch.argmax(logits[0,-1,:10]).item())
            gen.append(nxt)
            seq=torch.cat([seq,torch.tensor([[nxt]],dtype=torch.long,device=device)],dim=1)
    return _dec(gen)

model=TiedTransformer().to(DEVICE)
opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=STEPS)
ce=nn.CrossEntropyLoss()
for step in range(1,STEPS+1):
    model.train()
    x,y=sample_batch(BATCH)
    x=x.to(DEVICE); y=y.to(DEVICE)
    logits=model(x[:,:-1])
    targets=y[:,1:]
    mask=torch.zeros_like(targets,dtype=torch.bool); mask[:,28:]=True
    loss=ce(logits[mask],targets[mask])
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sched.step()
    if step%500==0:
        model.eval()
        with torch.no_grad():
            x_e,y_e=sample_batch(512)
            x_e=x_e.to(DEVICE); y_e=y_e.to(DEVICE)
            logits_e=model(x_e[:,:-1])
            preds=logits_e.argmax(-1); targets_e=y_e[:,1:]
            mask_e=torch.zeros_like(targets_e,dtype=torch.bool); mask_e[:,28:]=True
            acc=(preds[mask_e]==targets_e[mask_e]).float().mean().item()
            seq_acc=(preds[mask_e].view(-1,15)==targets_e[mask_e].view(-1,15)).all(dim=1).float().mean().item()
        print(f"step {step} loss {loss.item():.4f} tok {acc:.4f} seq {seq_acc:.4f} lr {sched.get_last_lr()[0]:.6f}")
    if step%5000==0 or step==STEPS:
        model.eval()
        correct=0
        for _ in range(2000):
            a=random.randint(0,MAX_VAL); b=random.randint(0,MAX_VAL)
            if add_fn(model,a,b)==a+b: correct+=1
        print(f" >>> add 2000: {correct}/2000 {correct/2000:.4f}")
        for a,b in [(0,0),(0,MAX_VAL),(MAX_VAL,0),(MAX_VAL,MAX_VAL),(MAX_VAL,1),(12345678901234,98765432109876),(50000000000000,50000000000000)]:
            pred=add_fn(model,a,b)
            print(f"  edge {a}+{b} {'OK' if pred==a+b else f'FAIL {pred} vs {a+b}'}")

print("done")
# save
sd=model.state_dict()
# need to handle tied weight: state_dict will have both tok_emb.weight and head.weight same tensor, but we need to save correctly
# When we reload, we need to tie again before loading
# Generate submission.py with tied architecture
import textwrap
# Build new submission content
with open("/workspace/submission_1161.py","r") as f:
    base=f.read()
# Create tied submission file
tied_sub = base.replace("D_MODEL = 10\nNHEAD = 2\nD_FF = 19", "D_MODEL = 10\nNHEAD = 2\nD_FF = 19")
# Need to inject tying in class
# Replace head line
old_head = "        self.head = nn.Linear(d_model, vocab)"
new_head = "        self.head = nn.Linear(d_model, vocab)\n        self.head.weight = self.tok_emb.weight"
if old_head in tied_sub:
    tied_sub = tied_sub.replace(old_head, new_head)
else:
    print("head replace failed")

# Now need to embed weights
# For tied model, state_dict has duplicate key but same value; we will embed both
weight_lines=[]
weight_lines.append("def _load_weights(model):")
weight_lines.append("    import torch")
weight_lines.append("    sd = {}")
for k,v in sd.items():
    lst=v.cpu().numpy().tolist()
    weight_lines.append(f"    sd['{k}'] = torch.tensor({repr(lst)}, dtype=torch.float32)")
weight_lines.append("    model.load_state_dict(sd)")
weight_lines.append("    return model")
weight_code="\n".join(weight_lines)

old_build="""def build_model():
    model = TinyTransformer()
    metadata = {
        "vocab_size": VOCAB_SIZE,
        "d_model": D_MODEL,
        "nhead": NHEAD,
        "d_ff": D_FF,
        "n_layers": N_LAYERS,
        "max_len": MAX_LEN,
    }
    return model, metadata"""
new_build=weight_code+"\n\ndef build_model():\n    model = TinyTransformer()\n    _load_weights(model)\n    metadata = {\n        \"vocab_size\": VOCAB_SIZE,\n        \"d_model\": D_MODEL,\n        \"nhead\": NHEAD,\n        \"d_ff\": D_FF,\n        \"n_layers\": N_LAYERS,\n        \"max_len\": MAX_LEN,\n    }\n    return model, metadata"

if old_build in tied_sub:
    tied_sub=tied_sub.replace(old_build,new_build)
else:
    tied_sub+= "\n"+weight_code+"\n"

# Need to ensure TinyTransformer class is named correctly - base uses TinyTransformer
# Write to /workspace/submission.py
with open("/workspace/submission.py","w") as f:
    f.write(tied_sub)
print("saved tied submission")
# verify
import importlib, sys
if 'submission' in sys.modules: del sys.modules['submission']
import submission as sub2
import importlib as imp
imp.reload(sub2)
m2,_=sub2.build_model()
print("params",sum(p.numel() for p in m2.parameters()))
for a,b in [(0,0),(0,MAX_VAL),(MAX_VAL,MAX_VAL),(12345678901234,98765432109876)]:
    pred=sub2.add(m2,a,b)
    print(f"verify {a}+{b} {pred} vs {a+b} {'OK' if pred==a+b else 'FAIL'}")
ok=0
for _ in range(500):
    a=random.randint(0,MAX_VAL); b=random.randint(0,MAX_VAL)
    if sub2.add(m2,a,b)==a+b: ok+=1
print(f"random 500 {ok}/500 {ok/500:.4f}")
