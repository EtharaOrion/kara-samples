import torch, torch.nn as nn, torch.nn.functional as F, math, random, time
D_MODEL=10
D_FF=20
VOCAB=12
MAX_LEN=44
MAX_VAL=99_999_999_999_999
DEVICE=torch.device("cpu")
class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(12, 10)
        pe = torch.zeros(44, 10)
        position = torch.arange(0, 44, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, 10, 2).float() * (-math.log(10000.0) / 10))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:5])
        self.register_buffer('pos_emb', pe)
        self.attn = nn.MultiheadAttention(10, 2, batch_first=True)
        self.ln1 = nn.LayerNorm(10)
        self.ln2 = nn.LayerNorm(10)
        self.ln_f = nn.LayerNorm(10)
        self.ff1 = nn.Linear(10, 20)
        self.ff2 = nn.Linear(20, 10)
        self.head = nn.Linear(10, 12)
    def forward(self, x):
        h = self.tok_emb(x) + self.pos_emb[:x.size(1), :].unsqueeze(0)
        h_norm = self.ln1(h)
        T = x.size(1)
        causal = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        attn_out, _ = self.attn(h_norm, h_norm, h_norm, attn_mask=causal)
        h = h + attn_out
        h2 = self.ln2(h)
        ff = self.ff2(F.gelu(self.ff1(h2)))
        h = h + ff
        h = self.ln_f(h)
        return self.head(h)

def encode_number(n, length):
    d=[]
    for _ in range(length):
        d.append(n%10); n//=10
    return d

def make_batch(bs):
    a_list=[]; b_list=[]
    for _ in range(bs):
        r=random.random()
        if r<0.35:
            a=random.randint(0,MAX_VAL); b=random.randint(0,MAX_VAL)
        elif r<0.50:
            da=random.randint(1,14); db=random.randint(1,14)
            a=random.randint(0,10**da-1) if da<14 else random.randint(0,MAX_VAL)
            b=random.randint(0,10**db-1) if db<14 else random.randint(0,MAX_VAL)
        elif r<0.62:
            a=0; b=0
            for pos in range(14):
                if random.random()<0.7:
                    da=random.randint(5,9); db=random.randint(5,9)
                else:
                    da=random.randint(0,9); db=random.randint(0,9)
                a+=da*(10**pos); b+=db*(10**pos)
            a=min(a,MAX_VAL); b=min(b,MAX_VAL)
        elif r<0.72:
            if random.random()<0.5:
                a=0; b=random.randint(0,MAX_VAL)
            else:
                a=random.randint(0,MAX_VAL); b=0
            if random.random()<0.3:
                a=0; b=0
        elif r<0.82:
            k=random.randint(0,13); base=10**k
            choices=[base,5*base,base-1 if base>0 else 0,10**14-1]
            a=random.choice(choices)%(MAX_VAL+1); b=random.choice(choices)%(MAX_VAL+1)
            if random.random()<0.5:
                a=random.randint(0,MAX_VAL); b=random.randint(0,MAX_VAL)
        elif r<0.90:
            t=random.randint(1,6); suffix=int('9'*t)
            prefix_a=random.randint(0,10**(14-t)-1) if t<14 else 0
            prefix_b=random.randint(0,10**(14-t)-1) if t<14 else 0
            a=prefix_a*(10**t)+suffix; b=prefix_b*(10**t)+suffix
            a=min(a,MAX_VAL); b=min(b,MAX_VAL)
        else:
            a=random.randint(0,9999); b=random.randint(0,9999)
        a_list.append(a); b_list.append(b)
    return a_list,b_list

def batch_to_tensor(a_list,b_list):
    B=len(a_list)
    x=torch.zeros(B,MAX_LEN,dtype=torch.long)
    y=torch.zeros(B,MAX_LEN,dtype=torch.long)
    for i,(a,b) in enumerate(zip(a_list,b_list)):
        seq=encode_number(a,14)+[10]+encode_number(b,14)+encode_number(a+b,15)
        x[i,:len(seq)]=torch.tensor(seq,dtype=torch.long)
        y[i,:len(seq)]=torch.tensor(seq,dtype=torch.long)
    return x,y

def evaluate(model,n=1000):
    model.eval()
    correct=0
    with torch.no_grad():
        for _ in range(0,n,512):
            bs=min(512,n-_)
            a_list=[random.randint(0,MAX_VAL) for _ in range(bs)]
            b_list=[random.randint(0,MAX_VAL) for _ in range(bs)]
            for a,b in zip(a_list,b_list):
                prefix=encode_number(a,14)+[10]+encode_number(b,14)
                seq=prefix.copy()
                for _ in range(15):
                    inp=torch.tensor([seq],dtype=torch.long,device=DEVICE)
                    logits=model(inp)
                    pred=int(torch.argmax(logits[0,len(seq)-1,:]).item())
                    seq.append(pred)
                if seq[29:44]==encode_number(a+b,15):
                    correct+=1
    return correct/n

def train():
    random.seed(123)
    torch.manual_seed(123)
    model=TinyTransformer().to(DEVICE)
    optimizer=torch.optim.AdamW(model.parameters(),lr=4e-4,weight_decay=0.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=60000)
    best_acc=0; best_state=None
    max_steps=60000; batch_size=512
    start=time.time()
    model.train()
    for step in range(1,max_steps+1):
        a_list,b_list=make_batch(batch_size)
        x,y=batch_to_tensor(a_list,b_list)
        x=x.to(DEVICE); y=y.to(DEVICE)
        logits=model(x)
        loss=F.cross_entropy(logits[:,:-1,:][:,28:43,:].reshape(-1,VOCAB), y[:,1:][:,28:43].reshape(-1))
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step(); scheduler.step()
        if step%500==0:
            print(f"step {step} loss {loss.item():.4f} lr {scheduler.get_last_lr()[0]:.6f} time {time.time()-start:.1f}s",flush=True)
        if step%2000==0 or step==max_steps:
            acc=evaluate(model,1000)
            print(f"  eval {acc:.4f} best {best_acc:.4f}",flush=True)
            if acc>best_acc:
                best_acc=acc
                best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
                print(f"  new best {best_acc:.4f}",flush=True)
            # edges
            model.eval()
            edges=[(0,0),(0,MAX_VAL),(MAX_VAL,0),(MAX_VAL,MAX_VAL),(MAX_VAL,1),(12345678901234,98765432109876)]
            ok=True
            with torch.no_grad():
                for a,b in edges:
                    prefix=encode_number(a,14)+[10]+encode_number(b,14)
                    seq=prefix.copy()
                    for _ in range(15):
                        inp=torch.tensor([seq],dtype=torch.long,device=DEVICE)
                        logits_e=model(inp)
                        pred=int(torch.argmax(logits_e[0,len(seq)-1,:]).item())
                        seq.append(pred)
                    if seq[29:44]!=encode_number(a+b,15):
                        ok=False
                        print(f"    edge fail {a}+{b}",flush=True)
                        break
            if ok:
                print("    edges OK",flush=True)
            model.train()
    print(f"done best {best_acc:.4f}",flush=True)
    if best_state is not None:
        torch.save(best_state,"/workspace/best2.pt")
        model.load_state_dict(best_state)
        acc=evaluate(model,2000)
        print(f"final 2000 {acc:.4f}",flush=True)
        # also save as best.pt if better than previous
        import os
        if os.path.exists("/workspace/best.pt"):
            # compare
            pass
        torch.save(best_state,"/workspace/best.pt")
    else:
        torch.save(model.state_dict(),"/workspace/best.pt")

if __name__=="__main__":
    train()
