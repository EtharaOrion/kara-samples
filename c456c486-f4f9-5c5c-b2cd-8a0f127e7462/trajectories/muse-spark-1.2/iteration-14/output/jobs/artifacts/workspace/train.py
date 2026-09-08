import torch, torch.nn as nn, torch.nn.functional as F, math, random
MAX_VAL=99_999_999_999_999
VOCAB=12; PLUS=10; SEQ_IN=29; SEQ_TOTAL=44
def encode_pair(a,b):
    return [(a//10**i)%10 for i in range(14)]+[PLUS]+[(b//10**i)%10 for i in range(14)], [(a+b)//10**i%10 for i in range(15)]
def make_batch(bs):
    inp_list=[]; tgt_list=[]
    for _ in range(bs):
        r=random.random()
        if r<0.35: a=random.randint(0,MAX_VAL); b=random.randint(0,MAX_VAL)
        elif r<0.50:
            la=random.randint(1,14); lb=random.randint(1,14)
            a=random.randint(10**(la-1) if la>1 else 0, 10**la-1)
            b=random.randint(10**(lb-1) if lb>1 else 0, 10**lb-1)
            a=min(a,MAX_VAL); b=min(b,MAX_VAL)
        elif r<0.62:
            def ch():
                v=0
                for i in range(14):
                    d=random.randint(5,9) if random.random()<0.7 else random.randint(0,9)
                    v+=d*10**i
                return min(v,MAX_VAL)
            a=ch(); b=ch()
        elif r<0.72:
            a=random.randint(0,MAX_VAL); b=random.randint(0,MAX_VAL)
            if random.random()<0.5: a=0
            if random.random()<0.5: b=0
        elif r<0.82:
            k=random.randint(0,13); a=10**k
            b=(10**random.randint(0,13)) if random.random()<0.5 else random.randint(0,MAX_VAL)
            if random.random()<0.3: a=5*10**k
        elif r<0.90:
            n9=random.randint(1,10)
            base=int('9'*n9)
            a=base+random.randint(0,10**(14-n9)-1) if n9<14 else base
            a=min(a,MAX_VAL); b=random.randint(0,MAX_VAL)
            if random.random()<0.5: a,b=b,a
        else: a=random.randint(0,9999); b=random.randint(0,9999)
        inp,ds=encode_pair(a,b); inp_list.append(inp); tgt_list.append(ds)
    return torch.tensor(inp_list,dtype=torch.long), torch.tensor(tgt_list,dtype=torch.long)

class Model(nn.Module):
    def __init__(self,d_model=8,nhead=2,d_ff=12,n_layers=2):
        super().__init__()
        self.tok_emb=nn.Embedding(VOCAB,d_model)
        pe=torch.zeros(SEQ_TOTAL,d_model)
        pos=torch.arange(0,SEQ_TOTAL,dtype=torch.float).unsqueeze(1)
        div=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(pos*div)
        pe[:,1::2]=torch.cos(pos*div[:d_model//2])
        self.register_buffer('pos_emb',pe)
        self.layers=nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({'attn':nn.MultiheadAttention(d_model,nhead,batch_first=True),'ln1':nn.LayerNorm(d_model),'ln2':nn.LayerNorm(d_model),'ff1':nn.Linear(d_model,d_ff),'ff2':nn.Linear(d_ff,d_model)}))
        self.ln_f=nn.LayerNorm(d_model); self.head=nn.Linear(d_model,VOCAB)
    def forward(self,x):
        h=self.tok_emb(x)+self.pos_emb[:x.shape[1]].unsqueeze(0)
        causal=torch.triu(torch.full((x.shape[1],x.shape[1]),float('-inf'),device=x.device),diagonal=1)
        for lyr in self.layers:
            h2=lyr['ln1'](h); a,_=lyr['attn'](h2,h2,h2,attn_mask=causal); h=h+a
            h2=lyr['ln2'](h); h=h+lyr['ff2'](F.gelu(lyr['ff1'](h2)))
        return self.head(self.ln_f(h))

def eval_acc(model,n=500,device='cpu'):
    model.eval(); c=0
    with torch.no_grad():
        for _ in range(n):
            a=random.randint(0,MAX_VAL); b=random.randint(0,MAX_VAL)
            inp,_=encode_pair(a,b); seq=torch.tensor([inp],dtype=torch.long,device=device)
            for _ in range(15):
                seq=torch.cat([seq,torch.tensor([[model(seq)[0,-1].argmax().item()]],device=device)],dim=1)
            if sum(d*10**i for i,d in enumerate(seq[0,SEQ_IN:].tolist()))==a+b: c+=1
    model.train(); return c/n

if __name__=="__main__":
    device='cuda' if torch.cuda.is_available() else 'cpu'
    cfg={'d_model':8,'nhead':2,'d_ff':12,'n_layers':2}
    random.seed(2); torch.manual_seed(2)
    m=Model(**cfg).to(device)
    print(f"Params {sum(p.numel() for p in m.parameters())}")
    opt=torch.optim.AdamW(m.parameters(),lr=8e-4,weight_decay=0.01)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=80000)
    best=0; best_state=None
    for step in range(1,80001):
        inp,tgt=make_batch(512)
        inp=inp.to(device); tgt=tgt.to(device)
        x_full=torch.cat([inp,tgt],dim=1); x=x_full[:,:-1]; y=x_full[:,1:]
        logits=m(x)
        loss=F.cross_entropy(logits[:,28:].reshape(-1,VOCAB), y[:,28:].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); sched.step()
        if step%4000==0:
            acc=eval_acc(m,n=400,device=device)
            print(f"step {step} loss {loss.item():.4f} acc {acc:.4f}")
            if acc>best: best=acc; best_state={k:v.cpu().clone() for k,v in m.state_dict().items()}
            if acc>=0.99:
                acc2=eval_acc(m,n=1200,device=device)
                print(f" verify {acc2:.4f}")
                if acc2>=0.99: break
    if best_state: torch.save(best_state,"/workspace/best.pt"); print(f"saved best {best:.4f}")
