import torch, torch.nn as nn, torch.nn.functional as F, random, math, time, os, sys

MAX_VAL = 99_999_999_999_999
VOCAB = 12  # 0-9, '+', '='
SEQ_IN = 29  # 14 +1 +14
SEQ_OUT = 15
SEQ_TOTAL = SEQ_IN + SEQ_OUT  # 44
DEVICE = torch.device("cpu")

def encode_pair(a,b):
    # LSD-first: 14 digits each + '+' + 14 digits, then 15 output digits LSD-first
    # tokens: 0-9 digits, 10='+', 11='='
    inp = []
    for i in range(14):
        inp.append(a % 10)
        a //= 10
    inp.append(10)
    for i in range(14):
        inp.append(b % 10)
        b //= 10
    return inp  # len 29

def encode_sum(s):
    out = []
    for i in range(15):
        out.append(s % 10)
        s //= 10
    return out

def make_batch(bs):
    inp_batch = []
    tgt_batch = []
    for _ in range(bs):
        r = random.random()
        if r < 0.35:
            a = random.randint(0, MAX_VAL)
            b = random.randint(0, MAX_VAL)
        elif r < 0.50:
            # digit-length uniform
            da = random.randint(1,14)
            db = random.randint(1,14)
            a = random.randint(10**(da-1) if da>1 else 0, 10**da -1)
            b = random.randint(10**(db-1) if db>1 else 0, 10**db -1)
            a = min(a, MAX_VAL); b = min(b, MAX_VAL)
        elif r < 0.62:
            # carry-heavy
            a = int(''.join(str(random.randint(5,9)) for _ in range(14))[:14]) % (MAX_VAL+1)
            b = int(''.join(str(random.randint(5,9)) for _ in range(14))[:14]) % (MAX_VAL+1)
        elif r < 0.72:
            # zeros
            if random.random()<0.5:
                a=0; b=random.randint(0,MAX_VAL)
            else:
                a=random.randint(0,MAX_VAL); b=0
        elif r < 0.82:
            # powers of 10
            k = random.randint(0,13)
            a = 10**k
            b = random.randint(0, MAX_VAL)
            if random.random()<0.5:
                a,b=b,a
        elif r < 0.90:
            # trailing 9s
            k = random.randint(1,7)
            a = int('9'*k) if k<=14 else MAX_VAL
            # pad with random prefix
            if random.random()<0.5:
                prefix = random.randint(0, 10**(14-k)-1) if k<14 else 0
                a = prefix * (10**k) + int('9'*k)
                a = min(a, MAX_VAL)
            b = random.randint(0, MAX_VAL)
            if random.random()<0.5:
                a,b=b,a
        else:
            # small
            a = random.randint(0, 9999)
            b = random.randint(0, 9999)
        s = a+b
        inp_batch.append(encode_pair(a,b))
        tgt_batch.append(encode_sum(s))
    return torch.tensor(inp_batch, dtype=torch.long), torch.tensor(tgt_batch, dtype=torch.long)

class Model(nn.Module):
    def __init__(self, d_model=10, nhead=2, d_ff=15, bias=False):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(VOCAB, d_model)
        pe = torch.zeros(SEQ_TOTAL, d_model)
        pos = torch.arange(SEQ_TOTAL, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div[:d_model//2])
        self.register_buffer('pos_emb', pe)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, bias=bias)
        self.ln1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ln3 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ff1 = nn.Linear(d_model, d_ff, bias=bias)
        self.ff2 = nn.Linear(d_ff, d_model, bias=bias)
        self.head = nn.Linear(d_model, VOCAB, bias=False)

    def forward(self, x):
        h = self.tok_emb(x) + self.pos_emb[:x.size(1)]
        seq = x.size(1)
        mask = torch.triu(torch.ones(seq, seq, device=x.device, dtype=torch.bool), diagonal=1)
        h2 = self.ln1(h)
        a,_ = self.attn(h2, h2, h2, attn_mask=mask, need_weights=False)
        h = h + a
        h2 = self.ln2(h)
        h = h + self.ff2(F.gelu(self.ff1(h2)))
        h = self.ln3(h)
        return self.head(h)

def count_params(m):
    return sum(p.numel() for p in m.parameters())

def evaluate(model, n=2000):
    model.eval()
    correct=0
    with torch.no_grad():
        for _ in range(n//512+1):
            bs = min(512, n-correct if False else 512)
            # just do random uniform eval
            pass
    # proper eval
    correct=0
    total=n
    with torch.no_grad():
        done=0
        while done < n:
            bs = min(512, n-done)
            a_vals = [random.randint(0,MAX_VAL) for _ in range(bs)]
            b_vals = [random.randint(0,MAX_VAL) for _ in range(bs)]
            inp = torch.tensor([encode_pair(a,b) for a,b in zip(a_vals,b_vals)], dtype=torch.long)
            tgt = [encode_sum(a+b) for a,b in zip(a_vals,b_vals)]
            # autoregressive decode
            cur = inp
            for pos in range(SEQ_OUT):
                logits = model(cur)  # (B, seq, vocab)
                nxt = logits[:,-1].argmax(-1, keepdim=True)  # (B,1)
                cur = torch.cat([cur, nxt], dim=1)
            pred = cur[:, SEQ_IN:].tolist()
            for p,t in zip(pred,tgt):
                if p==t:
                    correct+=1
            done+=bs
    return correct/n

def train_one(d_model, nhead, d_ff, bias, steps, lr, wd, batch_size, seed):
    random.seed(seed); torch.manual_seed(seed)
    model = Model(d_model, nhead, d_ff, bias=bias)
    print(f"Params: {count_params(model)} d={d_model} h={nhead} ff={d_ff} bias={bias}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    model.train()
    t0=time.time()
    for step in range(1, steps+1):
        inp_tgt, out_tgt = make_batch(batch_size)
        # build full sequence: inp (29) + out shifted
        # input to model: inp + out[:-1] with '=' separator? Actually we use inp + out prefix
        # For training, feed inp + out[0:14] and predict next token
        # Simpler: full seq = inp + out, input = full[:44-1]? Let's do teacher forcing on full 44
        # inp 29 tokens, out 15 tokens. Full = inp + out
        full = torch.cat([inp_tgt, out_tgt], dim=1)  # (B,44)
        x = full[:,:-1]  # (B,43)
        y = full[:,1:]   # but we only care about predicting out part? Actually we need to predict all next tokens
        # However inp part is deterministic given nothing - we should mask loss to only output positions
        # Loss on positions SEQ_IN-1 .. 42 (predicting out tokens)
        logits = model(x)  # (B,43,V)
        # y corresponds to x shifted by 1
        # positions 28..42 correspond to predicting out[0..14]
        # x[28] is last inp token, y[28] is out[0], etc.
        loss = F.cross_entropy(logits[:, SEQ_IN-1:].reshape(-1,VOCAB), y[:, SEQ_IN-1:].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sched.step()
        if step%5000==0 or step==1:
            acc = evaluate(model, n=1000)
            print(f" step {step}/{steps} loss {loss.item():.4f} acc {acc:.4f} lr {sched.get_last_lr()[0]:.2e} time {(time.time()-t0)/60:.1f}m")
            model.train()
            if acc>=0.99 and step>10000:
                # quick extra check
                acc2 = evaluate(model, n=2000)
                print(f"  extra eval 2000: {acc2:.4f}")
    acc = evaluate(model, n=5000)
    print(f"FINAL acc 5000: {acc:.4f}")
    return model, acc

if __name__ == "__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--d_model", type=int, default=10)
    ap.add_argument("--d_ff", type=int, default=15)
    ap.add_argument("--nhead", type=int, default=2)
    ap.add_argument("--bias", type=int, default=0)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args=ap.parse_args()
    m,acc = train_one(args.d_model, args.nhead, args.d_ff, bool(args.bias), args.steps, args.lr, args.wd, args.bs, args.seed)
    # save checkpoint
    torch.save(m.state_dict(), f"/workspace/ckpt_d{m.d_model}_ff{m.ff1.out_features}_bias{int(args.bias)}.pt")
    print(f"saved ckpt acc {acc:.4f}")
