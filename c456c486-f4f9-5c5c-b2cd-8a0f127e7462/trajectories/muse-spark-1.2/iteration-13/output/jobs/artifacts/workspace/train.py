import torch
import torch.nn as nn
import torch.nn.functional as F
import math, random, os, sys

# Config - will sweep
D_MODEL = 12
NHEAD = 2
D_FF = 20
N_LAYERS = 1
MAX_VAL = 99_999_999_999_999
VOCAB = 12  # 0-9, '+', '='
SEQ_IN = 29  # 14 +1 +14
SEQ_OUT = 15
SEQ_TOTAL = 44
PAD = 10  # not used, but vocab includes + and =
TOK_PLUS = 10
TOK_EQ = 11

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE {DEVICE} D_MODEL={D_MODEL} NHEAD={NHEAD} D_FF={D_FF} N_LAYERS={N_LAYERS}")

def make_sinusoidal(n_pos, d_model):
    pe = torch.zeros(n_pos, d_model)
    pos = torch.arange(0, n_pos, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[:d_model//2])
    return pe

def encode_pair(a,b):
    # LSD-first 14 digits each
    toks = []
    for i in range(14):
        toks.append(a % 10)
        a //= 10
    # Actually need 14 tokens for a LSD first
    # Let's do correctly: we already did 14? No loop above does 14 but we consumed a
    # Re-encode properly
    pass

def encode_pair_correct(a,b):
    ta = [(a // (10**i)) % 10 for i in range(14)]
    tb = [(b // (10**i)) % 10 for i in range(14)]
    inp = ta + [TOK_PLUS] + tb  # 29
    s = a + b
    tout = [(s // (10**i)) % 10 for i in range(15)]
    return inp, tout

def make_batch(bs):
    inp_batch = torch.zeros(bs, SEQ_IN, dtype=torch.long)
    tgt_batch = torch.zeros(bs, SEQ_OUT, dtype=torch.long)
    for i in range(bs):
        r = random.random()
        if r < 0.35:
            a = random.randint(0, MAX_VAL)
            b = random.randint(0, MAX_VAL)
        elif r < 0.50:
            # digit-length uniform
            da = random.randint(1,14)
            db = random.randint(1,14)
            a = random.randint(10**(da-1) if da>1 else 0, 10**da -1)
            if a>MAX_VAL: a=random.randint(0,MAX_VAL)
            b = random.randint(10**(db-1) if db>1 else 0, 10**db -1)
            if b>MAX_VAL: b=random.randint(0,MAX_VAL)
        elif r < 0.62:
            # carry heavy 5-9
            a = int(''.join(str(random.randint(5,9)) for _ in range(14))[:14]) % (MAX_VAL+1)
            b = int(''.join(str(random.randint(5,9)) for _ in range(14))[:14]) % (MAX_VAL+1)
            # simpler: random digits 5-9
            a = 0
            b = 0
            for k in range(14):
                a += random.randint(5,9) * (10**k)
                b += random.randint(5,9) * (10**k)
            a %= (MAX_VAL+1)
            b %= (MAX_VAL+1)
        elif r < 0.72:
            # zeros
            if random.random()<0.5:
                a=0
                b=random.randint(0,MAX_VAL)
            else:
                a=random.randint(0,MAX_VAL)
                b=0
        elif r < 0.82:
            # powers of 10
            k = random.randint(0,13)
            a = 10**k
            b = random.randint(0, MAX_VAL)
            if random.random()<0.5:
                a,b=b,a
        elif r < 0.90:
            # trailing 9s
            k = random.randint(1,6)
            base = random.randint(0, MAX_VAL // (10**k))
            a = base * (10**k) + (10**k -1)
            b = random.randint(0, MAX_VAL)
            if a>MAX_VAL: a=MAX_VAL
        else:
            # small
            a = random.randint(0, 10000)
            b = random.randint(0, 10000)
        inp, tout = encode_pair_correct(a,b)
        inp_batch[i] = torch.tensor(inp, dtype=torch.long)
        tgt_batch[i] = torch.tensor(tout, dtype=torch.long)
    return inp_batch, tgt_batch

class Model(nn.Module):
    def __init__(self, d_model, nhead, d_ff, n_layers):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(VOCAB, d_model)
        pe = make_sinusoidal(SEQ_TOTAL, d_model)
        self.register_buffer('pos_emb', pe)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'attn': nn.MultiheadAttention(d_model, nhead, batch_first=True),
                'ln1': nn.LayerNorm(d_model),
                'ln2': nn.LayerNorm(d_model),
                'ff1': nn.Linear(d_model, d_ff),
                'ff2': nn.Linear(d_ff, d_model),
            }))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB)

    def forward(self, x):
        # x: (B, S)
        B,S = x.shape
        h = self.tok_emb(x) + self.pos_emb[:S].unsqueeze(0)
        mask = torch.triu(torch.ones(S,S, device=x.device, dtype=torch.bool), diagonal=1)
        # causal mask: need float -inf
        attn_mask = torch.zeros(S,S, device=x.device)
        attn_mask.masked_fill_(mask, float('-inf'))
        for layer in self.layers:
            # pre-norm
            h2 = layer['ln1'](h)
            a,_ = layer['attn'](h2, h2, h2, attn_mask=attn_mask)
            h = h + a
            h2 = layer['ln2'](h)
            ff = layer['ff2'](F.gelu(layer['ff1'](h2)))
            h = h + ff
        h = self.ln_f(h)
        return self.head(h)

def count_params(m):
    return sum(p.numel() for p in m.parameters())

def eval_model(model, n=2000):
    model.eval()
    correct=0
    with torch.no_grad():
        for _ in range(n):
            a = random.randint(0, MAX_VAL)
            b = random.randint(0, MAX_VAL)
            inp,_ = encode_pair_correct(a,b)
            s = a+b
            target = [(s // (10**i))%10 for i in range(15)]
            # autoregressive
            seq = torch.tensor([inp], dtype=torch.long, device=DEVICE)
            # we need to feed 29 + generated
            # model expects full sequence, we iteratively append
            generated=[]
            for pos in range(15):
                logits = model(seq)  # (1, len, vocab)
                nxt = logits[0, -1].argmax().item()
                generated.append(nxt)
                seq = torch.cat([seq, torch.tensor([[nxt]], device=DEVICE)], dim=1)
            if generated==target:
                correct+=1
    return correct/n

# Allow CLI override
if len(sys.argv)>1:
    D_MODEL=int(sys.argv[1])
if len(sys.argv)>2:
    D_FF=int(sys.argv[2])
if len(sys.argv)>3:
    N_LAYERS=int(sys.argv[3])
if len(sys.argv)>4:
    NHEAD=int(sys.argv[4])

model = Model(D_MODEL, NHEAD, D_FF, N_LAYERS).to(DEVICE)
print(f"params {count_params(model)}")
opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60000)

best=0
best_state=None
steps=60000
bs=512

for step in range(1, steps+1):
    model.train()
    inp, tgt = make_batch(bs)
    # Build full sequence: inp (29) + tgt (15) -> input is inp + tgt[:-1] ? For training we feed 29+15 and predict next token?
    # We train to predict tgt autoregressively: input seq = inp + tgt, target = shift?
    # Simpler: full = inp + tgt, model predicts full, loss only on last 15 positions
    full = torch.cat([inp, tgt], dim=1).to(DEVICE)  # (B,44)
    # input to model is full[:, :-1] ? But we want to predict tgt tokens given inp + previous tgt
    # Use teacher forcing: feed full[:, :-1] and predict full[:,1:]? No, inp is not to be predicted.
    # Instead feed full[:, :-1] where last token is tgt[-2], predict tgt[-1] etc. But easier: feed inp + tgt shifted
    # Let's do: x = cat(inp, tgt) with last token removed? Actually we need 44 tokens input to predict 44 outputs, but we only care last 15.
    # Standard: x = full[:, :-1] (43 tokens), y = full[:,1:] (43 tokens), loss on positions 29..43 (which correspond to tgt)
    # But our model has pos_emb up to 44, so we can do x = torch.cat([inp, tgt[:,:-1]], dim=1) ??? Let's just use full as input and compute loss on tgt positions via shifting.
    # Approach: x = torch.cat([inp, tgt], dim=1)[:, :-1]  -> 43 tokens: inp(29) + tgt(14) (first 14 of tgt)
    # y = torch.cat([inp, tgt], dim=1)[:, 1:] -> 43 tokens, but we want to predict tgt[0] at position 29, etc.
    # Simpler: x = full[:, :-1]  (43), logits = model(x) (B,43,V), target = full[:,1:] (B,43), loss on last 14? Hmm off by one.
    # Let's instead define: x = torch.cat([inp, tgt[:,:-1]], dim=1) ??? Not.
    # Easiest: use full 44 as input, predict next token for each position, but we only have 44 positions, so we need to predict tgt autoregressively:
    # At training, input = inp + tgt[:-1] + ??? Actually to predict tgt[0], input is inp (29 tokens). To predict tgt[1], input is inp + tgt[0] (30 tokens), etc.
    # So we can just feed full[:, :-1] = inp(29) + tgt(14) (first 14 tgt tokens) -> 43 tokens, and target is full[:,1:] but we only care where target is tgt.
    # Let's compute: full = [inp0..inp28, tgt0..tgt14] (44)
    # x = full[:,:-1] = [inp0..inp28, tgt0..tgt13] (43)
    # y = full[:,1:] = [inp1..inp28, tgt0..tgt14] (43)
    # At position 28 (0-indexed), x[28]=inp28, y[28]=tgt0 -> predicts first output token from inp context -> correct
    # At position 29, x[29]=tgt0, y[29]=tgt1 -> predicts second output token -> correct
    # So loss should be on positions 28..42 (15 positions) where y is tgt0..tgt14
    x = full[:, :-1]
    y = full[:, 1:]
    logits = model(x)
    # logits (B,43,V), y (B,43)
    # loss on last 15 positions: indices 28..42 inclusive (15 tokens)
    loss = F.cross_entropy(logits[:, 28:].reshape(-1, VOCAB), y[:, 28:].reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
    sched.step()
    if step % 500 == 0:
        print(f"step {step} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.6f}")
    if step % 2000 == 0 or step==1000:
        acc = eval_model(model, n=1000)
        print(f"  eval {acc*100:.2f}%")
        if acc > best:
            best=acc
            best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
            print(f"  NEW BEST {best*100:.2f}%")
            # save
            torch.save(best_state, f"/workspace/best_D{D_MODEL}_FF{D_FF}_L{N_LAYERS}.pt")
        if acc>=0.99 and step>10000:
            # continue a bit more to ensure stable
            pass
    if step % 10000 == 0:
        # also quick 200 sample
        pass

print(f"BEST {best*100:.2f}%")
if best_state is not None:
    torch.save(best_state, f"/workspace/best_D{D_MODEL}_FF{D_FF}_L{N_LAYERS}.pt")
    print(f"saved best to /workspace/best_D{D_MODEL}_FF{D_FF}_L{N_LAYERS}.pt")
# final eval 5000
if best_state is not None:
    model.load_state_dict(best_state)
acc = eval_model(model, n=5000)
print(f"FINAL 5000 eval {acc*100:.4f}%")
