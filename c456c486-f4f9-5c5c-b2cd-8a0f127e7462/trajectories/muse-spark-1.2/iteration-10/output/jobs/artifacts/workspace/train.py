import torch
import torch.nn as nn
import torch.nn.functional as F
import random, math, os, sys

# Config
D_MODEL = 10
NHEAD = 2
D_FF = 19
VOCAB = 12
SEQ_LEN = 44
INPUT_LEN = 29
OUTPUT_LEN = 15
MAX_VAL = 99999999999999
BATCH = 512
STEPS = 60000
LR = 4e-4
WD = 0.01
DEVICE = torch.device("cpu")

def _sinusoidal_pe(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    pos = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])
    return pe

class TinyTransformer(nn.Module):
    def __init__(self, d_model=D_MODEL, nhead=NHEAD, d_ff=D_FF, vocab=VOCAB, seq_len=SEQ_LEN):
        super().__init__()
        self.d_model = d_model
        self.vocab = vocab
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab, d_model)
        pe = _sinusoidal_pe(seq_len, d_model)
        self.register_buffer('pos_emb', pe)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln_f = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.head = nn.Linear(d_model, vocab)
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
        self.register_buffer('causal_mask', mask)
    def forward(self, x):
        B,S = x.shape
        h = self.tok_emb(x) + self.pos_emb[:S].unsqueeze(0)
        h_norm = self.ln1(h)
        attn_out,_ = self.attn(h_norm, h_norm, h_norm, attn_mask=self.causal_mask[:S,:S])
        h = h + attn_out
        h_norm2 = self.ln2(h)
        ff = self.ff2(F.gelu(self.ff1(h_norm2)))
        h = h + ff
        h = self.ln_f(h)
        return self.head(h)

def encode_batch(a_list, b_list):
    # a_list, b_list: tensors (B,)
    B = len(a_list)
    inp = torch.zeros(B, INPUT_LEN, dtype=torch.long)
    for i,(a,b) in enumerate(zip(a_list,b_list)):
        for k in range(14):
            inp[i,k] = (a // (10**k)) % 10
        inp[i,14] = 10
        for k in range(14):
            inp[i,15+k] = (b // (10**k)) % 10
    # targets: 15 digits LSD-first of sum
    tgt = torch.zeros(B, OUTPUT_LEN, dtype=torch.long)
    for i,(a,b) in enumerate(zip(a_list,b_list)):
        s = a+b
        for k in range(15):
            tgt[i,k] = (s // (10**k)) % 10
    # full sequence: input + target
    full = torch.cat([inp, tgt], dim=1)  # (B,44)
    return full

def sample_batch(batch_size):
    a = []
    b = []
    for _ in range(batch_size):
        r = random.random()
        if r < 0.35:
            # uniform
            a.append(random.randint(0, MAX_VAL))
            b.append(random.randint(0, MAX_VAL))
        elif r < 0.50:
            # digit-length uniform
            da = random.randint(1,14)
            db = random.randint(1,14)
            a.append(random.randint(0, 10**da -1))
            b.append(random.randint(0, 10**db -1))
        elif r < 0.62:
            # carry-heavy 5-9
            def carry_num():
                v=0
                for k in range(14):
                    d = random.randint(5,9) if random.random()<0.7 else random.randint(0,9)
                    v+= d*(10**k)
                return min(v, MAX_VAL)
            a.append(carry_num()); b.append(carry_num())
        elif r < 0.72:
            # zeros
            if random.random()<0.5:
                a.append(0); b.append(random.randint(0,MAX_VAL))
            else:
                a.append(random.randint(0,MAX_VAL)); b.append(0)
        elif r < 0.82:
            # powers of 10 long carries
            k = random.randint(0,13)
            base = 10**k
            a.append(base); b.append(base)
            # sometimes add random
            if random.random()<0.5:
                a[-1]= random.randint(0,MAX_VAL)//base*base
                b[-1]= random.randint(0,MAX_VAL)//base*base
        elif r < 0.90:
            # trailing 9s
            k = random.randint(1,6)
            a_val = random.randint(0, MAX_VAL//10**k)*10**k + (10**k -1)
            b_val = random.randint(0,9)
            if random.random()<0.5:
                a.append(a_val); b.append(b_val)
            else:
                a.append(b_val); b.append(a_val)
        else:
            # small
            a.append(random.randint(0,1000)); b.append(random.randint(0,1000))
    return a,b

def evaluate(model, n=2000):
    model.eval()
    correct=0
    with torch.no_grad():
        for _ in range(n//BATCH):
            a,b = sample_batch(BATCH)
            # use uniform for eval? use random uniform
            # for true held-out, use uniform
            a = [random.randint(0,MAX_VAL) for _ in range(BATCH)]
            b = [random.randint(0,MAX_VAL) for _ in range(BATCH)]
            full = encode_batch(a,b)
            # autoregressive eval
            inp = full[:,:INPUT_LEN]
            tgt = full[:,INPUT_LEN:]
            seq = inp
            gen = torch.zeros(BATCH, OUTPUT_LEN, dtype=torch.long)
            for step in range(OUTPUT_LEN):
                logits = model(seq)
                nxt = logits[:,-1,:].argmax(dim=-1)
                # mask to 0-9
                # but argmax already will be 0-9 if trained
                gen[:,step]=nxt
                seq = torch.cat([seq, nxt.unsqueeze(1)], dim=1)
            correct += (gen==tgt).all(dim=1).sum().item()
    model.train()
    return correct / (n//BATCH * BATCH)

# training
model = TinyTransformer().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)

best_acc=0
best_state=None

for step in range(1, STEPS+1):
    a,b = sample_batch(BATCH)
    full = encode_batch(a,b).to(DEVICE)
    # input is full[:, :-1], target is full[:,1:] but we only care about output positions
    # Use teacher forcing: predict next token for all positions
    inp = full[:,:-1]  # 43
    tgt = full[:,1:]   # 43, but first 28 are input copying, last 15 are sum
    # Actually we want loss only on output positions? Let's compute loss on all but focus on output
    # Full seq len 44, inp 43, tgt 43. Output positions are 29..43 (15 tokens)
    logits = model(inp)  # (B,43,12)
    loss = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1))
    # alternative: weight output more? but uniform works per history
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    if step % 500 == 0:
        print(f"step {step} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.6f}")
    if step % 2000 == 0:
        acc = evaluate(model, n=2000)
        print(f"  eval acc {acc:.4f} best {best_acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            torch.save(best_state, "/workspace/best.pt")
            print(f"  saved best {best_acc:.4f}")
    if step % 10000 == 0:
        # also quick edge check
        model.eval()
        with torch.no_grad():
            tests = [(0,0),(0,MAX_VAL),(MAX_VAL,0),(MAX_VAL,MAX_VAL),(50000000000000,50000000000000)]
            for av,bv in tests:
                full2 = encode_batch([av],[bv])
                seq = full2[:,:INPUT_LEN]
                gen=[]
                for s in range(OUTPUT_LEN):
                    lg = model(seq)
                    nxt = lg[0,-1].argmax().item()
                    gen.append(nxt)
                    seq = torch.cat([seq, torch.tensor([[nxt]])], dim=1)
                val = sum(d*10**i for i,d in enumerate(gen))
                print(f"    edge {av}+{bv}={val} expected {av+bv} {'OK' if val==av+bv else 'FAIL'}")
        model.train()

print(f"done best {best_acc}")
if best_state is not None:
    model.load_state_dict(best_state)

# final eval
acc = evaluate(model, n=5000)
print(f"final eval 5000 {acc:.4f}")

# write submission.py with embedded weights
import textwrap

# Get state dict
sd = model.state_dict()

# Build submission file content with weights as literals
# We'll embed each tensor as nested lists via .tolist()

def tensor_to_code(t):
    # use repr of list
    return repr(t.tolist())

# Create new submission.py
code = f'''import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = 12
SEQ_LEN = 44
INPUT_LEN = 29
OUTPUT_LEN = 15
D_MODEL = {D_MODEL}
NHEAD = {NHEAD}
D_FF = {D_FF}

def _sinusoidal_pe(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    pos = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-torch.log(torch.tensor(10000.0)) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])
    return pe

class TinyTransformer(nn.Module):
    def __init__(self, d_model=D_MODEL, nhead=NHEAD, d_ff=D_FF, vocab=VOCAB, seq_len=SEQ_LEN):
        super().__init__()
        self.d_model = d_model
        self.vocab = vocab
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab, d_model)
        pe = _sinusoidal_pe(seq_len, d_model)
        self.register_buffer('pos_emb', pe)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln_f = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.head = nn.Linear(d_model, vocab)
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
        self.register_buffer('causal_mask', mask)
    def forward(self, x):
        B,S = x.shape
        h = self.tok_emb(x) + self.pos_emb[:S].unsqueeze(0)
        h_norm = self.ln1(h)
        attn_out,_ = self.attn(h_norm, h_norm, h_norm, attn_mask=self.causal_mask[:S,:S])
        h = h + attn_out
        h_norm2 = self.ln2(h)
        ff = self.ff2(F.gelu(self.ff1(h_norm2)))
        h = h + ff
        h = self.ln_f(h)
        return self.head(h)

def _load_weights(model):
    sd = model.state_dict()
'''

# add weight assignments
for k,v in sd.items():
    # skip buffers that are not parameters? pos_emb and causal_mask are buffers, but we regenerate them
    if k in ('pos_emb','causal_mask'):
        continue
    code += f"    sd['{k}'] = torch.tensor({tensor_to_code(v)}, dtype=torch.float32) if torch.tensor({tensor_to_code(v)}).dtype==torch.float32 else torch.tensor({tensor_to_code(v)})\n"
    # Actually need to handle int? All are float except maybe? Embedding weight is float
    # Simpler: just use torch.tensor with dtype inference

# Need to handle shapes correctly - for LayerNorm weight/bias etc, they are 1D
# The above double tensor creation is wasteful but ok
# Let's do cleaner: generate assignments with correct dtype

# Rebuild more cleanly
code2 = f'''import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = 12
SEQ_LEN = 44
INPUT_LEN = 29
OUTPUT_LEN = 15
D_MODEL = {D_MODEL}
NHEAD = {NHEAD}
D_FF = {D_FF}

def _sinusoidal_pe(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    pos = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-torch.log(torch.tensor(10000.0)) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])
    return pe

class TinyTransformer(nn.Module):
    def __init__(self, d_model=D_MODEL, nhead=NHEAD, d_ff=D_FF, vocab=VOCAB, seq_len=SEQ_LEN):
        super().__init__()
        self.d_model = d_model
        self.vocab = vocab
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab, d_model)
        pe = _sinusoidal_pe(seq_len, d_model)
        self.register_buffer('pos_emb', pe)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln_f = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.head = nn.Linear(d_model, vocab)
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
        self.register_buffer('causal_mask', mask)
    def forward(self, x):
        B,S = x.shape
        h = self.tok_emb(x) + self.pos_emb[:S].unsqueeze(0)
        h_norm = self.ln1(h)
        attn_out,_ = self.attn(h_norm, h_norm, h_norm, attn_mask=self.causal_mask[:S,:S])
        h = h + attn_out
        h_norm2 = self.ln2(h)
        ff = self.ff2(F.gelu(self.ff1(h_norm2)))
        h = h + ff
        h = self.ln_f(h)
        return self.head(h)

_WEIGHTS = {{}}
'''

for k,v in sd.items():
    if k in ('pos_emb','causal_mask'):
        continue
    code2 += f"_WEIGHTS['{k}'] = {repr(v.tolist())}\n"

code2 += '''
_model_cache = None

def build_model():
    global _model_cache
    model = TinyTransformer()
    sd = model.state_dict()
    for k, lst in _WEIGHTS.items():
        sd[k] = torch.tensor(lst, dtype=sd[k].dtype)
    model.load_state_dict(sd)
    _model_cache = model
    metadata = {"vocab": VOCAB, "seq_len": SEQ_LEN, "d_model": D_MODEL, "nhead": NHEAD, "d_ff": D_FF}
    return model, metadata

def _encode(a,b):
    tokens=[]
    for i in range(14):
        tokens.append((a // (10**i)) % 10)
    tokens.append(10)
    for i in range(14):
        tokens.append((b // (10**i)) % 10)
    return tokens

def _decode(tokens):
    val=0
    for i,d in enumerate(tokens):
        val+= int(d)*(10**i)
    return val

def add(model, a: int, b: int) -> int:
    model.eval()
    with torch.no_grad():
        inp = _encode(a,b)
        seq = torch.tensor([inp], dtype=torch.long)
        generated=[]
        for step in range(OUTPUT_LEN):
            logits = model(seq)
            nxt_logit = logits[0,-1,:]
            nxt_logit[10] = float('-inf')
            nxt_logit[11] = float('-inf')
            nxt = int(torch.argmax(nxt_logit).item())
            generated.append(nxt)
            seq = torch.cat([seq, torch.tensor([[nxt]], dtype=torch.long)], dim=1)
        return _decode(generated)
'''

open('/workspace/submission.py','w').write(code2)
print("wrote submission.py")
# quick test
import importlib.util, importlib
spec = importlib.util.spec_from_file_location("sub", "/workspace/submission.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
m,_ = mod.build_model()
for av,bv in [(0,0),(0,MAX_VAL),(MAX_VAL,MAX_VAL),(12345678901234,98765432109876),(50000000000000,50000000000000)]:
    print(av,bv, mod.add(m,av,bv), av+bv, mod.add(m,av,bv)==av+bv)
# param count
cnt = sum(p.numel() for p in m.parameters())
print(f"params {cnt}")
