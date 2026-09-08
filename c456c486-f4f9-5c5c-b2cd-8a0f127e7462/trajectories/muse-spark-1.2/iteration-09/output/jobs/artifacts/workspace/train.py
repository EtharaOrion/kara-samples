import torch
import torch.nn as nn
import torch.nn.functional as F
import math, random, os, sys

# Config - will try to push smaller
D_MODEL = 10
NHEAD = 2
D_FF = 19
N_LAYERS = 1
MAX_VAL = 99_999_999_999_999
VOCAB = 12  # 0-9 digits, 10='+', 11=pad/bos
SEQ_IN = 29  # 14 + 1 + 14
SEQ_OUT = 15
SEQ_TOTAL = SEQ_IN + SEQ_OUT  # 44

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}, config D{D_MODEL} H{NHEAD} FF{D_FF} L{N_LAYERS}")

def make_sinusoidal(n_pos, d_model):
    pe = torch.zeros(n_pos, d_model)
    pos = torch.arange(0, n_pos, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[:d_model//2])
    return pe

class TinyTransformer(nn.Module):
    def __init__(self, d_model=10, nhead=2, d_ff=19, n_layers=1):
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
                'ff1': nn.Linear(d_model, d_ff),
                'ff2': nn.Linear(d_ff, d_model),
                'ln2': nn.LayerNorm(d_model),
            }))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB)

    def forward(self, x):
        # x: (B, T)
        B, T = x.shape
        h = self.tok_emb(x) + self.pos_emb[:T].unsqueeze(0)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            a_out, _ = layer['attn'](h, h, h, attn_mask=mask, need_weights=False)
            h = layer['ln1'](h + a_out)
            ff = layer['ff2'](F.gelu(layer['ff1'](h)))
            h = layer['ln2'](h + ff)
        h = self.ln_f(h)
        return self.head(h)

def encode_pair(a, b):
    """Encode a,b into 29 tokens LSD-first"""
    toks = []
    for _ in range(14):
        toks.append(a % 10)
        a //= 10
    toks.append(10)  # '+'
    for _ in range(14):
        toks.append(b % 10)
        b //= 10
    return toks  # len 29

def encode_sum(s):
    toks = []
    for _ in range(15):
        toks.append(s % 10)
        s //= 10
    return toks

def make_batch(bs):
    inp = torch.zeros(bs, SEQ_TOTAL, dtype=torch.long)
    tgt = torch.zeros(bs, SEQ_TOTAL, dtype=torch.long)
    for i in range(bs):
        r = random.random()
        if r < 0.35:
            a = random.randint(0, MAX_VAL)
            b = random.randint(0, MAX_VAL)
        elif r < 0.50:
            # digit-length uniform
            da = random.randint(1, 14)
            db = random.randint(1, 14)
            a = random.randint(10**(da-1) if da>1 else 0, 10**da -1)
            b = random.randint(10**(db-1) if db>1 else 0, 10**db -1)
            if da==14: a = min(a, MAX_VAL)
            if db==14: b = min(b, MAX_VAL)
        elif r < 0.62:
            # carry-heavy
            a = 0; b = 0
            for pos in range(14):
                da = random.randint(5, 9)
                db = random.randint(5, 9)
                if random.random() < 0.3:
                    da = random.randint(0, 9)
                    db = random.randint(0, 9)
                a += da * (10**pos)
                b += db * (10**pos)
            a = min(a, MAX_VAL); b = min(b, MAX_VAL)
        elif r < 0.72:
            # zeros
            if random.random() < 0.5:
                a = 0; b = random.randint(0, MAX_VAL)
            else:
                a = random.randint(0, MAX_VAL); b = 0
            if random.random() < 0.3:
                a = random.randint(0, 1000); b = random.randint(0, 1000)
        elif r < 0.82:
            # powers of 10 / long carries
            if random.random() < 0.5:
                k = random.randint(0, 13)
                a = 10**k
                b = 10**k
                if random.random() < 0.5:
                    b = MAX_VAL - random.randint(0, 1000)
                    a = random.randint(0, 1000)
            else:
                a = random.randint(0, MAX_VAL)
                b = (10**14 - 1) - a + random.randint(-5, 5)
                b = max(0, min(MAX_VAL, b))
        elif r < 0.90:
            # trailing 9s
            nines = random.randint(1, 10)
            base_a = random.randint(0, 10**(14-nines)-1) if nines<14 else 0
            a = base_a * (10**nines) + (10**nines - 1)
            b = random.randint(0, MAX_VAL)
            if random.random() < 0.5:
                b, a = a, b
            a = min(a, MAX_VAL); b = min(b, MAX_VAL)
        else:
            # small
            a = random.randint(0, 10000)
            b = random.randint(0, 10000)

        enc_in = encode_pair(a, b)
        enc_out = encode_sum(a + b)
        full = enc_in + enc_out  # 44
        inp[i] = torch.tensor(full, dtype=torch.long)
        tgt[i] = torch.tensor(full, dtype=torch.long)
    return inp, tgt

def evaluate(model, n=2000):
    model.eval()
    correct = 0
    with torch.no_grad():
        for _ in range(n):
            a = random.randint(0, MAX_VAL)
            b = random.randint(0, MAX_VAL)
            enc = encode_pair(a, b)
            # autoregressive
            seq = torch.tensor([enc], dtype=torch.long, device=DEVICE)
            for step in range(SEQ_OUT):
                logits = model(seq)
                nxt = logits[0, -1].argmax().item()
                seq = torch.cat([seq, torch.tensor([[nxt]], device=DEVICE)], dim=1)
            pred_digits = seq[0, SEQ_IN:].tolist()
            # decode LSD-first
            pred = 0
            for idx, d in enumerate(pred_digits):
                pred += d * (10**idx)
            if pred == a + b:
                correct += 1
    return correct / n

# Build model
model = TinyTransformer(D_MODEL, NHEAD, D_FF, N_LAYERS).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60000)
crit = nn.CrossEntropyLoss()

# Count params
n_params = sum(p.numel() for p in model.parameters())
print(f"Params: {n_params}")

best = 0
step = 0
BATCH = 512
TOTAL_STEPS = 60000

for step in range(1, TOTAL_STEPS+1):
    model.train()
    inp, tgt = make_batch(BATCH)
    inp = inp.to(DEVICE)
    tgt = tgt.to(DEVICE)
    # input is inp[:, :-1], target is inp[:, 1:] but we only care about output positions
    # Actually we feed full sequence and predict next token; loss only on output part
    # Use teacher forcing: input = full sequence, predict next token
    # Simpler: feed inp[:, :-1] and predict inp[:, 1:]
    # But our seq is 44, we want to predict all positions; loss on last 15 is most important
    # We'll compute loss on all positions after SEQ_IN-1
    x = inp[:, :-1]  # 43
    y = inp[:, 1:]   # 43
    logits = model(x)  # (B, 43, vocab)
    # Only compute loss where y corresponds to output region: positions SEQ_IN-1 onwards
    # x positions 0..42, y positions 1..43, output starts at index SEQ_IN (29) in full seq
    # So in x/y, output region is from SEQ_IN-1 =28 onwards
    loss = crit(logits[:, SEQ_IN-1:].reshape(-1, VOCAB), y[:, SEQ_IN-1:].reshape(-1))
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()

    if step % 500 == 0:
        print(f"step {step} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.6f}")

    if step % 5000 == 0 or step == TOTAL_STEPS:
        acc = evaluate(model, n=1000)
        print(f"  eval {acc*100:.2f}%")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), "/workspace/best.pt")
            print(f"  new best {best*100:.2f}% saved")

# Final eval
model.load_state_dict(torch.load("/workspace/best.pt", map_location=DEVICE))
acc = evaluate(model, n=5000)
print(f"FINAL 5000 eval: {acc*100:.2f}%")

# Edge cases
edges = [(0,0),(0,MAX_VAL),(MAX_VAL,0),(MAX_VAL,MAX_VAL),(MAX_VAL,1),(12345678901234,98765432109876),(50000000000000,50000000000000),(99999999999999,1),(10000000000000,10000000000000)]
print("Edges:")
model.eval()
with torch.no_grad():
    for a,b in edges:
        enc = encode_pair(a,b)
        seq = torch.tensor([enc], dtype=torch.long, device=DEVICE)
        for _ in range(SEQ_OUT):
            logits = model(seq)
            nxt = logits[0,-1].argmax().item()
            seq = torch.cat([seq, torch.tensor([[nxt]], device=DEVICE)], dim=1)
        pred_digits = seq[0, SEQ_IN:].tolist()
        pred = sum(d*(10**i) for i,d in enumerate(pred_digits))
        ok = "OK" if pred==a+b else "FAIL"
        print(f"  {a}+{b}={a+b} pred {pred} {ok}")

# Write submission.py
import textwrap

# Load best weights
sd = torch.load("/workspace/best.pt", map_location="cpu")
# Convert to python lists for embedding in file
def tensor_to_list(t):
    return t.detach().cpu().numpy().tolist()

# Build submission content
# We need to embed weights as plain python lists and reconstruct tensors

# Get state dict keys
# We'll generate code that creates model and loads weights

# For reproducibility, save weights as lists in file
# Use torch.tensor() reconstruction

# Create submission.py
sub_code = f'''import torch
import torch.nn as nn
import torch.nn.functional as F
import math

VOCAB = 12
SEQ_IN = 29
SEQ_OUT = 15
SEQ_TOTAL = 44
D_MODEL = {D_MODEL}
NHEAD = {NHEAD}
D_FF = {D_FF}
N_LAYERS = {N_LAYERS}

def _sinusoidal(n_pos, d_model):
    pe = torch.zeros(n_pos, d_model)
    pos = torch.arange(0, n_pos, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[:d_model//2])
    return pe

class TinyTransformer(nn.Module):
    def __init__(self, d_model=D_MODEL, nhead=NHEAD, d_ff=D_FF, n_layers=N_LAYERS):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(VOCAB, d_model)
        pe = _sinusoidal(SEQ_TOTAL, d_model)
        self.register_buffer('pos_emb', pe)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({{
                'attn': nn.MultiheadAttention(d_model, nhead, batch_first=True),
                'ln1': nn.LayerNorm(d_model),
                'ff1': nn.Linear(d_model, d_ff),
                'ff2': nn.Linear(d_ff, d_model),
                'ln2': nn.LayerNorm(d_model),
            }}))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB)

    def forward(self, x):
        B, T = x.shape
        h = self.tok_emb(x) + self.pos_emb[:T].unsqueeze(0)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            a_out, _ = layer['attn'](h, h, h, attn_mask=mask, need_weights=False)
            h = layer['ln1'](h + a_out)
            ff = layer['ff2'](F.gelu(layer['ff1'](h)))
            h = layer['ln2'](h + ff)
        h = self.ln_f(h)
        return self.head(h)

_weights = {repr({k: tensor_to_list(v) for k,v in sd.items()})}

def build_model():
    model = TinyTransformer()
    sd2 = {{k: torch.tensor(v) for k,v in _weights.items()}}
    model.load_state_dict(sd2)
    model.eval()
    return model, {{"vocab": VOCAB, "seq_in": SEQ_IN, "seq_out": SEQ_OUT}}

def add(model, a: int, b: int) -> int:
    model.eval()
    device = next(model.parameters()).device
    toks = []
    aa, bb = a, b
    for _ in range(14):
        toks.append(aa % 10)
        aa //= 10
    toks.append(10)
    for _ in range(14):
        toks.append(bb % 10)
        bb //= 10
    seq = torch.tensor([toks], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(SEQ_OUT):
            logits = model(seq)
            nxt = int(logits[0, -1].argmax().item())
            seq = torch.cat([seq, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
    digits = seq[0, SEQ_IN:].tolist()
    s = 0
    for i, d in enumerate(digits):
        s += d * (10 ** i)
    return int(s)
'''

with open("/workspace/submission.py", "w") as f:
    f.write(sub_code)

print("Wrote /workspace/submission.py")
# Quick test
import importlib.util, sys
spec = importlib.util.spec_from_file_location("submission", "/workspace/submission.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["submission"] = mod
spec.loader.exec_module(mod)
m, meta = mod.build_model()
print("Test add:", mod.add(m, 123, 456), "expected", 579)
print("Test add max:", mod.add(m, MAX_VAL, MAX_VAL), "expected", MAX_VAL*2)
# corruption test
print("Params in submission:", sum(p.numel() for p in m.parameters()))
