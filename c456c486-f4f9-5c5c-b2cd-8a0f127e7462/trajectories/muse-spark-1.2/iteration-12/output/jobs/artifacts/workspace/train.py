import torch
import torch.nn as nn
import torch.nn.functional as F
import math, random, os

MAX_VAL = 99_999_999_999_999
VOCAB = 12  # 0-9 digits, 10='+', 11='<pad/bos>'
SEQ_IN = 29  # 14 + 1 + 14
SEQ_OUT = 15
SEQ_TOTAL = SEQ_IN + SEQ_OUT  # 44

D_MODEL = 10
NHEAD = 2
D_FF = 19
N_LAYERS = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}, D={D_MODEL} H={NHEAD} FF={D_FF}")

# Model definition (must match submission.py)
class TinyTransformer(nn.Module):
    def __init__(self, d_model=D_MODEL, nhead=NHEAD, d_ff=D_FF):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(VOCAB, d_model)
        # fixed sinusoidal pos emb as buffer
        pe = torch.zeros(SEQ_TOTAL, d_model)
        pos = torch.arange(0, SEQ_TOTAL, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pos_emb', pe)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB)

    def forward(self, x):
        # x: (B, T)
        B, T = x.shape
        h = self.tok_emb(x) + self.pos_emb[:T].unsqueeze(0)
        # causal mask
        mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        h2, _ = self.attn(h, h, h, attn_mask=mask)
        h = self.ln1(h + h2)
        h2 = self.ff2(F.gelu(self.ff1(h)))
        h = self.ln2(h + h2)
        h = self.ln_f(h)
        return self.head(h)

def encode_pair(a, b):
    # LSD-first: 14 digits each + '+' token
    s = []
    for i in range(14):
        s.append((a // (10**i)) % 10)
    s.append(10)
    for i in range(14):
        s.append((b // (10**i)) % 10)
    return s  # len 29

def encode_sum(c):
    # 15 digits LSD-first
    return [(c // (10**i)) % 10 for i in range(15)]

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
            a = min(a, MAX_VAL); b = min(b, MAX_VAL)
        elif r < 0.62:
            # carry-heavy digits 5-9
            a = int(''.join(str(random.randint(5,9)) for _ in range(14))[:14]) % (MAX_VAL+1)
            b = int(''.join(str(random.randint(5,9)) for _ in range(14))[:14]) % (MAX_VAL+1)
        elif r < 0.72:
            # zeros
            if random.random()<0.5:
                a=0; b=random.randint(0,MAX_VAL)
            else:
                a=random.randint(0,MAX_VAL); b=0
        elif r < 0.82:
            # powers of 10 / long carries
            k = random.randint(0,13)
            a = 10**k
            b = 10**k
            if random.random()<0.5:
                a = MAX_VAL - random.randint(0,1000)
                b = random.randint(1,1000)
        elif r < 0.90:
            # trailing 9s
            k = random.randint(1,10)
            a = int('9'*k) if k<=14 else MAX_VAL
            b = random.randint(1, 10**k)
            a = min(a, MAX_VAL); b = min(b, MAX_VAL)
        else:
            # small
            a = random.randint(0, 9999)
            b = random.randint(0, 9999)

        c = a + b
        inp_seq = encode_pair(a,b)
        out_seq = encode_sum(c)
        full = inp_seq + out_seq  # 44
        inp[i] = torch.tensor(full, dtype=torch.long)
        tgt[i] = torch.tensor(full, dtype=torch.long)
    return inp, tgt

def evaluate(model, n=2000):
    model.eval()
    correct=0
    with torch.no_grad():
        for _ in range(n):
            a = random.randint(0, MAX_VAL)
            b = random.randint(0, MAX_VAL)
            c = a+b
            inp_seq = encode_pair(a,b)
            # autoregressive decode 15 steps
            seq = torch.tensor([inp_seq], dtype=torch.long, device=DEVICE)
            # we need to feed 29 tokens then generate 15
            # start with 29 input tokens, then iteratively predict
            cur = seq[:, :SEQ_IN]  # (1,29)
            # pad to full length with zeros for now, we'll grow
            generated=[]
            for step in range(SEQ_OUT):
                # build input: 29 + generated so far
                inp_cur = torch.cat([seq[:, :SEQ_IN], torch.tensor([generated], dtype=torch.long, device=DEVICE)], dim=1) if generated else cur
                # need to handle batch dim
                logits = model(inp_cur)
                nxt = logits[0, -1].argmax().item()
                generated.append(nxt)
            pred_digits = generated  # LSD first 15
            pred = sum(d * (10**i) for i,d in enumerate(pred_digits))
            if pred==c:
                correct+=1
    return correct/n

# Training
model = TinyTransformer().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60000)

best_acc=0
best_state=None

import time
t0=time.time()
for step in range(1, 60001):
    model.train()
    inp, tgt = make_batch(512)
    inp = inp.to(DEVICE)
    tgt = tgt.to(DEVICE)
    # input is full sequence shifted? We train with teacher forcing on full 44
    # logits for all positions, loss only on output positions (29..43) predicting next? Actually we predict same token at each pos
    # Simpler: input = full[:-1], target = full[1:] but our encoding is not shifted - we want model to predict output tokens given input+previous outputs
    # Use standard: feed full sequence, predict full sequence (like decoder-only LM)
    # Loss on positions SEQ_IN .. SEQ_TOTAL-1 (the sum digits)
    logits = model(inp)  # (B,44,12)
    # shift: predict next token? For our setup, at position i we predict token i, but for autoregressive we want to predict token i given tokens <i
    # So loss on output positions: logits at pos 28 predicts token 29? Let's use standard LM: input is full, target is full, but we mask loss to output region
    # Actually we feed inp (which is full 44) and compute loss on last 15 positions
    loss = F.cross_entropy(logits[:, SEQ_IN-1:-1].reshape(-1, VOCAB), tgt[:, SEQ_IN:].reshape(-1))
    # Explanation: logits at position SEQ_IN-1 (last input token) predicts first output token, etc.
    # So logits[:, SEQ_IN-1 : SEQ_TOTAL-1] vs tgt[:, SEQ_IN : SEQ_TOTAL]
    # That's 15 positions
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()

    if step % 500 == 0:
        elapsed=time.time()-t0
        print(f"step {step} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.6f} time {elapsed:.0f}s")
    if step % 2000 == 0:
        acc = evaluate(model, n=1000)
        print(f"  eval 1000: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
            print(f"  *** new best {best_acc:.4f}")
            torch.save(best_state, "/workspace/best.pt")
        # also quick check
        if acc >= 0.99:
            print(f"  reached 99% at step {step}")

    if step % 10000 == 0:
        # save checkpoint
        torch.save(model.state_dict(), f"/workspace/ckpt_{step}.pt")

print(f"Done best {best_acc:.4f}")
if best_state is not None:
    torch.save(best_state, "/workspace/best.pt")
    # also evaluate best more thoroughly
    model.load_state_dict(best_state)
    model.to(DEVICE)
    acc2 = evaluate(model, n=5000)
    print(f"Final eval 5000: {acc2:.4f}")

# Write submission.py
import torch as _torch
# Load best
state = torch.load("/workspace/best.pt", map_location="cpu")
# Build model to get structure then dump weights as lists
m = TinyTransformer()
m.load_state_dict(state)

# Count params
total = sum(p.numel() for p in m.parameters())
print(f"Total params: {total}")

# Generate submission.py with inline weights
def tensor_to_list(t):
    return t.detach().cpu().numpy().tolist()

# Extract
tok_w = tensor_to_list(m.tok_emb.weight)
# pos_emb is buffer not param, but we need to recreate it as fixed sinusoidal in submission
attn_in_proj_w = tensor_to_list(m.attn.in_proj_weight)
attn_in_proj_b = tensor_to_list(m.attn.in_proj_bias)
attn_out_proj_w = tensor_to_list(m.attn.out_proj.weight)
attn_out_proj_b = tensor_to_list(m.attn.out_proj.bias)
ln1_w = tensor_to_list(m.ln1.weight)
ln1_b = tensor_to_list(m.ln1.bias)
ff1_w = tensor_to_list(m.ff1.weight)
ff1_b = tensor_to_list(m.ff1.bias)
ff2_w = tensor_to_list(m.ff2.weight)
ff2_b = tensor_to_list(m.ff2.bias)
ln2_w = tensor_to_list(m.ln2.weight)
ln2_b = tensor_to_list(m.ln2.bias)
lnf_w = tensor_to_list(m.ln_f.weight)
lnf_b = tensor_to_list(m.ln_f.bias)
head_w = tensor_to_list(m.head.weight)
head_b = tensor_to_list(m.head.bias)

code = f'''import torch
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

class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB, D_MODEL)
        pe = torch.zeros(SEQ_TOTAL, D_MODEL)
        pos = torch.arange(0, SEQ_TOTAL, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, D_MODEL, 2).float() * (-math.log(10000.0) / D_MODEL))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pos_emb', pe)
        self.attn = nn.MultiheadAttention(D_MODEL, NHEAD, batch_first=True)
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.ff1 = nn.Linear(D_MODEL, D_FF)
        self.ff2 = nn.Linear(D_FF, D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB)
        self._load_weights()

    def _load_weights(self):
        import torch as _t
        self.tok_emb.weight.data = _t.tensor({tok_w}, dtype=_t.float32)
        self.attn.in_proj_weight.data = _t.tensor({attn_in_proj_w}, dtype=_t.float32)
        self.attn.in_proj_bias.data = _t.tensor({attn_in_proj_b}, dtype=_t.float32)
        self.attn.out_proj.weight.data = _t.tensor({attn_out_proj_w}, dtype=_t.float32)
        self.attn.out_proj.bias.data = _t.tensor({attn_out_proj_b}, dtype=_t.float32)
        self.ln1.weight.data = _t.tensor({ln1_w}, dtype=_t.float32)
        self.ln1.bias.data = _t.tensor({ln1_b}, dtype=_t.float32)
        self.ff1.weight.data = _t.tensor({ff1_w}, dtype=_t.float32)
        self.ff1.bias.data = _t.tensor({ff1_b}, dtype=_t.float32)
        self.ff2.weight.data = _t.tensor({ff2_w}, dtype=_t.float32)
        self.ff2.bias.data = _t.tensor({ff2_b}, dtype=_t.float32)
        self.ln2.weight.data = _t.tensor({ln2_w}, dtype=_t.float32)
        self.ln2.bias.data = _t.tensor({ln2_b}, dtype=_t.float32)
        self.ln_f.weight.data = _t.tensor({lnf_w}, dtype=_t.float32)
        self.ln_f.bias.data = _t.tensor({lnf_b}, dtype=_t.float32)
        self.head.weight.data = _t.tensor({head_w}, dtype=_t.float32)
        self.head.bias.data = _t.tensor({head_b}, dtype=_t.float32)

    def forward(self, x):
        B, T = x.shape
        h = self.tok_emb(x) + self.pos_emb[:T].unsqueeze(0)
        mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        h2, _ = self.attn(h, h, h, attn_mask=mask)
        h = self.ln1(h + h2)
        h2 = self.ff2(F.gelu(self.ff1(h)))
        h = self.ln2(h + h2)
        h = self.ln_f(h)
        return self.head(h)

_model = None

def build_model():
    global _model
    _model = TinyTransformer()
    _model.eval()
    metadata = {{"params": sum(p.numel() for p in _model.parameters()), "d_model": D_MODEL, "nhead": NHEAD, "d_ff": D_FF}}
    return _model, metadata

def add(model, a: int, b: int) -> int:
    model.eval()
    with torch.no_grad():
        inp = []
        for i in range(14):
            inp.append((a // (10**i)) % 10)
        inp.append(10)
        for i in range(14):
            inp.append((b // (10**i)) % 10)
        generated = []
        for step in range(15):
            seq = torch.tensor([inp + generated], dtype=torch.long)
            logits = model(seq)
            nxt = int(logits[0, -1].argmax().item())
            generated.append(nxt)
        return sum(d * (10**i) for i, d in enumerate(generated))
'''

open("/workspace/submission.py","w").write(code)
print("Wrote /workspace/submission.py")

# quick test
import importlib.util, sys
spec = importlib.util.spec_from_file_location("sub", "/workspace/submission.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
m2, meta = mod.build_model()
print(meta)
for a,b in [(0,0),(0,99999999999999),(99999999999999,1),(12345678901234,98765432109876),(99999999999999,99999999999999)]:
    print(a,b, mod.add(m2,a,b), a+b, mod.add(m2,a,b)==a+b)
# random test
import random as _r
ok=0
for _ in range(500):
    a=_r.randint(0,MAX_VAL); b=_r.randint(0,MAX_VAL)
    if mod.add(m2,a,b)==a+b: ok+=1
print(f"random 500: {ok}/500 {ok/500:.4f}")
