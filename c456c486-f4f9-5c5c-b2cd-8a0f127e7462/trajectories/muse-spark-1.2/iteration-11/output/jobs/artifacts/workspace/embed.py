import torch, math, pathlib

# Load best.pt
sd = torch.load("/workspace/best.pt", map_location="cpu")
# Determine config from sd shapes
# ff1 weight shape is (d_ff, d_model)
d_ff = sd['ff1.weight'].shape[0]
d_model = sd['ff1.weight'].shape[1]
print(f"d_model {d_model} d_ff {d_ff}")
n_params = sum(v.numel() for v in sd.values())
print(f"params from sd {n_params}")
# Need to also count buffers? pos_emb is buffer not param, so params is as above
# But model also has LayerNorm params etc which are in sd
# Let's compute full param count via model
import torch.nn as nn
class TinyTransformer(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.tok_emb = nn.Embedding(12, d_model)
        pe = torch.zeros(44, d_model)
        position = torch.arange(0, 44, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])
        self.register_buffer('pos_emb', pe)
        self.attn = nn.MultiheadAttention(d_model, 2, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln_f = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.head = nn.Linear(d_model, 12)
    def forward(self, x):
        h = self.tok_emb(x) + self.pos_emb[:x.size(1), :].unsqueeze(0)
        h_norm = self.ln1(h)
        T = x.size(1)
        causal = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        attn_out, _ = self.attn(h_norm, h_norm, h_norm, attn_mask=causal)
        h = h + attn_out
        h2 = self.ln2(h)
        ff = self.ff2(torch.nn.functional.gelu(self.ff1(h2)))
        h = h + ff
        h = self.ln_f(h)
        return self.head(h)

model = TinyTransformer(d_model, d_ff)
model.load_state_dict(sd)
print("model params", sum(p.numel() for p in model.parameters()))

# Convert sd to lists
sd_lists = {k: v.cpu().numpy().tolist() for k, v in sd.items()}

# Build submission.py
code = f'''import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(12, {d_model})
        pe = torch.zeros(44, {d_model})
        position = torch.arange(0, 44, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, {d_model}, 2).float() * (-math.log(10000.0) / {d_model}))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:{d_model//2}])
        self.register_buffer('pos_emb', pe)
        self.attn = nn.MultiheadAttention({d_model}, 2, batch_first=True)
        self.ln1 = nn.LayerNorm({d_model})
        self.ln2 = nn.LayerNorm({d_model})
        self.ln_f = nn.LayerNorm({d_model})
        self.ff1 = nn.Linear({d_model}, {d_ff})
        self.ff2 = nn.Linear({d_ff}, {d_model})
        self.head = nn.Linear({d_model}, 12)
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

_weights = {repr(sd_lists)}

def build_model():
    model = TinyTransformer()
    state = {{}}
    for k, v in _weights.items():
        t = torch.tensor(v)
        orig = model.state_dict()[k]
        state[k] = t.to(orig.dtype)
    model.load_state_dict(state)
    return model, {{"params": sum(p.numel() for p in model.parameters())}}

def add(model, a: int, b: int) -> int:
    model.eval()
    with torch.no_grad():
        a_digits = []
        b_digits = []
        ta = a
        tb = b
        for _ in range(14):
            a_digits.append(ta % 10)
            b_digits.append(tb % 10)
            ta //= 10
            tb //= 10
        prefix = a_digits + [10] + b_digits
        seq = prefix.copy()
        for _ in range(15):
            inp = torch.tensor([seq], dtype=torch.long)
            logits = model(inp)
            pred = int(torch.argmax(logits[0, len(seq)-1, :]).item())
            seq.append(pred)
        s_digits = seq[29:44]
        s = 0
        for i, d in enumerate(s_digits):
            s += d * (10 ** i)
        return s
'''

open("/workspace/submission.py","w").write(code)
print("wrote submission.py")

# test
import importlib.util, sys, random
spec = importlib.util.spec_from_file_location("sub", "/workspace/submission.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["sub"] = mod
spec.loader.exec_module(mod)
m, meta = mod.build_model()
print(meta)
print("test", mod.add(m, 123, 456), "expected 579")
print("test", mod.add(m, 99999999999999, 1), "expected 100000000000000")
print("test", mod.add(m, 99999999999999, 99999999999999), "expected 199999999999998")
# quick eval 500
correct=0
for _ in range(500):
    a=random.randint(0,99999999999999)
    b=random.randint(0,99999999999999)
    if mod.add(m,a,b)!=a+b:
        print("fail",a,b,mod.add(m,a,b),a+b)
        break
    correct+=1
print(f"quick eval {correct}/500")
# check attention input dependence
import torch
m.eval()
with torch.no_grad():
    x1 = torch.randint(0,12,(1,10))
    x2 = torch.randint(0,12,(1,10))
    # hook attention weights
    # we can check that output changes with input
    out1 = m(x1)
    out2 = m(x2)
    print("output diff", (out1-out2).abs().mean().item())
    # check attn weights differ
    # get attn weights via forward with need_weights
    h1 = m.tok_emb(x1) + m.pos_emb[:10].unsqueeze(0)
    h1n = m.ln1(h1)
    _, w1 = m.attn(h1n, h1n, h1n, attn_mask=torch.triu(torch.full((10,10), float('-inf')), diagonal=1), need_weights=True, average_attn_weights=False)
    h2 = m.tok_emb(x2) + m.pos_emb[:10].unsqueeze(0)
    h2n = m.ln1(h2)
    _, w2 = m.attn(h2n, h2n, h2n, attn_mask=torch.triu(torch.full((10,10), float('-inf')), diagonal=1), need_weights=True, average_attn_weights=False)
    print("attn diff", (w1-w2).abs().mean().item())

