import torch, math

D_MODEL=10
NHEAD=2
D_FF=19
VOCAB=12
MAX_LEN=44

def build_model():
    import torch.nn as nn
    import torch.nn.functional as F
    import math
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
            self.ff1 = nn.Linear(10, 19)
            self.ff2 = nn.Linear(19, 10)
            self.head = nn.Linear(10, 12)
        def forward(self, x):
            h = self.tok_emb(x) + self.pos_emb[:x.size(1), :].unsqueeze(0)
            h_norm = self.ln1(h)
            try:
                attn_out, _ = self.attn(h_norm, h_norm, h_norm, is_causal=True)
            except TypeError:
                T = x.size(1)
                causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
                attn_out, _ = self.attn(h_norm, h_norm, h_norm, attn_mask=causal)
            h = h + attn_out
            h2 = self.ln2(h)
            ff = self.ff2(torch.nn.functional.gelu(self.ff1(h2)))
            h = h + ff
            h = self.ln_f(h)
            return self.head(h)
    model = TinyTransformer()
    # load weights if exists
    import os
    if os.path.exists("/workspace/best.pt"):
        sd = torch.load("/workspace/best.pt", map_location="cpu")
        model.load_state_dict(sd)
    # count params
    n = sum(p.numel() for p in model.parameters())
    print(f"params {n}")
    return model, {"params": n}

if __name__ == "__main__":
    m, meta = build_model()
    print(meta)

    # generate submission.py with embedded weights
    import torch
    sd = torch.load("/workspace/best.pt", map_location="cpu") if torch.cuda.is_available() or True else None
    # Actually load
    try:
        sd = torch.load("/workspace/best.pt", map_location="cpu")
    except:
        print("no best.pt, using random")
        sd = m.state_dict()

    # Convert to python lists
    def tensor_to_list(t):
        return t.detach().cpu().numpy().tolist()

    # Build submission file
    code = f'''import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
        self.ff1 = nn.Linear(10, 19)
        self.ff2 = nn.Linear(19, 10)
        self.head = nn.Linear(10, 12)
    def forward(self, x):
        h = self.tok_emb(x) + self.pos_emb[:x.size(1), :].unsqueeze(0)
        h_norm = self.ln1(h)
        try:
            attn_out, _ = self.attn(h_norm, h_norm, h_norm, is_causal=True)
        except TypeError:
            T = x.size(1)
            causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            attn_out, _ = self.attn(h_norm, h_norm, h_norm, attn_mask=causal)
        h = h + attn_out
        h2 = self.ln2(h)
        ff = self.ff2(F.gelu(self.ff1(h2)))
        h = h + ff
        h = self.ln_f(h)
        return self.head(h)

_weights = {repr({k: tensor_to_list(v) for k,v in sd.items()})}

def build_model():
    model = TinyTransformer()
    sd2 = {{k: torch.tensor(v, dtype=torch.float32) if v[0] if isinstance(v[0], list) else torch.tensor(v) for k,v in _weights.items()}}
    # Need to handle int vs float: embeddings are float, but we stored all as float lists
    # Reconstruct with correct dtypes
    state = {{}}
    for k, v in _weights.items():
        t = torch.tensor(v)
        # infer dtype from model
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
            tb_tmp = tb % 10
            b_digits.append(tb_tmp)
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
    print("wrote /workspace/submission.py")
    # test
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("sub", "/workspace/submission.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sub"] = mod
    spec.loader.exec_module(mod)
    model2, _ = mod.build_model()
    print("test add", mod.add(model2, 123, 456), "expected", 579)
    print("test add", mod.add(model2, 99999999999999, 1), "expected", 100000000000000)
    # quick eval
    import random
    correct=0
    for _ in range(200):
        a=random.randint(0,99999999999999)
        b=random.randint(0,99999999999999)
        if mod.add(model2,a,b)!=a+b:
            print("fail",a,b,mod.add(model2,a,b),a+b)
            break
        correct+=1
    print(f"quick eval {correct}/200")

