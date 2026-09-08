import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

# Config
D_MODEL = 16
NHEAD = 2
N_LAYERS = 1
D_FF = 32
VOCAB = 12  # 0-9, '+', '='
SEQ_INPUT = 30  # 14 + 1 + 14 + 1 (=)  -> actually 14+1+14+1=30
SEQ_OUTPUT = 15  # up to 15 digits for sum
SEQ_TOTAL = SEQ_INPUT + SEQ_OUTPUT  # 45
PAD = 11  # unused, for padding if needed
MAX_VAL = 99_999_999_999_999

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}, config: d={D_MODEL} h={NHEAD} L={N_LAYERS} ff={D_FF}")

# Token mapping: 0-9 -> 0-9, '+' -> 10, '=' -> 11
def encode_number(n, length=14):
    s = str(n).zfill(length)
    return [int(c) for c in reversed(s)]  # LSD first

def encode_input(a, b):
    # LSD first: digits of a, '+', digits of b, '='
    a_digits = encode_number(a)
    b_digits = encode_number(b)
    return a_digits + [10] + b_digits + [11]  # 14+1+14+1=30

def encode_output(s):
    s_digits = encode_number(s, 15)  # up to 15 digits
    return s_digits  # 15 tokens LSD first

def decode_output(tokens):
    # tokens are LSD first, 15 digits
    s = ''.join(str(t) for t in reversed(tokens))
    return int(s.lstrip('0') or '0')

class TinyTransformer(nn.Module):
    def __init__(self, d_model=16, nhead=2, n_layers=1, d_ff=32, vocab=12, seq_len=45, tie_weights=False):
        super().__init__()
        self.d_model = d_model
        self.vocab = vocab
        self.seq_len = seq_len
        self.tie_weights = tie_weights
        self.token_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'ln1': nn.LayerNorm(d_model),
                'attn': nn.MultiheadAttention(d_model, nhead, batch_first=True),
                'ln2': nn.LayerNorm(d_model),
                'ff1': nn.Linear(d_model, d_ff),
                'ff2': nn.Linear(d_ff, d_model),
            }))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=True)
        if tie_weights:
            self.head.weight = self.token_emb.weight

    def forward(self, x):
        # x: (B, T)
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.token_emb(x) + self.pos_emb(pos)
        # causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            # pre-norm
            h_norm = layer['ln1'](h)
            attn_out, _ = layer['attn'](h_norm, h_norm, h_norm, attn_mask=mask)
            h = h + attn_out
            h_norm2 = layer['ln2'](h)
            ff = layer['ff2'](F.gelu(layer['ff1'](h_norm2)))
            h = h + ff
        h = self.ln_f(h)
        logits = self.head(h)
        return logits

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def sample_pair():
    r = random.random()
    if r < 0.35:
        a = random.randint(0, MAX_VAL)
        b = random.randint(0, MAX_VAL)
    elif r < 0.50:
        la = random.randint(1, 14)
        lb = random.randint(1, 14)
        a = random.randint(0 if la == 1 else 10**(la-1), 10**la - 1)
        b = random.randint(0 if lb == 1 else 10**(lb-1), 10**lb - 1)
        if random.random() < 0.1:
            a = 0
        if random.random() < 0.1:
            b = 0
    elif r < 0.65:
        a = int(''.join(str(random.randint(5, 9)) for _ in range(14)))
        b = int(''.join(str(random.randint(5, 9)) for _ in range(14)))
    elif r < 0.80:
        # long carry chain: number ending with k 9s + small number
        k = random.randint(3, 14)
        high = random.randint(0, 10**(14-k)-1) if k < 14 else 0
        a = high * (10**k) + int('9'*k)
        b = random.randint(1, 10**k - 1)
        # keep b small-ish to ensure carry propagates
        if b >= 10**6:
            b = random.randint(1, 999999)
    else:
        a = random.randint(0, 99999)
        b = random.randint(0, 99999)
    return a, b

def make_batch(bs):
    inp = torch.zeros(bs, SEQ_TOTAL, dtype=torch.long)
    tgt = torch.zeros(bs, SEQ_TOTAL, dtype=torch.long)
    for i in range(bs):
        a, b = sample_pair()
        s = a + b
        enc_in = encode_input(a, b)
        enc_out = encode_output(s)
        full = enc_in + enc_out
        inp[i] = torch.tensor(full, dtype=torch.long)
        tgt[i] = torch.tensor(full, dtype=torch.long)
    return inp, tgt

def train(steps=50000, batch_size=512, lr=4e-4, wd=0.01, tie=False, d_model=None, nhead=None, n_layers=None, d_ff=None):
    if d_model is None: d_model=D_MODEL
    if nhead is None: nhead=NHEAD
    if n_layers is None: n_layers=N_LAYERS
    if d_ff is None: d_ff=D_FF
    model = TinyTransformer(d_model=d_model, nhead=nhead, n_layers=n_layers, d_ff=d_ff, tie_weights=tie).to(DEVICE)
    print(f"Params: {count_params(model)} d={d_model} h={nhead} L={n_layers} ff={d_ff}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    
    model.train()
    for step in range(1, steps + 1):
        inp, tgt = make_batch(batch_size)
        inp = inp.to(DEVICE)
        tgt = tgt.to(DEVICE)
        # input is inp[:, :-1], target is tgt[:, 1:] but we only care about output positions
        # Actually we do next-token prediction on full sequence
        # Loss only on output positions (30..44) predicting next token
        # Shift: logits for position i predicts token i+1
        logits = model(inp[:, :-1])  # (B, 44, vocab)
        # targets are inp[:, 1:]
        targets = inp[:, 1:]  # (B, 44)
        # Only compute loss on output region: positions 29..43 (0-indexed in logits)
        # inp has 45 tokens, inp[:-1] has 44, logits has 44
        # Output tokens are at positions 30..44 in inp, so in targets they are 29..43
        loss = F.cross_entropy(logits[:, 29:].reshape(-1, VOCAB), targets[:, 29:].reshape(-1))
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        
        if step % 2000 == 0 or step == 1:
            print(f"step {step}/{steps} loss={loss.item():.4f} lr={sched.get_last_lr()[0]:.6f}")
            # quick eval
            acc = evaluate(model, 1000)
            print(f"  eval 1000: {acc:.4f}")
    
    # final eval
    acc = evaluate(model, 5000)
    print(f"Final eval 5000: {acc:.4f}")
    return model

@torch.no_grad()
def evaluate(model, n=2000):
    model.eval()
    correct = 0
    for _ in range(n):
        a = random.randint(0, MAX_VAL)
        b = random.randint(0, MAX_VAL)
        pred = greedy_decode(model, a, b)
        if pred == a + b:
            correct += 1
    model.train()
    return correct / n

@torch.no_grad()
def greedy_decode(model, a, b):
    model.eval()
    enc_in = encode_input(a, b)
    tokens = enc_in[:]  # 30 tokens
    for _ in range(SEQ_OUTPUT):
        x = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
        logits = model(x)
        nxt = logits[0, -1].argmax().item()
        tokens.append(nxt)
    # last 15 are output
    out_tokens = tokens[SEQ_INPUT:SEQ_INPUT+SEQ_OUTPUT]
    return decode_output(out_tokens)

def save_submission(model, path="/workspace/submission.py"):
    sd = model.state_dict()
    # Generate submission.py with embedded weights
    import textwrap
    lines = []
    lines.append("import torch")
    lines.append("import torch.nn as nn")
    lines.append("import torch.nn.functional as F")
    lines.append("")
    lines.append(f"D_MODEL={model.d_model}")
    lines.append(f"NHEAD={model.layers[0]['attn'].num_heads if len(model.layers)>0 else NHEAD}")
    lines.append(f"N_LAYERS={len(model.layers)}")
    lines.append(f"D_FF={model.layers[0]['ff1'].out_features if len(model.layers)>0 else D_FF}")
    lines.append(f"VOCAB={model.vocab}")
    lines.append(f"SEQ_INPUT={SEQ_INPUT}")
    lines.append(f"SEQ_OUTPUT={SEQ_OUTPUT}")
    lines.append(f"SEQ_TOTAL={SEQ_TOTAL}")
    lines.append("")
    lines.append("class TinyTransformer(nn.Module):")
    lines.append("    def __init__(self):")
    lines.append(f"        super().__init__()")
    lines.append(f"        d_model=D_MODEL; nhead=NHEAD; n_layers=N_LAYERS; d_ff=D_FF; vocab=VOCAB; seq_len=SEQ_TOTAL")
    lines.append(f"        self.token_emb=nn.Embedding(vocab,d_model)")
    lines.append(f"        self.pos_emb=nn.Embedding(seq_len,d_model)")
    lines.append(f"        self.layers=nn.ModuleList()")
    lines.append(f"        for _ in range(n_layers):")
    lines.append(f"            self.layers.append(nn.ModuleDict({{'ln1':nn.LayerNorm(d_model),'attn':nn.MultiheadAttention(d_model,nhead,batch_first=True),'ln2':nn.LayerNorm(d_model),'ff1':nn.Linear(d_model,d_ff),'ff2':nn.Linear(d_ff,d_model)}}))")
    lines.append(f"        self.ln_f=nn.LayerNorm(d_model)")
    lines.append(f"        self.head=nn.Linear(d_model,vocab)")
    if model.tie_weights:
        lines.append(f"        self.head.weight=self.token_emb.weight")
    lines.append(f"    def forward(self,x):")
    lines.append(f"        B,T=x.shape; pos=torch.arange(T,device=x.device)")
    lines.append(f"        h=self.token_emb(x)+self.pos_emb(pos)")
    lines.append(f"        mask=torch.triu(torch.ones(T,T,device=x.device,dtype=torch.bool),diagonal=1)")
    lines.append(f"        for layer in self.layers:")
    lines.append(f"            h_norm=layer['ln1'](h)")
    lines.append(f"            attn_out,_=layer['attn'](h_norm,h_norm,h_norm,attn_mask=mask)")
    lines.append(f"            h=h+attn_out")
    lines.append(f"            h_norm2=layer['ln2'](h)")
    lines.append(f"            ff=layer['ff2'](F.gelu(layer['ff1'](h_norm2)))")
    lines.append(f"            h=h+ff")
    lines.append(f"        h=self.ln_f(h)")
    lines.append(f"        return self.head(h)")
    lines.append("")
    # Embed weights as tensors
    lines.append("_SD={}")
    for k, v in sd.items():
        # handle tied weights: skip head.weight if tied
        if model.tie_weights and k == "head.weight":
            continue
        vals = v.cpu().float().flatten().tolist()
        lines.append(f"_SD['{k}']=torch.tensor({vals},dtype=torch.float32).reshape({list(v.shape)})")
    lines.append("")
    lines.append("_MODEL=None")
    lines.append("def build_model():")
    lines.append("    global _MODEL")
    lines.append("    m=TinyTransformer()")
    lines.append("    m.load_state_dict(_SD, strict=False)")
    # if tied, need to re-tie after load
    if model.tie_weights:
        lines.append("    m.head.weight=m.token_emb.weight")
    lines.append("    m.eval()")
    lines.append("    meta={'params': sum(p.numel() for p in m.parameters()), 'd_model': D_MODEL, 'nhead': NHEAD, 'n_layers': N_LAYERS, 'd_ff': D_FF}")
    lines.append("    _MODEL=m")
    lines.append("    return m, meta")
    lines.append("")
    lines.append("def _encode_input(a,b):")
    lines.append("    s=str(a).zfill(14); ad=[int(c) for c in reversed(s)]")
    lines.append("    s=str(b).zfill(14); bd=[int(c) for c in reversed(s)]")
    lines.append("    return ad+[10]+bd+[11]")
    lines.append("def _decode_output(tokens):")
    lines.append("    s=''.join(str(t) for t in reversed(tokens))")
    lines.append("    return int(s.lstrip('0') or '0')")
    lines.append("def add(model,a,b):")
    lines.append("    model.eval()")
    lines.append("    with torch.no_grad():")
    lines.append("        enc=_encode_input(a,b)")
    lines.append("        toks=enc[:]")
    lines.append("        for _ in range(SEQ_OUTPUT):")
    lines.append("            x=torch.tensor([toks],dtype=torch.long)")
    lines.append("            logits=model(x)")
    lines.append("            nxt=int(logits[0,-1].argmax().item())")
    lines.append("            toks.append(nxt)")
    lines.append("        out=toks[SEQ_INPUT:SEQ_INPUT+SEQ_OUTPUT]")
    lines.append("        return _decode_output(out)")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved to {path}, params={sum(p.numel() for p in model.parameters())}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--d_model", type=int, default=16)
    parser.add_argument("--nhead", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=32)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--tie", action="store_true")
    parser.add_argument("--lr", type=float, default=4e-4)
    args = parser.parse_args()
    m = train(steps=args.steps, d_model=args.d_model, nhead=args.nhead, n_layers=args.n_layers, d_ff=args.d_ff, tie=args.tie, lr=args.lr)
    save_submission(m)
    # verify
    import submission
    import importlib
    importlib.reload(submission)
    mod, meta = submission.build_model()
    print(f"meta {meta}")
    tests = [(0,0),(0,MAX_VAL),(MAX_VAL,0),(MAX_VAL,MAX_VAL),(12345678901234,98765432109876),(1,1),(99999999999999,1)]
    for a,b in tests:
        print(f"{a}+{b}={a+b} pred={submission.add(mod,a,b)} {'OK' if submission.add(mod,a,b)==a+b else 'FAIL'}")
    # held-out
    correct=0
    for _ in range(2000):
        a=random.randint(0,MAX_VAL); b=random.randint(0,MAX_VAL)
        if submission.add(mod,a,b)==a+b: correct+=1
    print(f"held-out 2000: {correct/2000:.4f}")
