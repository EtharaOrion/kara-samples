import torch
import torch.nn as nn
import torch.nn.functional as F
import random, math

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

D_MODEL = 16
NHEAD = 2
N_LAYERS = 1
D_FF = 32
VOCAB = 12
SEQ_IN = 29
SEQ_OUT = 15
SEQ_TOTAL = 44
MAX_VAL = 99_999_999_999_999
BATCH = 512
STEPS = 60000
LR = 4e-4
WD = 0.01

class AddTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB, D_MODEL)
        self.pos_emb = nn.Embedding(SEQ_TOTAL, D_MODEL)
        self.layers = nn.ModuleList()
        for _ in range(N_LAYERS):
            self.layers.append(nn.ModuleDict({
                'attn': nn.MultiheadAttention(D_MODEL, NHEAD, batch_first=True),
                'ln1': nn.LayerNorm(D_MODEL),
                'ln2': nn.LayerNorm(D_MODEL),
                'ff1': nn.Linear(D_MODEL, D_FF),
                'ff2': nn.Linear(D_FF, D_MODEL),
            }))
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)
        # tie weights
        self.head.weight = self.tok_emb.weight

    def forward(self, x):
        B, T = x.shape
        h = self.tok_emb(x) + self.pos_emb(torch.arange(T, device=x.device))
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            h_norm = layer['ln1'](h)
            attn_out, _ = layer['attn'](h_norm, h_norm, h_norm, attn_mask=causal)
            h = h + attn_out
            h_norm2 = layer['ln2'](h)
            ff = layer['ff2'](F.gelu(layer['ff1'](h_norm2)))
            h = h + ff
        h = self.ln_f(h)
        return self.head(h)

def count_params(m):
    return sum(p.numel() for p in m.parameters())

def encode_pair(a, b):
    toks = []
    for i in range(14):
        toks.append((a // (10**i)) % 10)
    toks.append(10)
    for i in range(14):
        toks.append((b // (10**i)) % 10)
    return toks

def encode_sum(s):
    return [(s // (10**i)) % 10 for i in range(15)]

def decode_sum(toks):
    return sum(t * (10**i) for i, t in enumerate(toks))

def sample_batch(batch_size):
    a_list, b_list = [], []
    for _ in range(batch_size):
        r = random.random()
        if r < 0.40:
            a = random.randint(0, MAX_VAL)
            b = random.randint(0, MAX_VAL)
        elif r < 0.55:
            da = random.randint(1, 14)
            db = random.randint(1, 14)
            a = random.randint(0, 10**da - 1)
            b = random.randint(0, 10**db - 1)
        elif r < 0.70:
            a = sum(random.randint(5, 9) * (10**i) for i in range(14))
            b = sum(random.randint(5, 9) * (10**i) for i in range(14))
            a = min(a, MAX_VAL); b = min(b, MAX_VAL)
        elif r < 0.80:
            # zeros and small
            if random.random() < 0.3:
                a = 0
                b = random.randint(0, MAX_VAL)
                if random.random() < 0.5:
                    b = 0
            else:
                a = random.randint(0, 9999)
                b = random.randint(0, 9999)
        elif r < 0.90:
            # trailing 9s
            a = random.randint(0, MAX_VAL // 10) * 10 + 9
            b = random.randint(0, MAX_VAL // 10) * 10 + 9
            # sometimes many 9s
            if random.random() < 0.3:
                a = int("9" * random.randint(1, 14))
                b = int("9" * random.randint(1, 14))
        else:
            # powers of 10 and edge cases
            a = random.choice([0, 1, 10, 100, 1000, 10**7, 10**13, MAX_VAL, MAX_VAL-1])
            b = random.choice([0, 1, 10, 100, 1000, 10**7, 10**13, MAX_VAL, MAX_VAL-1])
            if random.random() < 0.5:
                a = random.randint(0, MAX_VAL)
                b = random.randint(0, MAX_VAL)
        a_list.append(a); b_list.append(b)
    return a_list, b_list

def make_batch_tensor(a_list, b_list, device):
    B = len(a_list)
    x = torch.zeros(B, SEQ_TOTAL, dtype=torch.long, device=device)
    y = torch.zeros(B, SEQ_TOTAL, dtype=torch.long, device=device)
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        prefix = encode_pair(a, b)
        s_toks = encode_sum(a+b)
        full = prefix + s_toks
        x[i] = torch.tensor(full, dtype=torch.long)
        y[i, :-1] = x[i, 1:]
        y[i, -1] = s_toks[-1]
    return x, y

model = AddTransformer().to(DEVICE)
print(f"Params: {count_params(model)} (tied)")
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)

best_acc = 0
best_step = 0
for step in range(1, STEPS+1):
    model.train()
    a_list, b_list = sample_batch(BATCH)
    x, y = make_batch_tensor(a_list, b_list, DEVICE)
    logits = model(x)
    loss = F.cross_entropy(logits[:, 28:43].reshape(-1, VOCAB), y[:, 28:43].reshape(-1))
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    if step % 2000 == 0 or step == 1:
        model.eval()
        correct = 0
        total = 1000
        with torch.no_grad():
            for _ in range(total // 50):
                ae, be = sample_batch(50)
                for a, b in zip(ae, be):
                    prefix = encode_pair(a, b)
                    toks = prefix[:]
                    for _ in range(SEQ_OUT):
                        inp = torch.tensor([toks], dtype=torch.long, device=DEVICE)
                        logits2 = model(inp)
                        nxt = logits2[0, -1].argmax().item()
                        toks.append(nxt)
                    pred = decode_sum(toks[29:44])
                    if pred == a+b:
                        correct += 1
        acc = correct/total
        # also test edges
        edge_ok = 0
        edges = [(0,0),(0,MAX_VAL),(MAX_VAL,0),(MAX_VAL,MAX_VAL),(0,1),(1,0),(50000000000000,50000000000000)]
        with torch.no_grad():
            for a,b in edges:
                prefix = encode_pair(a,b)
                toks = prefix[:]
                for _ in range(SEQ_OUT):
                    inp = torch.tensor([toks], dtype=torch.long, device=DEVICE)
                    logits2 = model(inp)
                    nxt = logits2[0, -1].argmax().item()
                    toks.append(nxt)
                pred = decode_sum(toks[29:44])
                if pred == a+b:
                    edge_ok += 1
        print(f"Step {step:5d} loss {loss.item():.4f} acc {acc:.4f} edges {edge_ok}/{len(edges)}")
        if acc >= best_acc and edge_ok == len(edges):
            best_acc = acc
            best_step = step
            torch.save(model.state_dict(), "/workspace/best2.pt")
            print(f"  -> saved best {acc:.4f} at {step}")
        elif acc > best_acc and edge_ok >= len(edges)-1:
            # save if acc is better even if one edge fails, but track
            pass

print(f"Best acc {best_acc:.4f} at {best_step}")
# load best and final eval
try:
    model.load_state_dict(torch.load("/workspace/best2.pt", map_location=DEVICE))
except:
    print("No best2.pt, using current")
model.eval()
correct = 0
total = 5000
with torch.no_grad():
    for _ in range(total // 100):
        ae, be = sample_batch(100)
        for a, b in zip(ae, be):
            prefix = encode_pair(a,b)
            toks = prefix[:]
            for _ in range(SEQ_OUT):
                inp = torch.tensor([toks], dtype=torch.long, device=DEVICE)
                logits2 = model(inp)
                nxt = logits2[0, -1].argmax().item()
                toks.append(nxt)
            pred = decode_sum(toks[29:44])
            if pred == a+b:
                correct += 1
print(f"Final 5k acc: {correct}/{total} = {correct/total:.4f}")
edges = [(0,0),(0,MAX_VAL),(MAX_VAL,0),(MAX_VAL,MAX_VAL),(MAX_VAL,1),(12345678901234,98765432109876),(1,1),(99999999999999,1),(50000000000000,50000000000000)]
print("Edges:")
for a,b in edges:
    prefix = encode_pair(a,b)
    toks = prefix[:]
    with torch.no_grad():
        for _ in range(SEQ_OUT):
            inp = torch.tensor([toks], dtype=torch.long, device=DEVICE)
            logits2 = model(inp)
            nxt = logits2[0, -1].argmax().item()
            toks.append(nxt)
    pred = decode_sum(toks[29:44])
    print(f"  {a}+{b}={a+b} pred={pred} {'OK' if pred==a+b else 'FAIL'}")

# write submission
sd = model.state_dict()
# need to handle tied weight: only one copy in state_dict? Actually both share same tensor, but state_dict will have both keys pointing to same storage? Let's check
print("SD keys:", list(sd.keys()))
# For submission, we need to ensure tying is preserved
# We'll create submission that ties after loading

lines = []
lines.append("import torch")
lines.append("import torch.nn as nn")
lines.append("import torch.nn.functional as F")
lines.append("")
lines.append(f"D_MODEL={D_MODEL}")
lines.append(f"NHEAD={NHEAD}")
lines.append(f"N_LAYERS={N_LAYERS}")
lines.append(f"D_FF={D_FF}")
lines.append(f"VOCAB={VOCAB}")
lines.append(f"SEQ_TOTAL={SEQ_TOTAL}")
lines.append(f"SEQ_IN={SEQ_IN}")
lines.append(f"SEQ_OUT={SEQ_OUT}")
lines.append("")
lines.append("class AddTransformer(nn.Module):")
lines.append("    def __init__(self):")
lines.append("        super().__init__()")
lines.append("        self.tok_emb = nn.Embedding(VOCAB, D_MODEL)")
lines.append("        self.pos_emb = nn.Embedding(SEQ_TOTAL, D_MODEL)")
lines.append("        self.layers = nn.ModuleList()")
lines.append("        for _ in range(N_LAYERS):")
lines.append("            self.layers.append(nn.ModuleDict({")
lines.append("                'attn': nn.MultiheadAttention(D_MODEL, NHEAD, batch_first=True),")
lines.append("                'ln1': nn.LayerNorm(D_MODEL),")
lines.append("                'ln2': nn.LayerNorm(D_MODEL),")
lines.append("                'ff1': nn.Linear(D_MODEL, D_FF),")
lines.append("                'ff2': nn.Linear(D_FF, D_MODEL),")
lines.append("            }))")
lines.append("        self.ln_f = nn.LayerNorm(D_MODEL)")
lines.append("        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)")
lines.append("        self.head.weight = self.tok_emb.weight")
lines.append("    def forward(self, x):")
lines.append("        B, T = x.shape")
lines.append("        h = self.tok_emb(x) + self.pos_emb(torch.arange(T, device=x.device))")
lines.append("        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)")
lines.append("        for layer in self.layers:")
lines.append("            h_norm = layer['ln1'](h)")
lines.append("            attn_out, _ = layer['attn'](h_norm, h_norm, h_norm, attn_mask=causal)")
lines.append("            h = h + attn_out")
lines.append("            h_norm2 = layer['ln2'](h)")
lines.append("            ff = layer['ff2'](F.gelu(layer['ff1'](h_norm2)))")
lines.append("            h = h + ff")
lines.append("        h = self.ln_f(h)")
lines.append("        return self.head(h)")
lines.append("")
lines.append("_model = None")
lines.append("")
lines.append("def _get_model():")
lines.append("    global _model")
lines.append("    if _model is not None:")
lines.append("        return _model")
lines.append("    _model = AddTransformer()")
lines.append("    _sd = {")
# Only save unique params - tok_emb and head share, so save tok_emb only, but need to handle loading
# Save all except head.weight (since it's tied)
for k, v in sd.items():
    if k == "head.weight":
        continue
    lst = v.cpu().numpy().tolist()
    lines.append(f"        {repr(k)}: torch.tensor({repr(lst)}),")
lines.append("    }")
lines.append("    _model.load_state_dict(_sd, strict=False)")
lines.append("    _model.eval()")
lines.append("    return _model")
lines.append("")
lines.append("def build_model():")
lines.append("    m = _get_model()")
lines.append("    return m, {}")
lines.append("")
lines.append("def add(model, a, b):")
lines.append("    model.eval()")
lines.append("    with torch.no_grad():")
lines.append("        toks = []")
lines.append("        for i in range(14):")
lines.append("            toks.append((a // (10**i)) % 10)")
lines.append("        toks.append(10)")
lines.append("        for i in range(14):")
lines.append("            toks.append((b // (10**i)) % 10)")
lines.append("        for _ in range(SEQ_OUT):")
lines.append("            inp = torch.tensor([toks], dtype=torch.long)")
lines.append("            logits = model(inp)")
lines.append("            nxt = int(logits[0, -1].argmax().item())")
lines.append("            toks.append(nxt)")
lines.append("        s_toks = toks[SEQ_IN:SEQ_IN+SEQ_OUT]")
lines.append("        return sum(t * (10**i) for i, t in enumerate(s_toks))")

with open("/workspace/submission.py", "w") as f:
    f.write("\n".join(lines))
print("Wrote submission.py tied")

# verify
import importlib, sys
if 'submission' in sys.modules:
    del sys.modules['submission']
import submission as sub2
m2, _ = sub2.build_model()
print(f"submission params: {sum(p.numel() for p in m2.parameters())} (unique: {sum(p.numel() for n,p in m2.named_parameters())})")
# count unique storage
seen = set()
total_unique = 0
for p in m2.parameters():
    pid = p.data_ptr()
    if pid not in seen:
        seen.add(pid)
        total_unique += p.numel()
print(f"unique params (tied counted once): {total_unique}")
for a,b in [(0,0),(MAX_VAL,MAX_VAL),(12345678901234,98765432109876),(50000000000000,50000000000000)]:
    r = sub2.add(m2, a, b)
    print(f"verify {a}+{b}={a+b} got {r} {'OK' if r==a+b else 'FAIL'}")
