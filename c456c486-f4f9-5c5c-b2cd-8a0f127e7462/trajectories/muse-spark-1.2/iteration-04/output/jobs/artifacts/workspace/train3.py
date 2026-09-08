import torch, random
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Try d_model=14, 1 layer, d_ff=32, no tying
D_MODEL = 14
NHEAD = 2
N_LAYERS = 1
D_FF = 32
VOCAB = 12
SEQ_IN = 29
SEQ_OUT = 15
SEQ_TOTAL = 44
MAX_VAL = 99_999_999_999_999
BATCH = 512
STEPS = 50000
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

def count_params(m): return sum(p.numel() for p in m.parameters())
def encode_pair(a, b):
    toks = []
    for i in range(14): toks.append((a // (10**i)) % 10)
    toks.append(10)
    for i in range(14): toks.append((b // (10**i)) % 10)
    return toks
def encode_sum(s): return [(s // (10**i)) % 10 for i in range(15)]
def decode_sum(toks): return sum(t * (10**i) for i, t in enumerate(toks))
def sample_batch(bs):
    a_list, b_list = [], []
    for _ in range(bs):
        r = random.random()
        if r < 0.50:
            a = random.randint(0, MAX_VAL); b = random.randint(0, MAX_VAL)
        elif r < 0.70:
            da = random.randint(1, 14); db = random.randint(1, 14)
            a = random.randint(0, 10**da - 1); b = random.randint(0, 10**db - 1)
        elif r < 0.85:
            a = sum(random.randint(5, 9) * (10**i) for i in range(14))
            b = sum(random.randint(5, 9) * (10**i) for i in range(14))
            a = min(a, MAX_VAL); b = min(b, MAX_VAL)
        else:
            a = random.randint(0, 9999); b = random.randint(0, 9999)
        a_list.append(a); b_list.append(b)
    return a_list, b_list
def make_batch(a_list, b_list, device):
    B = len(a_list)
    x = torch.zeros(B, SEQ_TOTAL, dtype=torch.long, device=device)
    y = torch.zeros(B, SEQ_TOTAL, dtype=torch.long, device=device)
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        full = encode_pair(a, b) + encode_sum(a+b)
        x[i] = torch.tensor(full, dtype=torch.long)
        y[i, :-1] = x[i, 1:]
        y[i, -1] = full[-1]
    return x, y

model = AddTransformer().to(DEVICE)
print(f"Params: {count_params(model)}")
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
best_acc = 0
for step in range(1, STEPS+1):
    model.train()
    a_list, b_list = sample_batch(BATCH)
    x, y = make_batch(a_list, b_list, DEVICE)
    logits = model(x)
    loss = F.cross_entropy(logits[:, 28:43].reshape(-1, VOCAB), y[:, 28:43].reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if step % 2000 == 0 or step == 1:
        model.eval()
        correct = 0; total = 1000
        with torch.no_grad():
            for _ in range(total//100):
                ae, be = sample_batch(100)
                for a, b in zip(ae, be):
                    toks = encode_pair(a, b)
                    for _ in range(SEQ_OUT):
                        inp = torch.tensor([toks], dtype=torch.long, device=DEVICE)
                        nxt = model(inp)[0, -1].argmax().item()
                        toks.append(nxt)
                    if decode_sum(toks[29:44]) == a+b: correct += 1
        acc = correct/total
        print(f"Step {step:5d} loss {loss.item():.4f} acc {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "/workspace/best3.pt")
            print(f"  -> saved {acc:.4f}")
print(f"Best {best_acc:.4f}")
# eval best
model.load_state_dict(torch.load("/workspace/best3.pt", map_location=DEVICE))
model.eval()
correct=0; total=5000
with torch.no_grad():
    for _ in range(total//100):
        ae, be = sample_batch(100)
        for a,b in zip(ae,be):
            toks=encode_pair(a,b)
            for _ in range(SEQ_OUT):
                inp=torch.tensor([toks],dtype=torch.long,device=DEVICE)
                toks.append(model(inp)[0,-1].argmax().item())
            if decode_sum(toks[29:44])==a+b: correct+=1
print(f"Final 5k: {correct}/{total}={correct/total:.4f}")
for a,b in [(0,0),(0,MAX_VAL),(MAX_VAL,0),(MAX_VAL,MAX_VAL),(50000000000000,50000000000000),(12345678901234,98765432109876)]:
    toks=encode_pair(a,b)
    with torch.no_grad():
        for _ in range(SEQ_OUT):
            inp=torch.tensor([toks],dtype=torch.long,device=DEVICE)
            toks.append(model(inp)[0,-1].argmax().item())
    pred=decode_sum(toks[29:44])
    print(f"  {a}+{b}={a+b} pred={pred} {'OK' if pred==a+b else 'FAIL'}")
