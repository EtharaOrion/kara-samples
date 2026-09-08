import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE = 12
D_MODEL = 10
NHEAD = 2
D_FF = 18
N_LAYERS = 1
MAX_LEN = 44
INPUT_LEN = 29
OUTPUT_LEN = 15
PAD_TOKEN = 11
PLUS_TOKEN = 10

def _sinusoidal_pos(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-torch.log(torch.tensor(10000.0)) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    n_odd = pe[:, 1::2].shape[1]
    pe[:, 1::2] = torch.cos(pos * div[:n_odd])
    return pe

class TinyTransformer(nn.Module):
    def __init__(self, d_model=D_MODEL, nhead=NHEAD, d_ff=D_FF, vocab=VOCAB_SIZE, max_len=MAX_LEN):
        super().__init__()
        self.d_model = d_model
        self.vocab = vocab
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab, d_model)
        pe = _sinusoidal_pos(max_len, d_model)
        self.register_buffer('pos_emb', pe)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, x):
        B, T = x.shape
        h = self.tok_emb(x) + self.pos_emb[:T].unsqueeze(0)
        h_res = h
        h_norm = self.ln1(h)
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn_out, _ = self.attn(h_norm, h_norm, h_norm, attn_mask=causal)
        h = h_res + attn_out
        h_res2 = h
        h_norm2 = self.ln2(h)
        ffn_out = self.ffn(h_norm2)
        h = h_res2 + ffn_out
        h = self.ln_f(h)
        logits = self.head(h)
        return logits

def build_model():
    model = TinyTransformer()
    metadata = {
        "vocab_size": VOCAB_SIZE,
        "d_model": D_MODEL,
        "nhead": NHEAD,
        "d_ff": D_FF,
        "n_layers": N_LAYERS,
        "max_len": MAX_LEN,
    }
    return model, metadata

def _encode(a: int, b: int):
    tokens = []
    for _ in range(14):
        tokens.append(a % 10)
        a //= 10
    tokens.append(PLUS_TOKEN)
    for _ in range(14):
        tokens.append(b % 10)
        b //= 10
    return tokens

def _decode(tokens):
    val = 0
    mult = 1
    for t in tokens:
        val += int(t) * mult
        mult *= 10
    return val

@torch.no_grad()
def add(model, a: int, b: int) -> int:
    model.eval()
    device = next(model.parameters()).device
    inp = _encode(a, b)
    seq = torch.tensor([inp], dtype=torch.long, device=device)
    generated = []
    for _ in range(OUTPUT_LEN):
        logits = model(seq)
        next_logits = logits[0, -1, :]
        nxt = int(torch.argmax(next_logits).item())
        if nxt >= 10:
            nxt = int(torch.argmax(next_logits[:10]).item())
        generated.append(nxt)
        nxt_tensor = torch.tensor([[nxt]], dtype=torch.long, device=device)
        seq = torch.cat([seq, nxt_tensor], dim=1)
    return _decode(generated)
