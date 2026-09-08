import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import os

# --- Config ---
VOCAB = 12  # 0-9 digits, 10='+', 11='='
PAD = 12    # not in vocab, used for masking if needed
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
D_FF = 128
MAX_INPUT = 30   # 14 + 1 + 14 + 1
MAX_OUTPUT = 15  # up to 15 digits
SEQ_LEN = MAX_INPUT + MAX_OUTPUT  # 45
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Tokenization: LSD-first
# Input seq: [a0,a1,...,a13, '+', b0,...,b13, '=']  (30 tokens)
# Output seq: [s0,s1,...,s14]  (15 tokens, LSD first, padded with 0 beyond actual length)
# Full seq for training: input + output = 45 tokens

def encode_pair(a, b):
    inp = []
    for i in range(14):
        inp.append(a % 10)
        a //= 10
    inp.append(10)  # '+'
    for i in range(14):
        inp.append(b % 10)
        b //= 10
    inp.append(11)  # '='
    return inp  # len 30

def encode_sum(s):
    out = []
    for i in range(15):
        out.append(s % 10)
        s //= 10
    return out  # len 15

def decode_sum(tokens):
    val = 0
    for i, t in enumerate(tokens):
        val += t * (10 ** i)
    return val

# --- Model ---
class TransformerAdd(nn.Module):
    def __init__(self, vocab=VOCAB, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF, max_len=SEQ_LEN):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.vocab = vocab
        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                'ln1': nn.LayerNorm(d_model),
                'ln2': nn.LayerNorm(d_model),
                'ff1': nn.Linear(d_model, d_ff),
                'ff2': nn.Linear(d_ff, d_model),
            }))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, x):
        # x: (B, T) token ids
        B, T = x.shape
        h = self.token_emb(x) + self.pos_emb(torch.arange(T, device=x.device))
        # causal mask
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            # pre-norm
            h_norm = layer['ln1'](h)
            attn_out, _ = layer['attn'](h_norm, h_norm, h_norm, attn_mask=causal, need_weights=False)
            h = h + attn_out
            h_norm2 = layer['ln2'](h)
            ff = layer['ff2'](F.gelu(layer['ff1'](h_norm2)))
            h = h + ff
        h = self.ln_f(h)
        logits = self.head(h)
        return logits  # (B, T, vocab)


def generate_batch(batch_size):
    # Mixed curriculum
    a_list, b_list = [], []
    for _ in range(batch_size):
        r = random.random()
        if r < 0.5:
            # uniform over full range
            a = random.randint(0, 99999999999999)
            b = random.randint(0, 99999999999999)
        elif r < 0.7:
            # random digit length
            da = random.randint(1, 14)
            db = random.randint(1, 14)
            a = random.randint(0, 10**da - 1)
            b = random.randint(0, 10**db - 1)
        elif r < 0.85:
            # carry-heavy: digits 5-9
            a = 0
            b = 0
            for i in range(14):
                a += random.randint(5, 9) * (10**i)
                b += random.randint(5, 9) * (10**i)
            # randomize a bit
            if random.random() < 0.5:
                a = random.randint(0, 99999999999999)
        else:
            # small numbers
            a = random.randint(0, 999999)
            b = random.randint(0, 999999)
        a_list.append(a)
        b_list.append(b)

    # Build sequences
    inp_seqs = []
    tgt_seqs = []
    for a, b in zip(a_list, b_list):
        inp = encode_pair(a, b)
        out = encode_sum(a + b)
        full = inp + out  # 45
        inp_seqs.append(full)
        tgt_seqs.append(full)
    x = torch.tensor(inp_seqs, dtype=torch.long, device=DEVICE)
    # For loss: only compute on output positions (30..44)
    return x

def train():
    model = TransformerAdd().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=120000)

    best_acc = 0
    step = 0
    max_steps = 120000
    batch_size = 512
    eval_every = 2000
    save_every = 10000

    # For quick eval
    def evaluate(n=2000):
        model.eval()
        correct = 0
        with torch.no_grad():
            for _ in range(n // 256 + 1):
                bs = min(256, n - correct - 0)
                # generate eval batch
                pairs = []
                for _ in range(256):
                    a = random.randint(0, 99999999999999)
                    b = random.randint(0, 99999999999999)
                    pairs.append((a, b))
                # autoregressive decode
                for a, b in pairs:
                    if correct + sum(1 for _ in []) >= n:
                        break
                    inp = encode_pair(a, b)
                    # greedy decode 15 steps
                    seq = torch.tensor([inp], dtype=torch.long, device=DEVICE)
                    for pos in range(15):
                        logits = model(seq)
                        nxt = logits[0, -1].argmax().item()
                        seq = torch.cat([seq, torch.tensor([[nxt]], device=DEVICE)], dim=1)
                    pred_tokens = seq[0, 30:45].tolist()
                    pred = decode_sum(pred_tokens)
                    if pred == a + b:
                        correct += 1
                    if correct + (256 - pairs.index((a,b)) - 1) < 0:
                        break
                if correct >= n:
                    break
            # simpler: just do n random
        # redo cleanly
        correct = 0
        with torch.no_grad():
            for _ in range(n):
                a = random.randint(0, 99999999999999)
                b = random.randint(0, 99999999999999)
                inp = encode_pair(a, b)
                seq = torch.tensor([inp], dtype=torch.long, device=DEVICE)
                for pos in range(15):
                    logits = model(seq)
                    nxt = logits[0, -1].argmax().item()
                    seq = torch.cat([seq, torch.tensor([[nxt]], device=DEVICE)], dim=1)
                pred_tokens = seq[0, 30:45].tolist()
                pred = decode_sum(pred_tokens)
                if pred == a + b:
                    correct += 1
        model.train()
        return correct / n

    # Training loop
    model.train()
    for step in range(1, max_steps + 1):
        x = generate_batch(batch_size)  # (B, 45)
        # Input is x[:, :-1], target is x[:, 1:]
        # But we want loss only on output positions 30..44
        # x has 45 tokens, we feed 0..43 and predict 1..44
        # Output positions in target: indices 30..44 correspond to input positions 29..43
        inp = x[:, :-1]  # (B, 44)
        tgt = x[:, 1:]   # (B, 44)
        logits = model(inp)  # (B, 44, vocab)
        # loss on positions 29..43 (which predict tokens 30..44)
        loss = F.cross_entropy(logits[:, 29:44].reshape(-1, VOCAB), tgt[:, 29:44].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 500 == 0:
            print(f"step {step} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.6f}")

        if step % eval_every == 0:
            acc = evaluate(500)
            print(f"  eval acc @ step {step}: {acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                print(f"  *** new best {best_acc:.4f} ***")
                torch.save(model.state_dict(), "/workspace/best.pt")
            if acc >= 0.99:
                print(f"  Reached 99% at step {step}!")
                # save and continue a bit to solidify
                torch.save(model.state_dict(), "/workspace/best.pt")

        if step % save_every == 0:
            torch.save(model.state_dict(), f"/workspace/ckpt_{step}.pt")

        if best_acc >= 0.995 and step > 30000:
            # good enough, can stop early
            pass

    # Final eval
    if os.path.exists("/workspace/best.pt"):
        model.load_state_dict(torch.load("/workspace/best.pt", map_location=DEVICE))
    acc = evaluate(2000)
    print(f"FINAL acc: {acc:.4f} best: {best_acc:.4f}")
    torch.save(model.state_dict(), "/workspace/best.pt")
    return model, best_acc

if __name__ == "__main__":
    train()
