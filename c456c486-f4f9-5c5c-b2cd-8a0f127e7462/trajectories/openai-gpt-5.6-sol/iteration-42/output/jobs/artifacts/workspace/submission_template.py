import math
import torch
from torch import nn
import torch.nn.functional as F

D, H = 9, 3

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D)
        self.proj = nn.Linear(D, D)

    def forward(self, x):
        z = self.norm(x)
        q, k, v = self.qkv(z).chunk(3, -1)
        n = x.shape[1]
        q = q.view(-1, n, H, 3).transpose(1, 2)
        k = k.view(-1, n, H, 3).transpose(1, 2)
        v = v.view(-1, n, H, 3).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(3)
        mask = torch.ones(n, n, dtype=torch.bool, device=x.device).tril()
        z = (scores.masked_fill(~mask, float('-inf')).softmax(-1) @ v)
        return x + self.proj(z.transpose(1, 2).reshape(-1, n, D))

class AdditionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.a_emb = nn.Embedding(11, D)
        self.b_emb = nn.Embedding(11, D)
        self.y_emb = nn.Embedding(10, D)
        self.pos = nn.Parameter(torch.empty(29, 2))
        self.attn1 = Attention()
        self.ffnorm = nn.LayerNorm(D)
        self.ff1 = nn.Linear(D, 1)
        self.ff2 = nn.Linear(1, D)
        self.attn2 = Attention()
        self.final_norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, 10)

    def forward(self, ad, bd, prefix):
        x = torch.cat((self.a_emb(ad) + self.b_emb(bd), self.y_emb(prefix)), 1)
        x = x + F.pad(self.pos[:x.shape[1]], (0, 7)).unsqueeze(0)
        x = self.attn1(x)
        x = x + self.ff2(F.gelu(self.ff1(self.ffnorm(x))))
        return self.head(self.final_norm(self.attn2(x)))

_STATE = __STATE__

def build_model():
    model = AdditionTransformer()
    model.load_state_dict({k: torch.tensor(v) for k, v in _STATE.items()})
    model.eval()
    return model, {'architecture': 'aligned autoregressive causal transformer', 'parameters': 1266}

def add(model, a: int, b: int) -> int:
    ad, bd = [], []
    for _ in range(14):
        ad.append(a % 10); a //= 10
        bd.append(b % 10); b //= 10
    ad.append(10); bd.append(10)
    device = next(model.parameters()).device
    at = torch.tensor([ad], dtype=torch.long, device=device)
    bt = torch.tensor([bd], dtype=torch.long, device=device)
    out = torch.empty((1, 0), dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(15):
            logits = model(at, bt, out)
            token = logits[:, 14 + out.shape[1]].argmax(-1, keepdim=True)
            out = torch.cat((out, token), 1)
    value = 0
    for digit in reversed(out[0].tolist()):
        value = value * 10 + digit
    return value
