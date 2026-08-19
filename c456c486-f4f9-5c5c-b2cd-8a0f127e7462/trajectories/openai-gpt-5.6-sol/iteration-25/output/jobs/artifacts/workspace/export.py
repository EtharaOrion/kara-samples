from pathlib import Path
import sys, torch
sys.path.insert(0,'/workspace')
from model import Adder
model=Adder(); model.load_state_dict(torch.load('/workspace/model_final.pt',map_location='cpu',weights_only=True))
values=torch.cat([p.detach().reshape(-1) for p in model.parameters()]).tolist()
prefix='''import torch
from torch import nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(10)
        self.qkv = nn.Linear(10, 30)
        self.proj = nn.Linear(10, 10)
        self.norm2 = nn.LayerNorm(10)
        self.fc1 = nn.Linear(10, 16)
        self.fc2 = nn.Linear(16, 10)

    def forward(self, x):
        z = self.norm1(x)
        q, k, v = self.qkv(z).chunk(3, -1)
        n, length, _ = q.shape
        q = q.view(n, length, 2, 5).transpose(1, 2)
        k = k.view(n, length, 2, 5).transpose(1, 2)
        v = v.view(n, length, 2, 5).transpose(1, 2)
        z = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(z.transpose(1, 2).reshape(n, length, 10))
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))

class Adder(nn.Module):
    def __init__(self):
        super().__init__()
        self.a_embed = nn.Embedding(11, 10)
        self.b_embed = nn.Embedding(11, 10)
        self.out_embed = nn.Embedding(11, 10)
        self.position = nn.Parameter(torch.empty(30, 10))
        self.blocks = nn.ModuleList([Block(), Block()])
        self.norm = nn.LayerNorm(10)
        self.head = nn.Linear(10, 10)

    def forward(self, a, b, previous):
        n = a.shape[0]
        source = self.a_embed(a) + self.b_embed(b)
        start = torch.full((n, 1), 10, dtype=torch.long, device=a.device)
        decoder = self.out_embed(torch.cat((start, previous), dim=1))
        x = torch.cat((source, decoder), dim=1)
        x = x + self.position[:x.shape[1]]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x[:, 15:]))

_WEIGHTS = '''
suffix='''

def build_model():
    model = Adder()
    values = torch.tensor(_WEIGHTS)
    offset = 0
    with torch.no_grad():
        for parameter in model.parameters():
            count = parameter.numel()
            parameter.copy_(values[offset:offset + count].view_as(parameter))
            offset += count
    return model.eval(), {"architecture": "aligned-column autoregressive transformer", "parameters": 2412}

def _digits(value):
    text = str(value).zfill(15)
    return [ord(character) - 48 for character in text[::-1]]

def add(model, a: int, b: int) -> int:
    device = next(model.parameters()).device
    left = torch.tensor([_digits(a)], dtype=torch.long, device=device)
    right = torch.tensor([_digits(b)], dtype=torch.long, device=device)
    generated = torch.empty((1, 0), dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(15):
            logits = model(left, right, generated)
            digit = logits[:, -1].argmax(-1, keepdim=True)
            generated = torch.cat((generated, digit), dim=1)
    characters = [chr(48 + int(digit)) for digit in generated[0].tolist()[::-1]]
    return int("".join(characters))
'''
Path('/workspace/submission.py').write_text(prefix+repr(values)+suffix)
print(len(values),Path('/workspace/submission.py').stat().st_size)
