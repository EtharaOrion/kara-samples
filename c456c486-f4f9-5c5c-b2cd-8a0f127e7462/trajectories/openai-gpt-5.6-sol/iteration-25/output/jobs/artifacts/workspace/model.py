import math
import torch
from torch import nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, width=10, hidden=16):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, width)

    def forward(self, x):
        z = self.norm1(x)
        q, k, v = self.qkv(z).chunk(3, dim=-1)
        batch, length, width = q.shape
        q = q.view(batch, length, 2, width // 2).transpose(1, 2)
        k = k.view(batch, length, 2, width // 2).transpose(1, 2)
        v = v.view(batch, length, 2, width // 2).transpose(1, 2)
        z = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        z = z.transpose(1, 2).reshape(batch, length, width)
        x = x + self.proj(z)
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
        nn.init.normal_(self.position, std=0.02)

    def forward(self, a, b, previous):
        batch = a.shape[0]
        source = self.a_embed(a) + self.b_embed(b)
        start = torch.full((batch, 1), 10, dtype=torch.long, device=a.device)
        decoder = self.out_embed(torch.cat((start, previous), dim=1))
        x = torch.cat((source, decoder), dim=1)
        x = x + self.position[:x.shape[1]]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x[:, 15:]))
