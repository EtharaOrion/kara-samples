import torch
from torch import nn
import torch.nn.functional as F

_D = 15
_VALUES = __VALUES__
_SHAPES = __SHAPES__


class Block(nn.Module):
    def __init__(self, shared):
        super().__init__()
        self.n1 = nn.LayerNorm(16)
        self.qkv = nn.Linear(16, 48)
        self.proj = nn.Linear(16, 16)
        self.n2 = nn.LayerNorm(16)
        self.mlp = shared

    def forward(self, x, mask):
        z = self.n1(x)
        q, k, v = self.qkv(z).chunk(3, -1)
        q = q.view(q.shape[0], q.shape[1], 2, 8).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], 2, 8).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], 2, 8).transpose(1, 2)
        a = F.softmax((q @ k.transpose(-2, -1)) * (8 ** -0.5) + mask, -1)
        x = x + self.proj((a @ v).transpose(1, 2).reshape_as(x))
        return x + self.mlp(self.n2(x))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.digit = nn.Embedding(10, 16)
        self.role = nn.Embedding(3, 16)
        shared = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 16))
        self.blocks = nn.ModuleList((Block(shared), Block(shared)))
        self.norm = nn.LayerNorm(16)
        self.head = nn.Linear(16, 10)

    def forward(self, tokens):
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device)
        x = self.digit(tokens) + self.role(positions.remainder(3))
        permitted = (positions[:, None] >= positions[None, :]) & ((positions[:, None] - positions[None, :]) < 6)
        mask = torch.where(permitted, 0.0, float('-inf')).view(1, 1, length, length)
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.norm(x))


def build_model():
    model = Model()
    state = model.state_dict()
    with torch.no_grad():
        for name, values in _VALUES.items():
            state[name].copy_(torch.tensor(values).reshape(_SHAPES[name]))
    model.eval()
    return model, {'architecture': 'two-block local causal transformer', 'digits': 15}


def _digits(value):
    result = []
    for _ in range(_D):
        result.append(value % 10)
        value //= 10
    return result


def add(model, a: int, b: int) -> int:
    ad = _digits(a)
    bd = _digits(b)
    tokens = []
    output = []
    with torch.no_grad():
        for column in range(_D):
            tokens.extend((ad[column], bd[column]))
            source = torch.tensor(tokens, dtype=torch.long).view(1, -1)
            digit = int(model(source)[0, -1].argmax())
            tokens.append(digit)
            output.append(str(digit))
    return int(''.join(reversed(output)))
