import sys
sys.path = [p for p in sys.path if "openhands-venv" not in p]
sys.path.extend(["/usr/local/lib/python3.11/dist-packages", "/usr/lib/python3/dist-packages"])
import torch
state = torch.load('/workspace/scalar_2_8_3.pt', map_location='cpu', weights_only=True)
weights = {k: v.flatten().tolist() for k,v in state.items()}
shapes = {k: list(v.shape) for k,v in state.items()}
source = '''import math

import torch
from torch import nn
import torch.nn.functional as F


class AdditionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(12, 2)
        self.qkv = nn.Linear(2, 6, bias=False)
        self.projection = nn.Linear(2, 2, bias=False)
        self.ff1 = nn.Linear(2, 8)
        self.ff2 = nn.Linear(8, 2)
        self.output = nn.Linear(2, 1)

    def forward(self, tokens):
        x = self.embedding(tokens)
        query, key, value = self.qkv(x).chunk(3, dim=-1)
        attention = F.softmax(query @ key.transpose(-2, -1) / math.sqrt(2), dim=-1)
        x = x + self.projection(attention @ value)
        x = x + self.ff2(F.silu(self.ff1(x)))
        return self.output(x[:, 2]).squeeze(-1)


_WEIGHTS = __WEIGHTS__
_SHAPES = __SHAPES__


def build_model():
    model = AdditionTransformer()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(torch.tensor(_WEIGHTS[name]).reshape(_SHAPES[name]))
    model.eval()
    metadata = {"architecture": "column-autoregressive transformer", "parameters": 85}
    return model, metadata


def add(model, a: int, b: int) -> int:
    device = next(model.parameters()).device
    carry = 0
    place = 1
    result = 0
    with torch.no_grad():
        for _ in range(8):
            tokens = torch.tensor([[a % 10, b % 10, 10 + carry]], device=device)
            column_total = int(model(tokens).round().clamp(0, 19).item())
            result += (column_total % 10) * place
            carry = column_total // 10
            a //= 10
            b //= 10
            place *= 10
    return result + carry * place
'''
source = source.replace('__WEIGHTS__', repr(weights)).replace('__SHAPES__', repr(shapes))
open('/workspace/submission.py','w').write(source)
print('wrote',len(source),'bytes')
