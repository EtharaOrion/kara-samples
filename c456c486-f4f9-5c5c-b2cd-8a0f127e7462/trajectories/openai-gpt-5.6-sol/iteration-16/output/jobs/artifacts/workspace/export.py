import torch, sys
sys.path.insert(0, '/workspace')
import train
state=torch.load('/workspace/model.pt',map_location='cpu',weights_only=True)
# Token 11 was never used. Move '=' row 12 to 11 and remove the unused row.
state['token.weight']=torch.cat((state['token.weight'][:11],state['token.weight'][12:13]),0)

def literal(t):
    return repr(t.detach().float().reshape(-1).tolist())

header='''import torch
from torch import nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(16)
        self.qkv = nn.Linear(16, 48)
        self.proj = nn.Linear(16, 16)
        self.norm2 = nn.LayerNorm(16)
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 16)

    def forward(self, x, mask):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q = q.view(batch, length, 2, 8).transpose(1, 2)
        k = k.view(batch, length, 2, 8).transpose(1, 2)
        v = v.view(batch, length, 2, 8).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * 0.3535533905932738
        scores = scores.masked_fill(mask, -10000.0)
        y = (scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(batch, length, width)
        x = x + self.proj(y)
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


class AdditionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token = nn.Embedding(12, 16)
        self.position = nn.Parameter(torch.empty(45, 16))
        self.blocks = nn.ModuleList([Block(), Block(), Block()])
        self.norm = nn.LayerNorm(16)
        self.output = nn.Linear(16, 10)
        self.register_buffer("causal", torch.triu(torch.ones(45, 45, dtype=torch.bool), diagonal=1), persistent=False)

    def forward(self, tokens):
        length = tokens.shape[1]
        x = self.token(tokens) + self.position[:length]
        mask = self.causal[:length, :length]
        for block in self.blocks:
            x = block(x, mask)
        return self.output(self.norm(x))


'''
items=[]
for name,t in state.items():
    shape=list(t.shape)
    items.append('        '+repr(name)+': torch.tensor('+literal(t)+', dtype=torch.float32).reshape('+repr(shape)+')')
weights='''def build_model():
    model = AdditionTransformer()
    state = {
'''+',\n'.join(items)+'''\n    }
    model.load_state_dict(state)
    return model, {"architecture": "three-block causal autoregressive transformer", "parameters": 7786}


def _digits(value):
    text = str(value).zfill(14)
    return [ord(char) - 48 for char in reversed(text)]


def add(model, a: int, b: int) -> int:
    tokens = [10]
    left = _digits(a)
    right = _digits(b)
    for index in range(14):
        tokens.append(left[index])
        tokens.append(right[index])
    tokens.append(11)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for _ in range(15):
            source = torch.tensor([tokens], dtype=torch.long, device=device)
            tokens.append(int(model(source)[0, -1].argmax()))
    text = "".join(chr(tokens[index] + 48) for index in range(44, 29, -1))
    return int(text)
'''
open('/workspace/submission.py','w').write(header+weights)
