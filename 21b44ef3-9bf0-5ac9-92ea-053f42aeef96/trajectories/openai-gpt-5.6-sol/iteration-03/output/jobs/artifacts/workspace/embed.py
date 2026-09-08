import sys
from pathlib import Path
sys.path.insert(0, '/usr/local/lib/python3.11/dist-packages')
import torch

checkpoint = torch.load('/workspace/candidate_w20f20_final.pt', map_location='cpu', weights_only=False)
state = checkpoint['state']
config = checkpoint['config']

def literal(t):
    return repr(t.flatten().tolist())

entries = []
for name, tensor in state.items():
    entries.append(f"    {name!r}: ({literal(tensor)}, {tuple(tensor.shape)!r}),")

text = '''import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, width, heads, ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, ff), nn.GELU(), nn.Linear(ff, width))

    def forward(self, x, mask):
        y = self.norm1(x)
        x = x + self.attn(y, y, y, attn_mask=mask, need_weights=False)[0]
        return x + self.ff(self.norm2(x))


class AdditionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        width = WIDTH
        self.token = nn.Embedding(11, width)
        self.position = nn.Embedding(25, width)
        self.blocks = nn.ModuleList([Block(width, HEADS, FF) for _ in range(2)])
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, 10, bias=False)

    def forward(self, tokens):
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device)
        x = self.token(tokens) + self.position(positions)
        mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=tokens.device), diagonal=1)
        for block in self.blocks:
            x = block(x, mask)
        return self.output(self.norm(x))


WIDTH = %d
HEADS = %d
FF = %d
_STATE = {
%s
}


def build_model():
    model = AdditionTransformer()
    with torch.no_grad():
        current = model.state_dict()
        for name, (values, shape) in _STATE.items():
            current[name].copy_(torch.tensor(values).reshape(shape))
    model.eval()
    return model, {"architecture": "two-layer causal digit transformer", "width": WIDTH, "parameters": %d}


def add(model, a: int, b: int) -> int:
    left = str(a)[::-1]
    right = str(b)[::-1]
    tokens = []
    for x, y in zip(left, right):
        tokens.extend((ord(x) - 48, ord(y) - 48))
    tokens.append(10)
    device = next(model.parameters()).device
    with torch.inference_mode():
        for _ in range(9):
            sequence = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
            tokens.append(int(model(sequence)[0, -1].argmax()))
    return int("".join(str(x) for x in tokens[-9:][::-1]))
''' % (config['width'], config['heads'], config['ff'], '\n'.join(entries), checkpoint['parameters'])
Path('/workspace/submission.py').write_text(text)
