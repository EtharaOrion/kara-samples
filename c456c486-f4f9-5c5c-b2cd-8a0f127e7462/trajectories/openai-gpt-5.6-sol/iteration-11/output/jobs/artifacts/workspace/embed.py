import torch
ck=torch.load('/workspace/final.pt',weights_only=True)
state=ck['model']
# Enough decimal precision to round-trip float32 exactly.
def lit(t):
    return repr(t.detach().cpu().flatten().tolist())
items=[]
for name,t in state.items():
    if name=='causal': continue
    items.append(f"        {name!r}: {lit(t)},")
weights='\n'.join(items)
text='''import torch
from torch import nn
import torch.nn.functional as F


class Adder(nn.Module):
    def __init__(self):
        super().__init__()
        self.width = 10
        self.digit = nn.Embedding(10, 10)
        self.position = nn.Parameter(torch.empty(15, 10))
        self.norm1 = nn.LayerNorm(10)
        self.qkv = nn.Linear(10, 30)
        self.project = nn.Linear(10, 10)
        self.norm2 = nn.LayerNorm(10)
        self.up = nn.Linear(10, 32)
        self.down = nn.Linear(32, 10)
        self.final_norm = nn.LayerNorm(10)
        self.output = nn.Linear(10, 10)
        self.register_buffer("causal", torch.tril(torch.ones(15, 15, dtype=torch.bool)), persistent=False)

    def forward(self, left, right):
        x = self.digit(left) + self.digit(right) + self.position
        for _ in range(6):
            z = self.norm1(x)
            q, k, v = self.qkv(z).chunk(3, dim=-1)
            q = q.view(-1, 15, 2, 5).transpose(1, 2)
            k = k.view(-1, 15, 2, 5).transpose(1, 2)
            v = v.view(-1, 15, 2, 5).transpose(1, 2)
            scores = (q @ k.transpose(-2, -1)) * 0.4472135954999579
            scores = scores.masked_fill(~self.causal, -torch.inf)
            attended = (scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(-1, 15, 10)
            x = x + self.project(attended)
            x = x + self.down(F.gelu(self.up(self.norm2(x))))
        return self.output(self.final_norm(x))


def _trained_weights():
    return {
WEIGHTS
    }


def build_model():
    model = Adder()
    state = model.state_dict()
    for name, values in _trained_weights().items():
        state[name].copy_(torch.tensor(values).reshape(state[name].shape))
    model.eval()
    return model, {
        "architecture": "shared-block causal parallel transformer",
        "parameters": 1542,
        "training": "random full 14-digit pairs with carry and pattern curriculum",
    }


def add(model, a: int, b: int) -> int:
    left = []
    right = []
    for _ in range(14):
        left.append(a % 10)
        right.append(b % 10)
        a //= 10
        b //= 10
    left.append(0)
    right.append(0)
    with torch.no_grad():
        logits = model(torch.tensor([left]), torch.tensor([right]))
    digits = logits.argmax(dim=-1)[0].tolist()
    return int("".join(str(digit) for digit in reversed(digits)))
'''.replace('WEIGHTS',weights)
open('/workspace/submission.py','w').write(text)
print(len(text),len(state))
