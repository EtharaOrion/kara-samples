from pathlib import Path
import torch

state=torch.load('/workspace/model_final.pt',map_location='cpu',weights_only=True)
entries=[]
for name,t in state.items():
    vals=', '.join(repr(float(x)) for x in t.reshape(-1))
    entries.append(f'        {name!r}: torch.tensor([{vals}]).reshape{tuple(t.shape)!r},')
weights='\n'.join(entries)
source=f'''import torch
from torch import nn
import torch.nn.functional as F


class AdditionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        width = 12
        self.digit = nn.Embedding(10, width)
        self.position = nn.Parameter(torch.empty(15, width))
        nn.init.normal_(self.position, std=0.02)
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.attention_out = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.mlp_in = nn.Linear(width, 24)
        self.mlp_out = nn.Linear(24, width)
        self.final_norm = nn.LayerNorm(width)
        self.classifier = nn.Linear(width, 10)
        self.register_buffer("causal_mask", torch.triu(torch.ones(15, 15, dtype=torch.bool), 1), persistent=False)

    def forward(self, left, right):
        x = self.digit(left) + self.digit(right) + self.position
        for _ in range(6):
            z = self.norm1(x)
            q, k, v = self.qkv(z).chunk(3, dim=-1)
            q = q.reshape(-1, 15, 2, 6).transpose(1, 2)
            k = k.reshape(-1, 15, 2, 6).transpose(1, 2)
            v = v.reshape(-1, 15, 2, 6).transpose(1, 2)
            scores = q @ k.transpose(-2, -1) / (6 ** 0.5)
            scores = scores.masked_fill(self.causal_mask, float("-inf"))
            context = F.softmax(scores, dim=-1) @ v
            context = context.transpose(1, 2).reshape(-1, 15, 12)
            x = x + self.attention_out(context)
            x = x + self.mlp_out(F.gelu(self.mlp_in(self.norm2(x))))
        return self.classifier(self.final_norm(x))


def build_model():
    model = AdditionTransformer()
    learned = {{
{weights}
    }}
    model.load_state_dict(learned)
    model.eval()
    return model, {{
        "architecture": "shared recurrent causal parallel transformer",
        "operand_digits": 14,
        "output_digits": 15,
        "width": 12,
        "heads": 2,
        "mlp_width": 24,
        "recurrent_rounds": 6,
        "training_steps": 33000,
    }}


def add(model, a: int, b: int) -> int:
    if not (0 <= a <= 99_999_999_999_999 and 0 <= b <= 99_999_999_999_999):
        raise ValueError("operands must be 14-digit nonnegative integers")
    device = next(model.parameters()).device
    left_digits = [int(c) for c in str(a).zfill(14)[::-1]]
    right_digits = [int(c) for c in str(b).zfill(14)[::-1]]
    left_digits.append(0)
    right_digits.append(0)
    left = torch.tensor([left_digits], dtype=torch.long, device=device)
    right = torch.tensor([right_digits], dtype=torch.long, device=device)
    with torch.no_grad():
        digits = model(left, right).argmax(dim=-1)[0].tolist()
    return int("".join(str(d) for d in digits[::-1]))
'''
Path('/workspace/submission.py').write_text(source)
