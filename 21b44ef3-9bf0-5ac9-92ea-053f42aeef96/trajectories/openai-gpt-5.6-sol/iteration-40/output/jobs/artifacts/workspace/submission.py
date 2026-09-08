import torch
from torch import nn
import torch.nn.functional as F


class AdditionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        d = 20
        self.token = nn.Embedding(11, d)
        self.pos_a = nn.Parameter(torch.empty(25, 2))
        self.pos_b = nn.Parameter(torch.empty(2, d))
        self.attn_norm = nn.LayerNorm(d)
        self.key = nn.Linear(d, 5, bias=False)
        self.value = nn.Linear(d, 5, bias=False)
        self.query = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(2)])
        self.project = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(2)])
        self.ff_norm = nn.LayerNorm(d)
        self.ff_in = nn.Linear(d, 2, bias=False)
        self.ff_out = nn.Linear(2, d, bias=False)
        self.final_norm = nn.LayerNorm(d)
        self.output = nn.Linear(d, 10, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.ndim > 1:
                nn.init.normal_(p, std=0.02)
            else:
                nn.init.ones_(p)
        nn.init.zeros_(self.pos_a)

    def forward(self, tokens):
        length = tokens.shape[1]
        x = self.token(tokens) + (self.pos_a @ self.pos_b)[:length]
        mask = torch.ones(length, length, device=x.device, dtype=torch.bool).triu(1)
        for layer in range(2):
            z = self.attn_norm(x)
            q = self.query[layer](z).view(z.shape[0], length, 4, 5).transpose(1, 2)
            k = self.key(z).unsqueeze(1)
            v = self.value(z).unsqueeze(1)
            scores = (q @ k.transpose(-2, -1)) * (5.0 ** -0.5)
            scores = scores.masked_fill(mask, float('-inf'))
            attention = scores.softmax(dim=-1)
            context = (attention @ v).transpose(1, 2).reshape(z.shape[0], length, 20)
            x = x + self.project[layer](context)
            x = x + self.ff_out(F.gelu(self.ff_in(self.ff_norm(x))))
        return self.output(self.final_norm(x))


_TRAINED_STATE = None


def build_model():
    model = AdditionTransformer()
    if _TRAINED_STATE is not None:
        model.load_state_dict(_TRAINED_STATE)
    model.eval()
    return model, {'architecture': 'causal grouped-query digit transformer', 'digit_order': 'least-significant-first'}


def add(model, a: int, b: int) -> int:
    sa = f'{a:08d}'[::-1]
    sb = f'{b:08d}'[::-1]
    sequence = []
    for da, db in zip(sa, sb):
        sequence.extend((ord(da) - 48, ord(db) - 48))
    sequence.append(10)
    device = next(model.parameters()).device
    with torch.no_grad():
        for _ in range(9):
            tokens = torch.tensor([sequence], dtype=torch.long, device=device)
            digit = int(model(tokens)[0, -1].argmax().item())
            sequence.append(digit)
    return int(''.join(str(d) for d in sequence[-1:-10:-1]))
