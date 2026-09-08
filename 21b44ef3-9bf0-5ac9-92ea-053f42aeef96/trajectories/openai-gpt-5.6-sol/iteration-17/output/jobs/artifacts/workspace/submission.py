import torch
from torch import nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        d = 20
        self.token = nn.Embedding(11, d)
        self.pos_left = nn.Parameter(torch.empty(25, 7))
        self.pos_right = nn.Parameter(torch.empty(7, d))
        self.pass_embedding = nn.Parameter(torch.empty(4, d))
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, 4)
        self.ff2 = nn.Linear(4, d)
        self.final_norm = nn.LayerNorm(d)
        self.output = nn.Linear(d, 10, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.token.weight, std=0.02)
        nn.init.normal_(self.pos_left, std=0.02)
        nn.init.normal_(self.pos_right, std=0.02)
        nn.init.normal_(self.pass_embedding, std=0.02)

    def forward(self, tokens):
        x = self.token(tokens) + self.pos_left.matmul(self.pos_right)
        for offset in self.pass_embedding:
            x = x + offset
            y = self.norm1(x)
            x = x + self.attn(y, y, y, need_weights=False)[0]
            x = x + self.ff2(torch.nn.functional.gelu(self.ff1(self.norm2(x))))
        return self.output(self.final_norm(x[:, 16:25]))


_STATE = None


def build_model():
    model = Model()
    if _STATE is not None:
        model.load_state_dict(_STATE)
    model.eval()
    return model, {"architecture": "shared iterative joint-output transformer", "digit_order": "least-significant-first"}


def add(model, a: int, b: int) -> int:
    digits = []
    for _ in range(8):
        digits.extend((a % 10, b % 10))
        a //= 10
        b //= 10
    tokens = torch.tensor([digits + [10] * 9], dtype=torch.long, device=next(model.parameters()).device)
    with torch.no_grad():
        predicted = model(tokens).argmax(-1)[0].tolist()
    return int("".join(str(d) for d in reversed(predicted)))
