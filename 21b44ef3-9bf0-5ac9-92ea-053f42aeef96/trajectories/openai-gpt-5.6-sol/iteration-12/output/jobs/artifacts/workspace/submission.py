import torch
from torch import nn

class AdderTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        d = 20
        self.token = nn.Embedding(11, d)
        self.pos_left = nn.Parameter(torch.empty(25, 7))
        self.pos_right = nn.Parameter(torch.empty(7, d))
        self.pass_left = nn.Parameter(torch.empty(2, 1))
        self.pass_right = nn.Parameter(torch.empty(1, d))
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, 4)
        self.ff2 = nn.Linear(4, d)
        self.final_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 10, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.pos_left, std=.02)
        nn.init.normal_(self.pos_right, std=.02)
        nn.init.normal_(self.pass_left, std=.02)
        nn.init.normal_(self.pass_right, std=.02)

    def forward(self, tokens):
        n = tokens.shape[1]
        x = self.token(tokens) + (self.pos_left[:n] @ self.pos_right)
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=tokens.device), 1)
        passes = self.pass_left @ self.pass_right
        for step in range(2):
            y = self.norm1(x + passes[step])
            x = x + self.attn(y, y, y, attn_mask=mask, need_weights=False)[0]
            x = x + self.ff2(torch.nn.functional.gelu(self.ff1(self.norm2(x))))
        return self.head(self.final_norm(x))


def build_model():
    model = AdderTransformer()
    model.eval()
    return model, {"architecture": "recurrent causal transformer", "parameters": 2741}


def add(model, a: int, b: int) -> int:
    left = f"{a:08d}"[::-1]
    right = f"{b:08d}"[::-1]
    tokens = []
    for x, y in zip(left, right):
        tokens.extend((ord(x) - 48, ord(y) - 48))
    tokens.append(10)
    device = next(model.parameters()).device
    sequence = torch.tensor([tokens], dtype=torch.long, device=device)
    digits = []
    with torch.no_grad():
        for _ in range(9):
            digit = int(model(sequence)[0, -1].argmax().item())
            digits.append(chr(48 + digit))
            sequence = torch.cat((sequence, torch.tensor([[digit]], device=device)), dim=1)
    return int("".join(reversed(digits)))
