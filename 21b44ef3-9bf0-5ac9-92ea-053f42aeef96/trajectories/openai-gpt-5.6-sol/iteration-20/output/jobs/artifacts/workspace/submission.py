import torch
from torch import nn


class AdditionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token = nn.Embedding(11, 20)
        self.pos_left = nn.Parameter(torch.empty(25, 3))
        self.pos_right = nn.Parameter(torch.empty(3, 20))
        self.norm_attn = nn.LayerNorm(20)
        self.norm_ff = nn.LayerNorm(20)
        self.q = nn.ModuleList([nn.Linear(20, 20, bias=False) for _ in range(2)])
        self.k = nn.Linear(20, 5, bias=False)
        self.v = nn.Linear(20, 5, bias=False)
        self.out = nn.ModuleList([nn.Linear(20, 20, bias=False) for _ in range(2)])
        self.ff1 = nn.Linear(20, 4, bias=False)
        self.ff2 = nn.Linear(4, 20, bias=False)
        self.final_norm = nn.LayerNorm(20)
        self.head = nn.Linear(20, 10, bias=False)
        self.register_buffer("causal", torch.triu(torch.ones(25, 25, dtype=torch.bool), 1), persistent=False)
        self.reset_parameters()

    def reset_parameters(self):
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.normal_(parameter, std=0.15)
            else:
                nn.init.ones_(parameter)
        nn.init.zeros_(self.norm_attn.bias)
        nn.init.zeros_(self.norm_ff.bias)
        nn.init.zeros_(self.final_norm.bias)

    def forward(self, tokens):
        length = tokens.shape[1]
        positions = self.pos_left[:length] @ self.pos_right
        x = self.token(tokens) + positions
        mask = self.causal[:length, :length]
        for layer in range(2):
            z = self.norm_attn(x)
            batch = z.shape[0]
            q = self.q[layer](z).view(batch, length, 4, 5).transpose(1, 2)
            k = self.k(z)
            v = self.v(z)
            scores = torch.einsum("bhid,bjd->bhij", q, k) * 0.4472135954999579
            scores = scores.masked_fill(mask, -torch.inf)
            attended = torch.einsum("bhij,bjd->bhid", scores.softmax(dim=-1), v)
            attended = attended.transpose(1, 2).reshape(batch, length, 20)
            x = x + self.out[layer](attended)
            x = x + self.ff2(torch.nn.functional.gelu(self.ff1(self.norm_ff(x))))
        return self.head(self.final_norm(x))


_WEIGHTS = None


def build_model():
    model = AdditionTransformer()
    if _WEIGHTS is not None:
        offset = 0
        with torch.no_grad():
            for parameter in model.parameters():
                count = parameter.numel()
                parameter.copy_(torch.tensor(_WEIGHTS[offset:offset + count], dtype=parameter.dtype).view_as(parameter))
                offset += count
    model.eval()
    return model, {"architecture": "rank-3 grouped-query causal digit transformer", "digits": 8}


def add(model, a: int, b: int) -> int:
    left = f"{a:08d}"[::-1]
    right = f"{b:08d}"[::-1]
    prefix = []
    for x, y in zip(left, right):
        prefix.extend((ord(x) - 48, ord(y) - 48))
    tokens = torch.tensor([prefix + [10]], dtype=torch.long, device=next(model.parameters()).device)
    digits = []
    with torch.no_grad():
        for _ in range(9):
            digit = int(model(tokens)[0, -1].argmax())
            digits.append(str(digit))
            tokens = torch.cat((tokens, torch.tensor([[digit]], device=tokens.device)), dim=1)
    return int("".join(reversed(digits)))
