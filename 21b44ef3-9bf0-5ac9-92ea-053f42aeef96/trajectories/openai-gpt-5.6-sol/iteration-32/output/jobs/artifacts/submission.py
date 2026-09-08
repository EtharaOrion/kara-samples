import torch
from torch import nn
import torch.nn.functional as F


class TinyAdder(nn.Module):
    def __init__(self):
        super().__init__()
        self.token = nn.Parameter(torch.empty(11, 20))
        self.pos_free = nn.Parameter(torch.empty(23, 2))
        self.pos_basis = nn.Parameter(torch.empty(2, 20))
        self.q = nn.Parameter(torch.empty(2, 20, 20))
        self.o = nn.Parameter(torch.empty(2, 20, 20))
        self.k = nn.Parameter(torch.empty(5, 20))
        self.v = nn.Parameter(torch.empty(5, 20))
        self.norm_weight = nn.Parameter(torch.ones(20))
        self.norm_bias = nn.Parameter(torch.zeros(20))
        self.ff1_weight = nn.Parameter(torch.empty(2, 20))
        self.ff1_bias = nn.Parameter(torch.zeros(2))
        self.ff2_weight = nn.Parameter(torch.empty(20, 2))
        self.head_weight = nn.Parameter(torch.empty(10, 20))
        self.head_bias = nn.Parameter(torch.zeros(10))
        self.register_buffer("causal", torch.tril(torch.ones(25, 25, dtype=torch.bool)), persistent=False)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.ndim > 1:
                nn.init.normal_(p, std=0.02)

    def positions(self):
        c = torch.empty(25, 2, device=self.pos_free.device, dtype=self.pos_free.dtype)
        c[:17] = self.pos_free[:17]
        c[17] = torch.tensor((1.0, 0.0), device=c.device, dtype=c.dtype)
        c[18:22] = self.pos_free[17:21]
        c[22] = torch.tensor((0.0, 1.0), device=c.device, dtype=c.dtype)
        c[23:] = self.pos_free[21:]
        return c @ self.pos_basis

    def forward(self, tokens):
        length = tokens.shape[1]
        x = F.embedding(tokens, self.token) + self.positions()[:length]
        mask = self.causal[:length, :length]
        for layer in range(2):
            z = F.layer_norm(x, (20,), self.norm_weight, self.norm_bias)
            q = F.linear(z, self.q[layer]).view(-1, length, 4, 5).transpose(1, 2)
            k = F.linear(z, self.k)
            v = F.linear(z, self.v)
            scores = torch.einsum("bhtd,bsd->bhts", q, k) * 0.4472135954999579
            scores = scores.masked_fill(~mask, -torch.inf)
            attn = scores.softmax(-1)
            mixed = torch.einsum("bhts,bsd->bhtd", attn, v).transpose(1, 2).reshape(-1, length, 20)
            x = x + F.linear(mixed, self.o[layer])
            z = F.layer_norm(x, (20,))
            x = x + F.linear(F.gelu(F.linear(z, self.ff1_weight, self.ff1_bias)), self.ff2_weight)
        return F.linear(F.layer_norm(x, (20,)), self.head_weight, self.head_bias)


_WEIGHTS = None


def build_model():
    model = TinyAdder()
    if _WEIGHTS is not None:
        with torch.no_grad():
            for parameter, values in zip(model.parameters(), _WEIGHTS):
                parameter.copy_(torch.tensor(values, dtype=parameter.dtype).reshape_as(parameter))
    model.eval()
    return model, {"architecture": "causal grouped-query digit transformer", "digits": 8}


def add(model, a: int, b: int) -> int:
    da = str(a)[::-1]
    db = str(b)[::-1]
    tokens = []
    for xa, xb in zip(da, db):
        tokens.extend((ord(xa) - 48, ord(xb) - 48))
    tokens.append(10)
    device = next(model.parameters()).device
    with torch.no_grad():
        for _ in range(9):
            x = torch.tensor([tokens], dtype=torch.long, device=device)
            tokens.append(int(model(x)[0, -1].argmax()))
    return int("".join(str(d) for d in tokens[-9:][::-1]))
