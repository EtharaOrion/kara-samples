import torch
from torch import nn

D = 20
H = 4
DH = 5
N = 25


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.ParameterList([nn.Parameter(torch.empty(D, D)) for _ in range(2)])
        self.o = nn.ParameterList([nn.Parameter(torch.empty(D, D)) for _ in range(2)])
        self.k_free = nn.Parameter(torch.empty(DH, D - DH))
        self.v_free = nn.Parameter(torch.empty(DH, D - DH))

    def forward(self, x, layer):
        batch, length, _ = x.shape
        q = torch.nn.functional.linear(x, self.q[layer]).view(batch, length, H, DH).transpose(1, 2)
        eye = torch.eye(DH, dtype=x.dtype, device=x.device)
        k = torch.nn.functional.linear(x, torch.cat((eye, self.k_free), dim=1))
        v = torch.nn.functional.linear(x, torch.cat((eye, self.v_free), dim=1))
        mixed = torch.nn.functional.scaled_dot_product_attention(
            q, k[:, None], v[:, None], is_causal=True, enable_gqa=True
        )
        return torch.nn.functional.linear(mixed.transpose(1, 2).reshape(batch, length, D), self.o[layer])


class AdditionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_free = nn.Parameter(torch.empty(11, 19))
        self.pos_free = nn.Parameter(torch.empty(23, 2))
        self.pos_basis = nn.Parameter(torch.empty(2, D))
        self.attn_norm = nn.LayerNorm(D)
        self.attn = Attention()
        self.ff1_weight = nn.Parameter(torch.empty(2, 19))
        self.ff1_bias = nn.Parameter(torch.empty(2))
        self.ff2_weight = nn.Parameter(torch.empty(D - 1, 2))
        self.head_weight = nn.Parameter(torch.empty(9, 19))
        self.head_bias = nn.Parameter(torch.empty(9))

    def forward(self, tokens):
        zero = torch.zeros((11, 1), dtype=self.token_free.dtype, device=self.token_free.device)
        token = torch.cat((zero, self.token_free), dim=1)
        eye = torch.eye(2, dtype=self.pos_free.dtype, device=self.pos_free.device)
        positions = torch.cat((eye, self.pos_free), dim=0) @ self.pos_basis
        x = torch.nn.functional.embedding(tokens, token) + positions[:tokens.shape[1]]
        for layer in range(2):
            x = x + self.attn(self.attn_norm(x), layer)
            z = torch.nn.functional.layer_norm(x, (D,))[..., :19]
            hidden = torch.nn.functional.gelu(torch.nn.functional.linear(z, self.ff1_weight, self.ff1_bias))
            x = x + torch.nn.functional.pad(torch.nn.functional.linear(hidden, self.ff2_weight), (1, 0))
        z = torch.nn.functional.layer_norm(x, (D,))[..., :19]
        logits = torch.nn.functional.linear(z, self.head_weight, self.head_bias)
        return torch.cat((logits, torch.zeros_like(logits[..., :1])), dim=-1)


_TRAINED_STATE = STATE_LITERAL


def build_model():
    model = AdditionTransformer()
    with torch.no_grad():
        for parameter, value in zip(model.parameters(), _TRAINED_STATE):
            parameter.copy_(torch.tensor(value, dtype=parameter.dtype).reshape(parameter.shape))
    model.eval()
    return model, {"architecture": "causal grouped-query digit transformer", "digits": 8}


def add(model, a: int, b: int) -> int:
    left = [ord(c) - 48 for c in reversed(str(a))]
    right = [ord(c) - 48 for c in reversed(str(b))]
    sequence = []
    for x, y in zip(left, right):
        sequence.extend((x, y))
    sequence.append(10)
    with torch.no_grad():
        for _ in range(9):
            tokens = torch.tensor(sequence, dtype=torch.long).unsqueeze(0)
            digit = int(model(tokens)[0, -1].argmax())
            sequence.append(digit)
    return int("".join(str(d) for d in reversed(sequence[-9:])))
