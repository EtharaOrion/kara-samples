import torch
from torch import nn
from torch.nn import functional as F


class AdditionTransformer(nn.Module):
    def __init__(self, ff_width=4, pos_rank=3):
        super().__init__()
        self.token = nn.Embedding(11, 20)
        self.pos_left = nn.Parameter(torch.randn(25, pos_rank) * 0.02)
        self.pos_right = nn.Parameter(torch.randn(pos_rank, 20) * 0.02)
        self.queries = nn.ModuleList([nn.Linear(20, 20, bias=False) for _ in range(2)])
        self.outputs = nn.ModuleList([nn.Linear(20, 20, bias=False) for _ in range(2)])
        self.key = nn.Linear(20, 5, bias=False)
        self.value = nn.Linear(20, 5, bias=False)
        self.attn_norm = nn.LayerNorm(20)
        self.ff_norm = nn.LayerNorm(20)
        self.ff_in = nn.Linear(20, ff_width, bias=False)
        self.ff_out = nn.Linear(ff_width, 20, bias=False)
        self.final_norm = nn.LayerNorm(20)
        self.classifier = nn.Linear(20, 10, bias=False)

    def forward(self, tokens):
        length = tokens.shape[1]
        x = self.token(tokens) + (self.pos_left[:length] @ self.pos_right)
        for query, output in zip(self.queries, self.outputs):
            y = self.attn_norm(x)
            q = query(y).view(y.shape[0], length, 4, 5).transpose(1, 2)
            k = self.key(y).unsqueeze(1)
            v = self.value(y).unsqueeze(1)
            attended = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, enable_gqa=True
            )
            attended = attended.transpose(1, 2).reshape(y.shape[0], length, 20)
            x = x + output(attended)
            x = x + self.ff_out(F.gelu(self.ff_in(self.ff_norm(x))))
        return self.classifier(self.final_norm(x))


_TRAINED_STATE = None


def build_model():
    model = AdditionTransformer(ff_width=4)
    if _TRAINED_STATE is not None:
        model.load_state_dict(_TRAINED_STATE)
    model.eval()
    return model, {
        "architecture": "causal grouped-query autoregressive digit transformer",
        "digit_order": "least-significant-first",
        "trained": _TRAINED_STATE is not None,
    }


def add(model, a: int, b: int) -> int:
    left = f"{a:08d}"[::-1]
    right = f"{b:08d}"[::-1]
    sequence = []
    for x, y in zip(left, right):
        sequence.extend((ord(x) - 48, ord(y) - 48))
    sequence.append(10)
    device = next(model.parameters()).device
    with torch.no_grad():
        for _ in range(9):
            tokens = torch.tensor([sequence], dtype=torch.long, device=device)
            digit = int(model(tokens)[0, -1].argmax().item())
            sequence.append(digit)
    return int("".join(str(d) for d in sequence[-9:][::-1]))
