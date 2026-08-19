import torch
from torch import nn
import torch.nn.functional as F

_VALUES = __VALUES__
_SHAPES = __SHAPES__


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.LayerNorm(16)
        self.qkv = nn.Linear(16, 48)
        self.proj = nn.Linear(16, 16)
        self.n2 = nn.LayerNorm(16)
        self.mlp = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 16))

    def forward(self, x, mask):
        z = self.n1(x)
        q, k, v = self.qkv(z).chunk(3, -1)
        q = q.view(q.shape[0], q.shape[1], 2, 8).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], 2, 8).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], 2, 8).transpose(1, 2)
        attention = F.softmax((q @ k.transpose(-2, -1)) * (8 ** -0.5) + mask, -1)
        x = x + self.proj((attention @ v).transpose(1, 2).reshape_as(x))
        return x + self.mlp(self.n2(x))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.digit = nn.Embedding(10, 16)
        self.role = nn.Embedding(3, 16)
        self.blocks = nn.ModuleList((Block(), Block()))
        self.norm = nn.LayerNorm(16)
        self.head = nn.Linear(16, 10)

    def forward(self, tokens):
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device)
        x = self.digit(tokens) + self.role(positions.remainder(3))
        permitted = (positions[:, None] >= positions[None, :]) & ((positions[:, None] - positions[None, :]) < 6)
        mask = torch.where(permitted, 0.0, float('-inf')).view(1, 1, length, length)
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.norm(x))


def build_model():
    model = Model()
    state = model.state_dict()
    with torch.no_grad():
        for name, values in _VALUES.items():
            state[name].copy_(torch.tensor(values).reshape(_SHAPES[name]))
    model.eval()
    return model, {'architecture': 'two-block local causal transformer', 'decoding': 'symmetric beam reranking'}


def _digits(value):
    result = []
    for _ in range(15):
        result.append(value % 10)
        value //= 10
    return result


def _candidates(model, left, right):
    beams = [([], torch.tensor(0.0))]
    for column in range(15):
        sequences = [item[0] + [left[column], right[column]] for item in beams]
        scores = model(torch.tensor(sequences))[:, -1].log_softmax(-1)
        choices = []
        for row, (old, prior) in enumerate(beams):
            values, indices = scores[row].topk(5)
            for rank in range(5):
                score = torch.stack((prior, values[rank])).sum()
                choices.append((sequences[row] + [int(indices[rank])], score))
        choices.sort(key=lambda item: float(item[1]), reverse=True)
        beams = choices[:5]
    return [item[0][2::3] for item in beams]


def _score(model, left, right, outputs):
    rows = []
    for output in outputs:
        rows.append([token for column in range(15) for token in (left[column], right[column], output[column])])
    tokens = torch.tensor(rows)
    logits = model(tokens[:, :-1]).log_softmax(-1)
    positions = torch.arange(1, 44, 3)
    return logits[:, positions].gather(2, tokens[:, 2::3, None]).squeeze(-1).sum(1)


def add(model, a: int, b: int) -> int:
    left = _digits(a)
    right = _digits(b)
    with torch.no_grad():
        outputs = _candidates(model, left, right)
        outputs.extend(_candidates(model, right, left))
        scores = torch.stack((_score(model, left, right, outputs), _score(model, right, left, outputs))).sum(0)
        answer = outputs[int(scores.argmax())]
    value = 0
    for digit in reversed(answer):
        value = value * 10 + digit
    return value
