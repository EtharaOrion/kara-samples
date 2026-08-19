import torch
from torch import nn


class TinyAdder(nn.Module):
    """One-head recurrent transformer trained on digit-addition transitions."""

    def __init__(self):
        super().__init__()
        self.digit = nn.Embedding(10, 1)
        self.cls = nn.Parameter(torch.empty(1))
        self.carry_role = nn.Parameter(torch.empty(1))
        self.qkv = nn.Linear(1, 3, bias=False)
        self.out = nn.Linear(1, 1)

    def forward(self, triples):
        tokens = self.digit(triples)
        carry = tokens[:, 2:3] + self.carry_role
        tokens = torch.cat((self.cls.expand(len(triples), 1, 1), tokens[:, :2], carry), dim=1)
        query, key, value = self.qkv(tokens).chunk(3, dim=-1)
        attention = (query[:, :1] @ key.transpose(1, 2)).softmax(dim=-1)
        return self.out((attention @ value)[:, 0])[:, 0]


def build_model():
    model = TinyAdder()
    with torch.no_grad():
        model.cls.copy_(torch.tensor([-0.072577565908432]))
        model.carry_role.copy_(torch.tensor([-0.35722485184669495]))
        model.digit.weight.copy_(torch.tensor([
            [-0.2731322646141052], [0.3886472284793854],
            [1.0130418539047241], [1.6391336917877197],
            [2.256282091140747], [2.864173412322998],
            [3.4637081623077393], [4.054599285125732],
            [4.636523723602295], [5.208891868591309],
        ]))
        model.qkv.weight.copy_(torch.tensor([
            [0.28583890199661255],
            [-0.7531923651695251],
            [2.6703505516052246],
        ]))
        model.out.weight.copy_(torch.tensor([[2.3608388900756836]]))
        model.out.bias.copy_(torch.tensor([1.827791690826416]))
    model.eval()
    return model, {
        "architecture": "recurrent one-head digit transformer",
        "parameters": 17,
        "training_states": 200,
    }


def add(model, a: int, b: int) -> int:
    triples = []
    carry = 0
    place = 1
    answer = 0
    with torch.no_grad():
        for _ in range(14):
            triples.append((a % 10, b % 10, carry))
            local_sum = int(torch.round(model(torch.tensor([triples[-1]], device=model.cls.device))).item())
            answer += local_sum % 10 * place
            carry = local_sum // 10
            place *= 10
            a //= 10
            b //= 10
        answer += carry * place
    return answer
