import math

import torch
from torch import nn
import torch.nn.functional as F


class AdditionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(12, 2)
        self.qkv = nn.Linear(2, 6, bias=False)
        self.projection = nn.Linear(2, 2, bias=False)
        self.ff1 = nn.Linear(2, 8)
        self.ff2 = nn.Linear(8, 2)
        self.output = nn.Linear(2, 1)

    def forward(self, tokens):
        x = self.embedding(tokens)
        query, key, value = self.qkv(x).chunk(3, dim=-1)
        attention = F.softmax(query @ key.transpose(-2, -1) / math.sqrt(2), dim=-1)
        x = x + self.projection(attention @ value)
        x = x + self.ff2(F.silu(self.ff1(x)))
        return self.output(x[:, 2]).squeeze(-1)


_WEIGHTS = {'embedding.weight': [-1.1376090049743652, 0.8630433082580566, -0.694071888923645, -0.12391350418329239, -0.3712376356124878, 0.10423767566680908, 0.11256604641675949, -1.285683274269104, 0.3235735595226288, 0.14500899612903595, 0.7438042759895325, -0.5342994332313538, 1.1059380769729614, -0.5880676507949829, 1.6484355926513672, -2.504279613494873, 1.8600870370864868, -0.9203830361366272, 2.1488869190216064, -0.10293141007423401, 0.632377028465271, -0.4161830544471741, 0.8001331090927124, -0.4996287524700165], 'qkv.weight': [0.08560382574796677, -0.08639592677354813, -0.09082253277301788, -0.4475117027759552, 0.36352142691612244, -0.3068321645259857, -0.38876157999038696, 0.17576107382774353, -0.90336012840271, 0.6732826828956604, 1.3166829347610474, 0.4835999011993408], 'projection.weight': [-0.18482719361782074, 1.5560733079910278, 0.8450965881347656, -0.6680983901023865], 'ff1.weight': [-0.15489932894706726, 0.4439607858657837, 0.8529818654060364, -0.7829912304878235, 1.036790132522583, -0.4274991750717163, -1.2070109844207764, 0.7018871903419495, -0.14685218036174774, -0.7941642999649048, 0.39678990840911865, -0.7830847501754761, 0.86629319190979, 0.07315003871917725, 0.7779969573020935, 0.06656020134687424], 'ff1.bias': [-0.05183824896812439, 0.44302114844322205, 0.8162128329277039, 0.29100653529167175, 0.33467844128608704, 0.7897145748138428, 0.2246304303407669, 0.8748230338096619], 'ff2.weight': [0.08053752034902573, -0.797090470790863, -0.7723528742790222, 0.669269859790802, 0.13264425098896027, -0.724806547164917, -0.36708346009254456, -0.7534905672073364, 0.0980054959654808, -0.5885090827941895, -0.5590003132820129, 0.7257468104362488, -0.4301242232322693, -0.029211146757006645, -0.37699785828590393, -0.27879244089126587], 'ff2.bias': [-0.5510322451591492, -0.07850231975317001], 'output.weight': [-0.5790552496910095, -0.683193564414978], 'output.bias': [0.37459468841552734]}
_SHAPES = {'embedding.weight': [12, 2], 'qkv.weight': [6, 2], 'projection.weight': [2, 2], 'ff1.weight': [8, 2], 'ff1.bias': [8], 'ff2.weight': [2, 8], 'ff2.bias': [2], 'output.weight': [1, 2], 'output.bias': [1]}


def build_model():
    model = AdditionTransformer()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(torch.tensor(_WEIGHTS[name]).reshape(_SHAPES[name]))
    model.eval()
    metadata = {"architecture": "column-autoregressive transformer", "parameters": 85}
    return model, metadata


def add(model, a: int, b: int) -> int:
    device = next(model.parameters()).device
    carry = 0
    place = 1
    result = 0
    with torch.no_grad():
        for _ in range(8):
            tokens = torch.tensor([[a % 10, b % 10, 10 + carry]], device=device)
            column_total = int(model(tokens).round().clamp(0, 19).item())
            result += (column_total % 10) * place
            carry = column_total // 10
            a //= 10
            b //= 10
            place *= 10
    return result + carry * place
