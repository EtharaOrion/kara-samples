import os
import torch
from submission import AdditionTransformer as Original

D,H,HD=20,4,5
state=torch.load('/workspace/final.pt',map_location='cpu',weights_only=True)['model']

# Fold affine FF LayerNorms into their following linear maps.
for i in range(2):
    w=state[f'ff1.{i}.weight']
    gamma=state[f'ff_norm.{i}.weight']
    beta=state[f'ff_norm.{i}.bias']
    state[f'ff1.{i}.weight']=w*gamma[None,:]
    state[f'ff1.{i}.bias']=w@beta
    del state[f'ff_norm.{i}.weight'],state[f'ff_norm.{i}.bias']
# Fold final affine LayerNorm and replace class 9 by the zero reference logit.
w=state['head.weight']; gamma=state['final_norm.weight']; beta=state['final_norm.bias']
w=w*gamma[None,:]; bias=w.new_tensor(state['head.weight']@beta) # original W times beta
# Correction: beta contribution uses original head W, while scaled matrix uses gamma.
orig=torch.load('/workspace/final.pt',map_location='cpu',weights_only=True)['model']['head.weight']
bias=orig@beta
state['head.weight']=w[:9]-w[9:10]
state['head.bias']=bias[:9]-bias[9]
del state['final_norm.weight'],state['final_norm.bias']

header='''import torch
from torch import nn

D = 20
H = 4
HD = 5


class AdditionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token = nn.Embedding(11, D)
        self.position = nn.Parameter(torch.empty(25, D))
        self.attn_norm = nn.ModuleList([nn.LayerNorm(D) for _ in range(2)])
        self.ff_norm = nn.ModuleList([nn.LayerNorm(D, elementwise_affine=False) for _ in range(2)])
        self.q = nn.ModuleList([nn.Linear(D, D, bias=False) for _ in range(2)])
        self.k = nn.ModuleList([nn.Linear(D, D, bias=False) for _ in range(2)])
        self.v = nn.ModuleList([nn.Linear(D, D, bias=False) for _ in range(2)])
        self.o = nn.ModuleList([nn.Linear(D, D, bias=False) for _ in range(2)])
        self.ff1 = nn.ModuleList([nn.Linear(D, 4, bias=True) for _ in range(2)])
        self.ff2 = nn.ModuleList([nn.Linear(4, D, bias=False) for _ in range(2)])
        self.final_norm = nn.LayerNorm(D, elementwise_affine=False)
        self.head = nn.Linear(D, 9, bias=True)

    def forward(self, tokens):
        length = tokens.shape[1]
        x = self.token(tokens) + self.position[:length]
        mask = torch.ones(length, length, device=x.device, dtype=torch.bool).tril()
        for layer in range(2):
            z = self.attn_norm[layer](x)
            q = self.q[layer](z).view(*z.shape[:2], H, HD).transpose(1, 2)
            k = self.k[layer](z).view(*z.shape[:2], H, HD).transpose(1, 2)
            v = self.v[layer](z).view(*z.shape[:2], H, HD).transpose(1, 2)
            scores = (q @ k.transpose(-2, -1)) * (HD ** -0.5)
            scores = scores.masked_fill(~mask, -torch.inf)
            attended = (scores.softmax(-1) @ v).transpose(1, 2).reshape_as(x)
            x = x + self.o[layer](attended)
            x = x + self.ff2[layer](torch.nn.functional.gelu(self.ff1[layer](self.ff_norm[layer](x))))
        logits = self.head(self.final_norm(x))
        return torch.cat((logits, logits.new_zeros(*logits.shape[:-1], 1)), dim=-1)


_TRAINED_STATE = '''
literal="{\n"+",\n".join(repr(k)+": torch.tensor("+repr(v.tolist())+")" for k,v in state.items())+"\n}"
tail='''


def build_model():
    model = AdditionTransformer()
    model.load_state_dict(_TRAINED_STATE)
    model.eval()
    return model, {"architecture": "two-layer causal autoregressive transformer", "digit_order": "least-significant-first"}


def add(model, a: int, b: int) -> int:
    da = [ord(c) - 48 for c in str(a)[::-1]]
    db = [ord(c) - 48 for c in str(b)[::-1]]
    prefix = []
    for x, y in zip(da, db):
        prefix.extend((x, y))
    tokens = prefix + [10]
    output = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for _ in range(9):
            digit = int(model(torch.tensor([tokens], dtype=torch.long, device=device))[0, -1].argmax())
            output.append(digit)
            tokens.append(digit)
    return int("".join(str(x) for x in output[::-1]))
'''
path='/workspace/submission.py'
with open(path+'.new','w') as f:f.write(header+literal+tail)
os.replace(path+'.new',path)
print('exported',len(header+literal+tail))
