import itertools
from pathlib import Path
import torch

from train import AdditionTransformer

state = torch.load('/workspace/width2_polish.pt', map_location='cpu', weights_only=True)

# Fold the shared attention LayerNorm scale into every following projection.
gamma = state['attn_norm.weight']
state['attn_norm.bias'] = state['attn_norm.bias'] / gamma
for name in ('q.0.weight', 'q.1.weight', 'k.weight', 'v.weight'):
    state[name] = state[name] * gamma

# Fold two affine LayerNorms into their following linear maps.
state['ff_in.bias'] = state['ff_in.weight'] @ state['ff_norm.bias']
state['ff_in.weight'] = state['ff_in.weight'] * state['ff_norm.weight']
state['output.bias'] = state['output.weight'] @ state['final_norm.bias']
state['output.weight'] = state['output.weight'] * state['final_norm.weight']

# Fix a well-conditioned basis for the rank-two positional factorization.
L, R = state['pos_left'], state['pos_right']
pair = max(itertools.combinations(range(25), 2), key=lambda p: abs(float(torch.det(L[list(p)]))))
A = L[list(pair)]
L = L @ torch.linalg.inv(A)
R = A @ R
# Per-position all-ones translations disappear in every LayerNorm.
R = R - R[:, -1:]
state['pos_left_free'] = L[[i for i in range(25) if i not in pair]]
state['pos_right_free'] = R[:, :-1]

# The same LayerNorm translation invariance fixes one coordinate per token.
E = state['token.weight']
E = E - E[:, -1:]
state['token_free'] = E[:, :-1]

# Independently fix bases in shared key and value spaces.
def fix_kv(name):
    W = state[name + '.weight']
    cols = max(itertools.combinations(range(20), 5), key=lambda c: abs(float(torch.det(W[:, list(c)]))))
    B = W[:, list(cols)]
    Wn = torch.linalg.solve(B, W)
    state[name + '_free'] = Wn[:, [i for i in range(20) if i not in cols]]
    return cols, B

kcols, KA = fix_kv('k')
for layer in range(2):
    W = state[f'q.{layer}.weight'].clone()
    for head in range(4):
        W[head*5:(head+1)*5] = KA.T @ W[head*5:(head+1)*5]
    state[f'q.{layer}.weight'] = W
vcols, VA = fix_kv('v')
for layer in range(2):
    W = state[f'o.{layer}.weight'].clone()
    for head in range(4):
        W[:, head*5:(head+1)*5] = W[:, head*5:(head+1)*5] @ VA
    # Any per-token all-ones residual is erased by every subsequent LayerNorm.
    W = W - W[-1:]
    state[f'o.{layer}.free'] = W[:-1]
W = state['ff_out.weight']
W = W - W[-1:]
state['ff_out.free'] = W[:-1]

# Non-affine LayerNorm outputs sum to zero, fixing one weight per output row.
W = state['ff_in.weight']
W = W - W[:, -1:]
state['ff_in_free'] = W[:, :-1]
W = state['output.weight']
W = W - W[:, -1:]
b = state['output.bias']
# Logits are defined only up to a shared affine scalar; class nine is reference.
W = W - W[9:10]
b = b - b[9]
state['output_free'] = W[:9, :-1]
state['output_bias_free'] = b[:9]

keep = {
    'token_free', 'pos_left_free', 'pos_right_free',
    'q.0.weight', 'q.1.weight', 'k_free', 'v_free', 'o.0.free', 'o.1.free',
    'attn_norm.bias', 'ff_in_free', 'ff_in.bias',
    'ff_out.free', 'output_free', 'output_bias_free'
}
state = {k: v for k, v in state.items() if k in keep}
rename = {
    'q.0.weight': 'q.0', 'q.1.weight': 'q.1',
    'o.0.free': 'o.0', 'o.1.free': 'o.1', 'attn_norm.bias': 'attn_norm_bias',
    'ff_in.bias': 'ff_in_bias', 'ff_out.free': 'ff_out'
}
state = {rename.get(k, k): v for k, v in state.items()}

header = '''import torch
from torch import nn
import torch.nn.functional as F


class AdditionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_free = nn.Parameter(torch.empty(11, 19))
        self.pos_left_free = nn.Parameter(torch.empty(23, 2))
        self.pos_right_free = nn.Parameter(torch.empty(2, 19))
        self.q = nn.ParameterList([nn.Parameter(torch.empty(20, 20)) for _ in range(2)])
        self.k_free = nn.Parameter(torch.empty(5, 15))
        self.v_free = nn.Parameter(torch.empty(5, 15))
        self.o = nn.ParameterList([nn.Parameter(torch.empty(19, 20)) for _ in range(2)])
        self.attn_norm_bias = nn.Parameter(torch.empty(20))
        self.ff_in_free = nn.Parameter(torch.empty(2, 19))
        self.ff_in_bias = nn.Parameter(torch.empty(2))
        self.ff_out = nn.Parameter(torch.empty(19, 2))
        self.output_free = nn.Parameter(torch.empty(9, 19))
        self.output_bias_free = nn.Parameter(torch.empty(9))
        self.register_buffer("causal", torch.triu(torch.ones(25, 25, dtype=torch.bool), 1), persistent=False)

    def _position(self):
        left = torch.empty(25, 2, device=self.pos_left_free.device, dtype=self.pos_left_free.dtype)
        left[list(POS_FIXED)] = torch.eye(2, device=left.device, dtype=left.dtype)
        left[list(POS_OTHER)] = self.pos_left_free
        right = F.pad(self.pos_right_free, (0, 1))
        return left @ right

    @staticmethod
    def _projection(free, fixed):
        weight = torch.empty(5, 20, device=free.device, dtype=free.dtype)
        weight[:, list(fixed)] = torch.eye(5, device=free.device, dtype=free.dtype)
        weight[:, [i for i in range(20) if i not in fixed]] = free
        return weight

    def forward(self, tokens):
        length = tokens.shape[1]
        embedding = F.pad(self.token_free, (0, 1))
        x = F.embedding(tokens, embedding) + self._position()[:length]
        mask = self.causal[:length, :length]
        k_weight = self._projection(self.k_free, K_FIXED)
        v_weight = self._projection(self.v_free, V_FIXED)
        for layer in range(2):
            z = F.layer_norm(x, (20,), bias=self.attn_norm_bias)
            q = F.linear(z, self.q[layer]).view(tokens.shape[0], length, 4, 5).transpose(1, 2)
            k = F.linear(z, k_weight)
            v = F.linear(z, v_weight)
            scores = torch.einsum("bhtd,bsd->bhts", q, k) * 0.4472135954999579
            attention = scores.masked_fill(mask, -torch.inf).softmax(-1)
            context = torch.einsum("bhts,bsd->bhtd", attention, v)
            residual = F.pad(F.linear(context.transpose(1, 2).reshape(tokens.shape[0], length, 20), self.o[layer]), (0, 1))
            x = x + residual
            normalized = F.layer_norm(x, (20,))
            ff_weight = F.pad(self.ff_in_free, (0, 1))
            residual = F.pad(F.linear(F.gelu(F.linear(normalized, ff_weight, self.ff_in_bias)), self.ff_out), (0, 1))
            x = x + residual
        normalized = F.layer_norm(x, (20,))
        output_weight = F.pad(self.output_free, (0, 1, 0, 1))
        output_bias = F.pad(self.output_bias_free, (0, 1))
        return F.linear(normalized, output_weight, output_bias)

'''
header += f'POS_FIXED = {tuple(pair)!r}\nPOS_OTHER = {tuple(i for i in range(25) if i not in pair)!r}\nK_FIXED = {tuple(kcols)!r}\nV_FIXED = {tuple(vcols)!r}\n\n'
values=[]
for name,t in state.items():
    flat=','.join(format(x,'.9g') for x in t.float().reshape(-1).tolist())
    values.append(f"    {name!r}: torch.tensor([{flat}]).reshape{tuple(t.shape)},")
weights='_TRAINED_STATE = {\n'+'\n'.join(values)+'\n}\n\n'
footer='''def build_model():
    model = AdditionTransformer()
    model.load_state_dict(_TRAINED_STATE)
    model.eval()
    return model, {"architecture": "causal grouped-query digit transformer", "digit_order": "least-significant-first"}


def add(model, a: int, b: int) -> int:
    sa = str(a)[::-1]
    sb = str(b)[::-1]
    tokens = []
    for i in range(8):
        tokens.extend((ord(sa[i]) - 48, ord(sb[i]) - 48))
    tokens.append(10)
    device = next(model.parameters()).device
    sequence = torch.tensor([tokens], dtype=torch.long, device=device)
    digits = []
    with torch.no_grad():
        for _ in range(9):
            digit = int(model(sequence)[0, -1].argmax().item())
            digits.append(str(digit))
            sequence = torch.cat((sequence, torch.tensor([[digit]], device=device)), dim=1)
    return int("".join(reversed(digits)))
'''
Path('/workspace/submission_compressed.py').write_text(header+weights+footer)
print('fixed', pair, kcols, vcols)
print('parameters', sum(x.numel() for x in state.values()))
