import importlib.util
import itertools
from pathlib import Path

import torch
import torch.nn.functional as F

from submission import AdditionTransformer
from train import encode, uniform

ROOT = Path('/workspace')
DEVICE = 'cuda'


def best_rows(matrix, count):
    best = None
    for rows in itertools.combinations(range(matrix.shape[0]), count):
        block = matrix[list(rows)]
        score = float(torch.linalg.slogdet(block.float()).logabsdet)
        if best is None or score > best[0]:
            best = (score, rows)
    return list(best[1])


state = torch.load(ROOT / 'polish_final.pt', map_location=DEVICE, weights_only=True)
base = AdditionTransformer(ff_width=2).to(DEVICE)
base.load_state_dict(state)
base.eval()

# Fold affine LayerNorms into their following linear maps.
ff1 = state['ff_weight'][:, None] * state['ff1']
ff1_bias = state['ff_bias'] @ state['ff1']
classifier = state['final_weight'][:, None] * state['classifier']
classifier_bias = state['final_bias'] @ state['classifier']

# Remove residual-stream all-ones directions, which every later LayerNorm ignores.
token = state['token'] - state['token'][:, -1:]
pos_right = state['pos_right'] - state['pos_right'][:, -1:]
o = state['o'] - state['o'][:, :, -1:]
ff2 = state['ff2'] - state['ff2'][:, -1:]

# Gauge-fix positional rank basis.
pos_rows = best_rows(state['pos_left'], 2)
pa = state['pos_left'][pos_rows]
pos_left = state['pos_left'] @ torch.linalg.inv(pa)
pos_right = pa @ pos_right

# Independently gauge-fix key and value latent bases.
k_rows = best_rows(state['k'], 5)
ka = state['k'][k_rows]
k = state['k'] @ torch.linalg.inv(ka)
q = state['q'].clone()
for layer in range(2):
    for head in range(4):
        q[layer, :, head * 5:(head + 1) * 5] = q[layer, :, head * 5:(head + 1) * 5] @ ka.T

v_rows = best_rows(state['v'], 5)
va = state['v'][v_rows]
v = state['v'] @ torch.linalg.inv(va)
for layer in range(2):
    for head in range(4):
        block = o[layer, head * 5:(head + 1) * 5]
        o[layer, head * 5:(head + 1) * 5] = va @ block

# Non-affine LayerNorm outputs sum to zero, so common row offsets vanish.
ff1 = ff1 - ff1[-1:]
classifier = classifier - classifier[-1:]
# Fix the last output-logit class as the zero reference.
classifier_bias = classifier_bias - classifier_bias[-1]
classifier = classifier - classifier[:, -1:]

pos_free = torch.stack([pos_left[i] for i in range(25) if i not in pos_rows])
k_free = torch.stack([k[i] for i in range(20) if i not in k_rows])
v_free = torch.stack([v[i] for i in range(20) if i not in v_rows])
values = [
    token[:, :19], pos_free, pos_right[:, :19], q, o[:, :, :19], k_free, v_free,
    state['att_weight'], state['att_bias'], ff1[:19], ff1_bias, ff2[:, :19],
    classifier[:19, :9], classifier_bias[:9],
]

HEADER = '''import torch
from torch import nn
import torch.nn.functional as F

_POS_ROWS = %r
_K_ROWS = %r
_V_ROWS = %r
_STATE = %s


class AdditionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        shapes = [(11,19),(23,2),(2,19),(2,20,20),(2,20,19),(15,5),(15,5),(20,),(20,),(19,2),(2,),(2,19),(19,9),(9,)]
        names = ('token_free','pos_free','pos_right_free','q','o_free','k_free','v_free','att_weight','att_bias','ff1_free','ff1_bias','ff2_free','classifier_free','classifier_bias_free')
        for name, shape, value in zip(names, shapes, _STATE):
            setattr(self, name, nn.Parameter(torch.tensor(value, dtype=torch.float32).reshape(shape)))

    @staticmethod
    def _insert_rows(free, fixed_rows, size):
        rows = []
        at = 0
        eye = torch.eye(len(fixed_rows), dtype=free.dtype, device=free.device)
        lookup = {row: j for j, row in enumerate(fixed_rows)}
        for i in range(size):
            if i in lookup:
                rows.append(eye[lookup[i]])
            else:
                rows.append(free[at])
                at += 1
        return torch.stack(rows)

    def forward(self, tokens):
        zero_col = self.token_free.new_zeros(self.token_free.shape[0], 1)
        token = torch.cat((self.token_free, zero_col), 1)
        pos_left = self._insert_rows(self.pos_free, _POS_ROWS, 25)
        pos_right = torch.cat((self.pos_right_free, self.pos_right_free.new_zeros(2, 1)), 1)
        k = self._insert_rows(self.k_free, _K_ROWS, 20)
        v = self._insert_rows(self.v_free, _V_ROWS, 20)
        n = tokens.shape[1]
        x = F.embedding(tokens, token) + pos_left[:n] @ pos_right
        mask = torch.ones(n, n, dtype=torch.bool, device=x.device).triu(1)
        for layer in range(2):
            z = F.layer_norm(x, (20,), self.att_weight, self.att_bias)
            q = (z @ self.q[layer]).view(-1, n, 4, 5).transpose(1, 2)
            keys = (z @ k).unsqueeze(1)
            vals = (z @ v).unsqueeze(1)
            scores = (q @ keys.transpose(-2, -1)) * 0.4472135954999579
            scores = scores.masked_fill(mask, -1e4)
            attended = (scores.softmax(-1) @ vals).transpose(1, 2).reshape(-1, n, 20)
            out = torch.cat((self.o_free[layer], self.o_free.new_zeros(20, 1)), 1)
            x = x + attended @ out
            ff1 = torch.cat((self.ff1_free, self.ff1_free.new_zeros(1, 2)), 0)
            hidden = F.gelu(F.layer_norm(x, (20,)) @ ff1 + self.ff1_bias)
            ff2 = torch.cat((self.ff2_free, self.ff2_free.new_zeros(2, 1)), 1)
            x = x + hidden @ ff2
        classifier_rows = torch.cat((self.classifier_free, self.classifier_free.new_zeros(1, 9)), 0)
        classifier = torch.cat((classifier_rows, classifier_rows.new_zeros(20, 1)), 1)
        bias = torch.cat((self.classifier_bias_free, self.classifier_bias_free.new_zeros(1)))
        return F.layer_norm(x, (20,)) @ classifier + bias


def build_model():
    model = AdditionTransformer()
    model.eval()
    return model, {"name": "compressed causal grouped-query digit transformer", "digit_order": "least-significant-first"}


def add(model, a: int, b: int) -> int:
    sa, sb = str(a)[::-1], str(b)[::-1]
    tokens = []
    for da, db in zip(sa, sb):
        tokens.extend((ord(da) - 48, ord(db) - 48))
    tokens.append(10)
    device = next(model.parameters()).device
    with torch.no_grad():
        for _ in range(9):
            x = torch.tensor([tokens], dtype=torch.long, device=device)
            digit = int(model(x)[0, -1].argmax().item())
            tokens.append(digit)
    return int("".join(str(d) for d in tokens[-9:][::-1]))
'''

def literal(t):
    return repr(t.detach().cpu().flatten().tolist())

state_literal = '[\n' + ',\n'.join(literal(x) for x in values) + '\n]'
source = HEADER % (pos_rows, k_rows, v_rows, state_literal)
(ROOT / 'submission_compressed.py').write_text(source)
print('rows', pos_rows, k_rows, v_rows)
print('stored parameters', sum(x.numel() for x in values))

spec = importlib.util.spec_from_file_location('compressed', ROOT / 'submission_compressed.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
compressed, _ = module.build_model()
compressed.cuda().eval()
print('registered parameters', sum(p.numel() for p in compressed.parameters()))
with torch.no_grad():
    a, b = uniform(4096)
    x, _ = encode(a, b)
    old = base(x)
    new = compressed(x)
    adjusted = old - old[:, :, -1:]
    print('max logit delta', float((new - adjusted).abs().max()))
    print('argmax disagreements', int((new.argmax(-1) != old.argmax(-1)).sum()))
''
