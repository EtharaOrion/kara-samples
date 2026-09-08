from pathlib import Path
import math
import torch
from torch import nn


class StandardModel(nn.Module):
    def __init__(self, ff_width):
        super().__init__()
        self.token = nn.Embedding(11, 20)
        self.pos_left = nn.Parameter(torch.empty(25, 2))
        self.pos_right = nn.Parameter(torch.empty(2, 20))
        self.attn_norm = nn.LayerNorm(20)
        self.queries = nn.ModuleList([nn.Linear(20, 20, bias=False) for _ in range(2)])
        self.keys = nn.Linear(20, 5, bias=False)
        self.values = nn.Linear(20, 5, bias=False)
        self.outputs = nn.ModuleList([nn.Linear(20, 20, bias=False) for _ in range(2)])
        self.ff_norm = nn.LayerNorm(20)
        self.ff_in = nn.Linear(20, ff_width, bias=False)
        self.ff_out = nn.Linear(ff_width, 20, bias=False)
        self.final_norm = nn.LayerNorm(20)
        self.classifier = nn.Linear(20, 10, bias=False)

    def forward(self, tokens):
        length = tokens.shape[1]
        x = self.token(tokens) + (self.pos_left @ self.pos_right)[:length]
        mask = torch.ones(length, length, device=tokens.device, dtype=torch.bool).triu(1)
        for query, output in zip(self.queries, self.outputs):
            z = self.attn_norm(x)
            q = query(z).view(tokens.shape[0], length, 4, 5).transpose(1, 2)
            k, v = self.keys(z), self.values(z)
            scores = torch.einsum('bhtd,bsd->bhts', q, k) / math.sqrt(5.0)
            scores = scores.masked_fill(mask, -torch.inf)
            context = torch.einsum('bhts,bsd->bhtd', scores.softmax(-1), v)
            x = x + output(context.transpose(1, 2).reshape(tokens.shape[0], length, 20))
            x = x + self.ff_out(torch.nn.functional.gelu(self.ff_in(self.ff_norm(x))))
        return self.classifier(self.final_norm(x))


def best_square_columns(weight, size):
    import itertools
    best, best_det = None, -1.0
    for cols in itertools.combinations(range(weight.shape[1]), size):
        det = torch.linalg.det(weight[:, cols]).abs().item()
        if det > best_det:
            best, best_det = cols, det
    return list(best)


def literal(name, tensor):
    return f"        self.{name} = nn.Parameter(torch.tensor({tensor.detach().cpu().float().tolist()!r}))\n"

m = StandardModel(2)
m.load_state_dict(torch.load('/workspace/final.pt', map_location='cpu', weights_only=True))
s = m.state_dict()

# Fold affine LayerNorms into their following linear maps.
ff_w = s['ff_in.weight'] * s['ff_norm.weight']
ff_b = s['ff_in.weight'] @ s['ff_norm.bias']
clf_w = s['classifier.weight'] * s['final_norm.weight']
clf_b = s['classifier.weight'] @ s['final_norm.bias']

# Position GL(2) gauge, choosing the best-conditioned pair of rows.
left, right = s['pos_left'], s['pos_right']
best_pair, best_det = None, -1.0
for i in range(25):
    for j in range(i + 1, 25):
        d = torch.linalg.det(left[[i, j]]).abs().item()
        if d > best_det: best_pair, best_det = [i, j], d
A = left[best_pair]
left = left @ torch.linalg.inv(A)
right = A @ right
# LayerNorm ignores a row-dependent uniform translation; fix last basis column.
right = right - right[:, -1:]
left_free = left[[i for i in range(25) if i not in best_pair]]
right_free = right[:, :19]

# LayerNorm ignores token-dependent uniform translations.
token = s['token.weight'] - s['token.weight'][:, -1:]
token_free = token[:, :19]

queries = [s[f'queries.{i}.weight'].clone() for i in range(2)]
outputs = [s[f'outputs.{i}.weight'].clone() for i in range(2)]

# Independent GL(5) gauges for shared keys and values.
kcols = best_square_columns(s['keys.weight'], 5)
KA = s['keys.weight'][:, kcols]
key = torch.linalg.solve(KA, s['keys.weight'])
for layer in range(2):
    q = queries[layer].view(4, 5, 20)
    queries[layer] = torch.einsum('ij,hjk->hik', KA.T, q).reshape(20, 20)

vcols = best_square_columns(s['values.weight'], 5)
VA = s['values.weight'][:, vcols]
value = torch.linalg.solve(VA, s['values.weight'])
for layer in range(2):
    w = outputs[layer]
    outputs[layer] = torch.cat([w[:, h*5:(h+1)*5] @ VA for h in range(4)], dim=1)

# Residual-stream uniform directions are invisible to every subsequent LayerNorm.
outputs = [w - w[-1:] for w in outputs]
ff_out = s['ff_out.weight'] - s['ff_out.weight'][-1:]
# Argmax is invariant to a shared reference-class logit.
clf_w = clf_w - clf_w[-1:]
clf_b = clf_b - clf_b[-1]

header = '''import math
import torch
from torch import nn


class AdditionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
'''
body = ''
body += literal('token_free', token_free)
body += literal('pos_left_free', left_free)
body += literal('pos_right_free', right_free)
body += literal('attn_weight', s['attn_norm.weight'])
body += literal('attn_bias', s['attn_norm.bias'])
for i in range(2): body += literal(f'query_{i}', queries[i])
body += literal('key_free', key[:, [i for i in range(20) if i not in kcols]])
body += literal('value_free', value[:, [i for i in range(20) if i not in vcols]])
for i in range(2): body += literal(f'output_{i}_free', outputs[i][:-1])
body += literal('ff_in_weight', ff_w)
body += literal('ff_in_bias', ff_b)
body += literal('ff_out_free', ff_out[:-1])
body += literal('classifier_weight_free', clf_w[:-1])
body += literal('classifier_bias_free', clf_b[:-1])

constants = f'''        self.pos_fixed = {best_pair!r}
        self.key_fixed = {kcols!r}
        self.value_fixed = {vcols!r}

    @staticmethod
    def _rows(free, fixed, rows, cols):
        out = free.new_zeros(rows, cols)
        rest = [i for i in range(rows) if i not in fixed]
        out[rest] = free
        out[fixed] = torch.eye(len(fixed), device=free.device, dtype=free.dtype)
        return out

    def forward(self, tokens):
        batch, length = tokens.shape
        token = torch.cat((self.token_free, self.token_free.new_zeros(11, 1)), 1)
        left = self._rows(self.pos_left_free, self.pos_fixed, 25, 2)
        right = torch.cat((self.pos_right_free, self.pos_right_free.new_zeros(2, 1)), 1)
        x = torch.nn.functional.embedding(tokens, token) + (left @ right)[:length]
        mask = torch.ones(length, length, device=tokens.device, dtype=torch.bool).triu(1)
        key = self._rows(self.key_free.T, self.key_fixed, 20, 5).T
        value = self._rows(self.value_free.T, self.value_fixed, 20, 5).T
        for query, output_free in ((self.query_0, self.output_0_free), (self.query_1, self.output_1_free)):
            z = torch.nn.functional.layer_norm(x, (20,), self.attn_weight, self.attn_bias)
            q = torch.nn.functional.linear(z, query).view(batch, length, 4, 5).transpose(1, 2)
            k = torch.nn.functional.linear(z, key)
            v = torch.nn.functional.linear(z, value)
            scores = torch.einsum('bhtd,bsd->bhts', q, k) / math.sqrt(5.0)
            scores = scores.masked_fill(mask, -torch.inf)
            context = torch.einsum('bhts,bsd->bhtd', scores.softmax(-1), v)
            output = torch.cat((output_free, output_free.new_zeros(1, 20)), 0)
            x = x + torch.nn.functional.linear(context.transpose(1, 2).reshape(batch, length, 20), output)
            z = torch.nn.functional.layer_norm(x, (20,))
            hidden = torch.nn.functional.gelu(torch.nn.functional.linear(z, self.ff_in_weight, self.ff_in_bias))
            ff_out = torch.cat((self.ff_out_free, self.ff_out_free.new_zeros(1, 2)), 0)
            x = x + torch.nn.functional.linear(hidden, ff_out)
        z = torch.nn.functional.layer_norm(x, (20,))
        weight = torch.cat((self.classifier_weight_free, self.classifier_weight_free.new_zeros(1, 20)), 0)
        bias = torch.cat((self.classifier_bias_free, self.classifier_bias_free.new_zeros(1)), 0)
        return torch.nn.functional.linear(z, weight, bias)


def build_model():
    model = AdditionTransformer()
    model.eval()
    return model, {{'architecture': 'causal grouped-query digit transformer', 'digits': 8}}


def add(model, a: int, b: int) -> int:
    sa = f'{{a:08d}}'[::-1]
    sb = f'{{b:08d}}'[::-1]
    tokens = []
    for da, db in zip(sa, sb):
        tokens.extend((ord(da) - 48, ord(db) - 48))
    tokens.append(10)
    device = next(model.parameters()).device
    sequence = torch.tensor([tokens], dtype=torch.long, device=device)
    digits = []
    with torch.no_grad():
        for _ in range(9):
            digit = int(model(sequence)[0, -1].argmax())
            digits.append(digit)
            if len(digits) < 9:
                sequence = torch.cat((sequence, torch.tensor([[digit]], device=device)), dim=1)
    return int(''.join(str(d) for d in reversed(digits)))
'''
Path('/workspace/submission.py').write_text(header + body + constants)
print('fixed position rows', best_pair, 'key columns', kcols, 'value columns', vcols)
