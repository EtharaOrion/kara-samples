import importlib.util
import os
import torch

PATH = "/workspace/submission.py"
spec = importlib.util.spec_from_file_location("source_submission", PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
model, _ = mod.build_model()
s = {k: v.detach().double() for k, v in model.state_dict().items()}

# Fold affine LayerNorms into their immediately following projections.
ff_w = s["ff1.weight"] * s["ff_norm.weight"][None, :]
ff_b = s["ff1.weight"].matmul(s["ff_norm.bias"])
head_w = s["head.weight"] * s["final_norm.weight"][None, :]
head_b = s["head.weight"].matmul(s["final_norm.bias"])

# Use class 9 as the zero-logit reference; softmax argmax is unchanged.
head_b = head_b[:9] - head_b[9]
head_w = head_w[:9] - head_w[9]

# Residual-stream gauge: scalar multiples of all-ones are erased by every LayerNorm.
def zero_last_output(weight):
    return (weight - weight[-1:, :])[:-1, :]

token = (s["token.weight"] - s["token.weight"][:, -1:])[:, :-1]
pos_right = (s["pos_right"] - s["pos_right"][:, -1:])[:, :-1]
o = [zero_last_output(s[f"o.{i}.weight"]) for i in range(2)]
ff2 = zero_last_output(s["ff2.weight"])

# Exact GL(2) gauge: make two well-conditioned positional coefficient rows identity.
left = s["pos_left"]
dets = []
for i in range(25):
    for j in range(i + 1, 25):
        dets.append((abs(torch.linalg.det(left[[i, j]])), i, j))
_, pi, pj = max(dets)
pa = left[[pi, pj]]
left = left.matmul(torch.linalg.inv(pa))
pos_right = pa.matmul(torch.cat((pos_right, torch.zeros(2, 1, dtype=torch.double)), 1))[:, :-1]
pos_free = left[[i for i in range(25) if i not in (pi, pj)]]

# Independent exact GL(5) gauges for shared keys and values.
def basis_gauge(weight):
    best = None
    # Deterministic greedy pivoted selection of five independent input columns.
    chosen = []
    for _ in range(5):
        candidate = max((float(torch.linalg.det(weight[:, chosen + [j]].T.matmul(weight[:, chosen + [j]]))) if chosen else float(weight[:, j].square().sum()), j)
                        for j in range(20) if j not in chosen)[1]
        # The simple criterion above is weak for intermediate rectangular matrices; use residual norm.
        if chosen:
            base = weight[:, chosen]
            residuals = []
            for j in range(20):
                if j in chosen:
                    continue
                coef = torch.linalg.lstsq(base, weight[:, j]).solution
                residuals.append((float((weight[:, j] - base.matmul(coef)).square().sum()), j))
            candidate = max(residuals)[1]
        chosen.append(candidate)
    chosen = sorted(chosen)
    a = weight[:, chosen]
    transformed = torch.linalg.solve(a, weight)
    free = transformed[:, [j for j in range(20) if j not in chosen]]
    return chosen, a, free

gamma, beta = s["attn_norm.weight"], s["attn_norm.bias"]
k_raw = s["k.weight"] * gamma[None, :]
k_raw = k_raw - k_raw[:, -1:]
v_raw = s["v.weight"] * gamma[None, :]
v_bias = s["v.weight"].matmul(beta)
v_raw = v_raw - v_raw[:, -1:]
ki, ka, kfree_full = basis_gauge(k_raw)
vi, va, vfree_full = basis_gauge(v_raw)
kfree = kfree_full[:, :14] if 19 not in ki else (_ for _ in ()).throw(RuntimeError("key gauge selected zero column"))
vfree = vfree_full[:, :14] if 19 not in vi else (_ for _ in ()).throw(RuntimeError("value gauge selected zero column"))
q = []
for layer in range(2):
    old_full = s[f"q.{layer}.weight"] * gamma[None, :]
    old_bias = s[f"q.{layer}.weight"].matmul(beta).reshape(4, 5)
    old = (old_full - old_full[:, -1:]).reshape(4, 5, 20)
    q.append((torch.stack([ka.T.matmul(old[h]) for h in range(4)]).reshape(20, 20)[:, :19],
              torch.stack([ka.T.matmul(old_bias[h]) for h in range(4)]).reshape(20)))
# V'=VA^-1 V, so each output projection consumes VA * V'.
for layer in range(2):
    full_o = torch.cat((o[layer], torch.zeros(1, 20, dtype=torch.double)), 0)
    blocks = [full_o[:, 5*h:5*h+5].matmul(va) for h in range(4)]
    o[layer] = zero_last_output(torch.cat(blocks, 1))

weights = {
    "token": token, "pos_free": pos_free, "pos_right": pos_right,
    "q0": q[0][0], "qb0": q[0][1], "q1": q[1][0], "qb1": q[1][1], "o0": o[0], "o1": o[1],
    "k_free": kfree, "v_free": vfree, "v_bias": torch.linalg.solve(va, v_bias),
    "ff1_weight": (ff_w - ff_w[:, -1:])[:, :-1], "ff1_bias": ff_b, "ff2": ff2,
    "head_weight": (head_w - head_w[:, -1:])[:, :-1], "head_bias": head_b,
}

def literal(t):
    return "torch.tensor(" + repr(t.float().reshape(-1).tolist()) + ",dtype=torch.float32).reshape(" + repr(tuple(t.shape)) + ")"

items = ",\n".join(repr(k) + ":" + literal(v) for k, v in weights.items())
source = '''import torch
from torch import nn

_W = {{WEIGHTS}}
_POS_FIXED = ({PI}, {PJ})
_K_FIXED = {KI}
_V_FIXED = {VI}

class AdditionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        for name, value in _W.items():
            setattr(self, name, nn.Parameter(value))

    @staticmethod
    def _matrix(free, fixed):
        rows = []
        cursor = 0
        eye = torch.eye(len(fixed), dtype=free.dtype, device=free.device)
        for j in range(20):
            if j == 19:
                rows.append(torch.zeros(len(fixed), dtype=free.dtype, device=free.device))
            elif j in fixed:
                rows.append(eye[:, fixed.index(j)])
            else:
                rows.append(free[:, cursor])
                cursor += 1
        return torch.stack(rows, 1)

    def forward(self, tokens):
        length = tokens.shape[1]
        token_table = torch.cat((self.token, torch.zeros(11, 1, device=tokens.device)), 1)
        left_rows = []
        cursor = 0
        eye = torch.eye(2, dtype=self.pos_free.dtype, device=tokens.device)
        for j in range(25):
            if j == _POS_FIXED[0]: left_rows.append(eye[0])
            elif j == _POS_FIXED[1]: left_rows.append(eye[1])
            else:
                left_rows.append(self.pos_free[cursor]); cursor += 1
        left = torch.stack(left_rows)
        right = torch.cat((self.pos_right, torch.zeros(2, 1, device=tokens.device)), 1)
        x = token_table[tokens] + left[:length].matmul(right)
        k_weight = self._matrix(self.k_free, _K_FIXED)
        v_weight = self._matrix(self.v_free, _V_FIXED)
        mask = torch.ones(length, length, dtype=torch.bool, device=tokens.device).triu(1)
        for layer in range(2):
            z = torch.nn.functional.layer_norm(x, (20,))
            qw = self.q0 if layer == 0 else self.q1
            qb = self.qb0 if layer == 0 else self.qb1
            qw = torch.cat((qw, torch.zeros(20, 1, device=tokens.device)), 1)
            q = torch.nn.functional.linear(z, qw, qb).view(-1, length, 4, 5).transpose(1, 2)
            k = torch.nn.functional.linear(z, k_weight).unsqueeze(1)
            v = torch.nn.functional.linear(z, v_weight, self.v_bias).unsqueeze(1)
            scores = q.matmul(k.transpose(-2, -1)) * (5.0 ** -0.5)
            scores = scores.masked_fill(mask, -torch.inf)
            context = scores.softmax(-1).matmul(v).transpose(1, 2).reshape(-1, length, 20)
            ow = self.o0 if layer == 0 else self.o1
            attention = torch.nn.functional.linear(context, ow)
            attention = torch.cat((attention, torch.zeros_like(attention[..., :1])), -1)
            x = x + attention
            z = torch.nn.functional.layer_norm(x, (20,))
            zero_column = torch.zeros(self.ff1_weight.shape[0], 1, device=tokens.device)
            hidden = torch.nn.functional.gelu(torch.nn.functional.linear(z, torch.cat((self.ff1_weight, zero_column), 1), self.ff1_bias))
            feedforward = torch.nn.functional.linear(hidden, self.ff2)
            x = x + torch.cat((feedforward, torch.zeros_like(feedforward[..., :1])), -1)
        z = torch.nn.functional.layer_norm(x, (20,))
        zero_column = torch.zeros(self.head_weight.shape[0], 1, device=tokens.device)
        logits = torch.nn.functional.linear(z, torch.cat((self.head_weight, zero_column), 1), self.head_bias)
        return torch.cat((logits, torch.zeros_like(logits[..., :1])), -1)


def build_model():
    model = AdditionTransformer()
    model.eval()
    return model, {"architecture": "causal grouped-query digit transformer", "digits": 8}


def add(model, a: int, b: int) -> int:
    da, db = str(a)[::-1], str(b)[::-1]
    tokens = []
    for xa, xb in zip(da, db):
        tokens.extend((ord(xa) - 48, ord(xb) - 48))
    tokens.append(10)
    device = next(model.parameters()).device
    sequence = torch.tensor([tokens], dtype=torch.long, device=device)
    answer = []
    with torch.no_grad():
        for _ in range(9):
            digit = int(model(sequence)[0, -1].argmax())
            answer.append(str(digit))
            sequence = torch.cat((sequence, torch.tensor([[digit]], device=device)), 1)
    return int("".join(reversed(answer)))
'''.replace("{WEIGHTS}", items).replace("{PI}", str(pi)).replace("{PJ}", str(pj)).replace("{KI}", repr(ki)).replace("{VI}", repr(vi))

new_path = PATH + ".compressed"
with open(new_path, "w") as f:
    f.write(source)
    f.flush(); os.fsync(f.fileno())
print("position fixed", pi, pj, "K", ki, "V", vi)
print("compressed parameters", sum(v.numel() for v in weights.values()))
print(new_path)
