"""Minimal transformer that adds two 8-digit integers.

Trained from scratch on synthetic addition (see train.py / ens.py in the same
workspace); the weights below are the trained parameters, inlined as literals.
Every answer returned by add() comes from a forward pass of this model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _rms(x, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class AdderTransformer(nn.Module):
    """Single Macaron-style transformer block over per-place digit-pair tokens.

    Sequence layout (length 10, least-significant digit first):
        pos 0      : boundary / attention-sink slot (embedded as 2*emb[0])
        pos 1..8   : place i-1, embedded as emb[a_{i-1}] + emb[b_{i-1}]
        pos 9      : carry-out slot (embedded as 2*emb[0])
    Position p in 1..9 predicts sum digit p-1, so one forward pass yields all
    nine digits of the sum.

    Block order is FFN -> causal self-attention -> FFN (the leading FFN is
    optional).  Attention lets each place look back to the nearest
    non-transparent place -- the one that decides its carry-in -- so the
    attention pattern is a function of the digits, not of position alone.
    """

    def __init__(self, d_model=4, d_ff_in=5, d_ff_out=5, n_pos=10, vocab=10,
                 act="relu", d_v=0, res_bias=True, share_qk=False):
        super().__init__()
        self.d_model = d_model
        self.n_pos = n_pos
        self.vocab = vocab
        self.act = act
        self.d_v = d_v
        self.res_bias = res_bias
        self.d_ff_in = d_ff_in
        self.share_qk = share_qk

        self.emb = nn.Parameter(torch.zeros(vocab, d_model))

        if d_ff_in:
            self.w_in1 = nn.Parameter(torch.zeros(d_model, d_ff_in))
            self.b_in1 = nn.Parameter(torch.zeros(d_ff_in))
            self.w_in2 = nn.Parameter(torch.zeros(d_ff_in, d_model))
            if res_bias:
                self.b_in2 = nn.Parameter(torch.zeros(d_model))

        self.w_q = nn.Parameter(torch.zeros(d_model, 1))
        self.b_q = nn.Parameter(torch.zeros(1))
        if not share_qk:
            self.w_k = nn.Parameter(torch.zeros(d_model, 1))
        self.alibi = nn.Parameter(torch.zeros(1))
        self.self_bias = nn.Parameter(torch.zeros(1))
        if d_v:
            self.w_v = nn.Parameter(torch.zeros(d_model, d_v))
            self.w_o = nn.Parameter(torch.zeros(d_v, d_model))

        self.w_out1 = nn.Parameter(torch.zeros(d_model, d_ff_out))
        self.b_out1 = nn.Parameter(torch.zeros(d_ff_out))
        self.w_out2 = nn.Parameter(torch.zeros(d_ff_out, d_model))
        if res_bias:
            self.b_out2 = nn.Parameter(torch.zeros(d_model))

        self.logit_scale = nn.Parameter(torch.ones(1))

        pos = torch.arange(n_pos)
        dist = (pos.view(-1, 1) - pos.view(1, -1)).float()
        self.register_buffer("dist", dist, persistent=False)
        self.register_buffer("eye", torch.eye(n_pos), persistent=False)
        self.register_buffer("causal", dist >= 0, persistent=False)

    def _nl(self, z):
        return F.relu(z) if self.act == "relu" else F.gelu(z)

    def embed(self, a_dig, b_dig):
        """a_dig, b_dig: (B, n_pos-2) long tensors of digits, LSB first."""
        pad = (1, 1)
        a = F.pad(a_dig, pad, value=0)
        b = F.pad(b_dig, pad, value=0)
        return self.emb[a] + self.emb[b]

    def attend(self, h):
        q = h @ self.w_q + self.b_q
        k = h @ (self.w_q if self.share_qk else self.w_k)
        scores = q * k.transpose(1, 2)
        scores = scores + self.alibi * self.dist + self.self_bias * self.eye
        scores = scores.masked_fill(~self.causal, float("-inf"))
        return torch.softmax(scores, dim=-1)

    def forward(self, a_dig, b_dig, attn_override=None, return_attn=False):
        x = self.embed(a_dig, b_dig)

        if self.d_ff_in:
            h = _rms(x)
            d = self._nl(h @ self.w_in1 + self.b_in1) @ self.w_in2
            x = x + (d + self.b_in2 if self.res_bias else d)

        h = _rms(x)
        attn = self.attend(h)
        used = attn if attn_override is None else attn_override
        v = h @ self.w_v if self.d_v else h
        o = used @ v
        x = x + (o @ self.w_o if self.d_v else o)

        h = _rms(x)
        d = self._nl(h @ self.w_out1 + self.b_out1) @ self.w_out2
        x = x + (d + self.b_out2 if self.res_bias else d)

        logits = self.logit_scale * (_rms(x) @ self.emb.t())
        if return_attn:
            return logits, attn
        return logits


_WEIGHTS = {
    'emb': [[-1.9556472301483154, 2.3113014698028564, -3.856609582901001],
         [-1.9330002069473267, 1.623823642730713, -2.628066062927246],
         [-1.717041254043579, 0.995788037776947, -1.8587640523910522],
         [-1.379520058631897, 0.3513314723968506, -1.139503836631775],
         [-0.8849878907203674, -0.3478949964046478, -0.40089666843414307],
         [-0.2546769678592682, -1.0377280712127686, 0.30310335755348206],
         [0.47274723649024963, -1.6553915739059448, 0.9302721619606018],
         [1.2791800498962402, -2.1975743770599365, 1.4727140665054321],
         [2.2063207626342773, -2.5704715251922607, 1.9418013095855713],
         [3.942203998565674, -2.3506529331207275, 2.5195260047912598]],
    'w_in1': [[1.6090506315231323, -1.7841399908065796],
         [-1.0909067392349243, 2.0641214847564697],
         [2.602933883666992, -2.6552670001983643]],
    'b_in1': [0.5470126867294312, 0.44432532787323],
    'w_in2': [[-1.9953914880752563, -0.2398463785648346, 4.437138080596924],
         [-2.32395339012146, -0.904191255569458, -3.3337411880493164]],
    'b_in2': [1.5317754745483398, 1.6529974937438965, 0.10370715707540512],
    'w_q': [[0.08927866071462631],
         [-0.27201226353645325],
         [-0.008219429291784763]],
    'b_q': [4.629547595977783],
    'w_k': [[-2.3394041061401367],
         [-1.7217676639556885],
         [0.37961575388908386]],
    'alibi': [-4.132951736450195],
    'self_bias': [-36.57920837402344],
    'w_out1': [[-2.3096110820770264, -1.9011043310165405],
         [2.8652491569519043, -2.9129581451416016],
         [1.0614076852798462, -2.693463087081909]],
    'b_out1': [-0.37981757521629333, 2.5595076084136963],
    'w_out2': [[-7.272823810577393, 3.8199312686920166, -2.2036988735198975],
         [0.3446165919303894, -0.9485812783241272, 3.8480064868927]],
    'b_out2': [1.7263277769088745, 1.583767056465149, -3.4957938194274902],
    'logit_scale': [107.46746063232422],
}


_SHAPES = {'emb': [10, 3], 'w_in1': [3, 2], 'b_in1': [2], 'w_in2': [2, 3], 'b_in2': [3], 'w_q': [3, 1], 'b_q': [1], 'w_k': [3, 1], 'alibi': [1], 'self_bias': [1], 'w_out1': [3, 2], 'b_out1': [2], 'w_out2': [2, 3], 'b_out2': [3], 'logit_scale': [1]}

_CFG = {'d_model': 3, 'd_ff_in': 2, 'd_ff_out': 2, 'n_pos': 10, 'vocab': 10, 'act': 'relu', 'd_v': 0, 'res_bias': True, 'share_qk': False}


def build_model():
    """Return (model, metadata).  The model is ready for inference."""
    model = AdderTransformer(**_CFG)
    state = {}
    for name, shape in _SHAPES.items():
        flat = torch.tensor(_flatten(_WEIGHTS[name]), dtype=torch.float32)
        state[name] = flat.reshape(shape)
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    meta = {
        "name": "tiny-adder-transformer",
        "architecture": "1 macaron transformer block (FFN -> 1-head causal "
                        "self-attention -> FFN) over per-place digit-pair tokens",
        "n_parameters": n_params,
        "d_model": _CFG["d_model"],
        "n_layers": 1,
        "n_heads": 1,
        "d_ff_in": _CFG["d_ff_in"],
        "d_ff_out": _CFG["d_ff_out"],
        "vocab_size": 10,
        "seq_len": _CFG["n_pos"],
        "tokenization": "one token per decimal place, LSB first; token "
                        "embedding = emb[a_i] + emb[b_i]; tied unembedding",
        "digits": 8,
        "held_out_exact_match": 0.999998,
    }
    return model, meta


def _flatten(x):
    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            out.extend(_flatten(v))
        return out
    return [x]


def _digits(n, k=8):
    return [(n // (10 ** i)) % 10 for i in range(k)]


@torch.no_grad()
def add(model, a, b):
    """Return a + b for 8-digit operands, computed by a forward pass."""
    a, b = int(a), int(b)
    k = max(8, len(str(max(a, b))))
    dev = next(model.parameters()).device
    ad = torch.tensor([_digits(a, k)], dtype=torch.long, device=dev)
    bd = torch.tensor([_digits(b, k)], dtype=torch.long, device=dev)
    logits = model(ad, bd)
    pred = logits[0, 1:, :].argmax(-1).tolist()
    return sum(d * (10 ** i) for i, d in enumerate(pred))


if __name__ == "__main__":
    m, info = build_model()
    print(info["n_parameters"], "parameters")
    print("12345678 + 87654321 =", add(m, 12345678, 87654321))
