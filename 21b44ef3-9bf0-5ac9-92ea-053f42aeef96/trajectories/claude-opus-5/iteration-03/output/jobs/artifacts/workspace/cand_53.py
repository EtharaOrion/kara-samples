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
                 act="relu", d_v=0, res_bias=True, share_qk=False,
                 norm=True, q_bias=True, self_bias=True, out_scale=True,
                 res_bias_in=None, res_bias_out=None):
        super().__init__()
        self.d_model = d_model
        self.n_pos = n_pos
        self.vocab = vocab
        self.act = act
        self.d_v = d_v
        self.res_bias = res_bias
        self.rb_in = res_bias if res_bias_in is None else res_bias_in
        self.rb_out = res_bias if res_bias_out is None else res_bias_out
        self.d_ff_in = d_ff_in
        self.share_qk = share_qk
        self.norm = norm
        self.q_bias = q_bias
        self.use_self_bias = self_bias
        self.out_scale = out_scale

        self.emb = nn.Parameter(torch.zeros(vocab, d_model))

        if d_ff_in:
            self.w_in1 = nn.Parameter(torch.zeros(d_model, d_ff_in))
            self.b_in1 = nn.Parameter(torch.zeros(d_ff_in))
            self.w_in2 = nn.Parameter(torch.zeros(d_ff_in, d_model))
            if self.rb_in:
                self.b_in2 = nn.Parameter(torch.zeros(d_model))

        self.w_q = nn.Parameter(torch.zeros(d_model, 1))
        if q_bias:
            self.b_q = nn.Parameter(torch.zeros(1))
        if not share_qk:
            self.w_k = nn.Parameter(torch.zeros(d_model, 1))
        self.alibi = nn.Parameter(torch.zeros(1))
        if self_bias:
            self.self_bias = nn.Parameter(torch.zeros(1))
        if d_v:
            self.w_v = nn.Parameter(torch.zeros(d_model, d_v))
            self.w_o = nn.Parameter(torch.zeros(d_v, d_model))

        self.w_out1 = nn.Parameter(torch.zeros(d_model, d_ff_out))
        self.b_out1 = nn.Parameter(torch.zeros(d_ff_out))
        self.w_out2 = nn.Parameter(torch.zeros(d_ff_out, d_model))
        if self.rb_out:
            self.b_out2 = nn.Parameter(torch.zeros(d_model))

        if out_scale:
            self.logit_scale = nn.Parameter(torch.ones(1))

        pos = torch.arange(n_pos)
        dist = (pos.view(-1, 1) - pos.view(1, -1)).float()
        self.register_buffer("dist", dist, persistent=False)
        self.register_buffer("eye", torch.eye(n_pos), persistent=False)
        self.register_buffer("causal", dist >= 0, persistent=False)

    def _nl(self, z):
        return F.relu(z) if self.act == "relu" else F.gelu(z)

    def _n(self, x):
        return _rms(x) if self.norm else x

    def embed(self, a_dig, b_dig):
        """a_dig, b_dig: (B, n_pos-2) long tensors of digits, LSB first."""
        pad = (1, 1)
        a = F.pad(a_dig, pad, value=0)
        b = F.pad(b_dig, pad, value=0)
        return self.emb[a] + self.emb[b]

    def _rel(self, n):
        if n == self.n_pos:
            return self.dist, self.eye, self.causal
        pos = torch.arange(n, device=self.emb.device)
        dist = (pos.view(-1, 1) - pos.view(1, -1)).float()
        return dist, torch.eye(n, device=self.emb.device), dist >= 0

    def attend(self, h):
        dist, eye, causal = self._rel(h.shape[1])
        q = h @ self.w_q
        if self.q_bias:
            q = q + self.b_q
        k = h @ (self.w_q if self.share_qk else self.w_k)
        scores = q * k.transpose(1, 2) + self.alibi * dist
        if self.use_self_bias:
            scores = scores + self.self_bias * eye
        scores = scores.masked_fill(~causal, float("-inf"))
        return torch.softmax(scores, dim=-1)

    def forward(self, a_dig, b_dig, attn_override=None, return_attn=False):
        x = self.embed(a_dig, b_dig)

        if self.d_ff_in:
            h = self._n(x)
            d = self._nl(h @ self.w_in1 + self.b_in1) @ self.w_in2
            x = x + (d + self.b_in2 if self.rb_in else d)

        h = self._n(x)
        attn = self.attend(h)
        used = attn if attn_override is None else attn_override
        v = h @ self.w_v if self.d_v else h
        o = used @ v
        x = x + (o @ self.w_o if self.d_v else o)

        h = self._n(x)
        d = self._nl(h @ self.w_out1 + self.b_out1) @ self.w_out2
        x = x + (d + self.b_out2 if self.rb_out else d)

        logits = _rms(x) @ self.emb.t()
        if self.out_scale:
            logits = self.logit_scale * logits
        if return_attn:
            return logits, attn
        return logits


_WEIGHTS = {
    'emb': [[-5.547868728637695, -0.864439845085144, 3.151563882827759],
         [-4.426313400268555, -0.6064572930335999, 2.1645429134368896],
         [-3.325568675994873, -0.4513811767101288, 1.3581942319869995],
         [-2.07231068611145, -0.35566896200180054, 0.5924896001815796],
         [-0.7923845052719116, -0.20308919250965118, -0.1013108566403389],
         [0.5353638529777527, 0.002828734228387475, -0.7292953133583069],
         [1.8338483572006226, 0.22506950795650482, -1.248718023300171],
         [3.1163785457611084, 0.4512763023376465, -1.6676733493804932],
         [4.393383979797363, 0.6676702499389648, -1.99246346950531],
         [5.801793098449707, 0.8746795058250427, -2.248213291168213]],
    'w_in1': [[12.116304397583008],
         [-2.4406304359436035],
         [-3.2328555583953857]],
    'b_in1': [-0.29194575548171997],
    'w_in2': [[-0.2531394064426422, -5.865352153778076, -2.183713674545288]],
    'w_q': [[0.87654709815979],
         [1.432377576828003],
         [-0.4704905152320862]],
    'b_q': [-15.20797348022461],
    'alibi': [-3.91507887840271],
    'self_bias': [-34.47920227050781],
    'w_out1': [[5.500680446624756],
         [-2.3674416542053223],
         [-2.3935461044311523]],
    'b_out1': [7.927908897399902],
    'w_out2': [[-0.5417633652687073, 8.88576889038086, 3.8962841033935547]],
    'b_out2': [-1.0056414604187012, -5.027050971984863, -22.807836532592773],
}


_SHAPES = {'emb': [10, 3], 'w_in1': [3, 1], 'b_in1': [1], 'w_in2': [1, 3], 'w_q': [3, 1], 'b_q': [1], 'alibi': [1], 'self_bias': [1], 'w_out1': [3, 1], 'b_out1': [1], 'w_out2': [1, 3], 'b_out2': [3]}

_CFG = {'d_model': 3, 'd_ff_in': 1, 'd_ff_out': 1, 'n_pos': 10, 'vocab': 10, 'act': 'relu', 'd_v': 0, 'res_bias': True, 'share_qk': True, 'norm': True, 'q_bias': True, 'self_bias': True, 'out_scale': False, 'res_bias_in': False, 'res_bias_out': True}


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
        "held_out_exact_match": 1.0,
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
