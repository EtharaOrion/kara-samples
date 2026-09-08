# --- BEGIN MODEL SOURCE (inlined verbatim into submission.py) ---
import torch
import torch.nn as nn

SEQ_LEN = 10  # pos 0 = sink, pos 1..8 = digit places 0..7 (LSB first), pos 9 = carry-out


def _rms(x, eps=1e-5):
    """Parameter-free RMS normalisation."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class SelfAttention(nn.Module):
    """Causal multi-head self-attention with an ALiBi-style relative bias.

    The attention logit for query position t over key position j is

        (W_q x_t + b_q) . (W_k x_j)  +  slope * (t - j)  +  self_bias * [t == j]

    The learned bias supplies only a generic recency preference, identical for
    every input; which earlier position actually wins is decided by the key
    content, and therefore by the operands.
    """

    def __init__(self, d_model, n_heads, d_head):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        inner = n_heads * d_head
        self.w_q = nn.Linear(d_model, inner, bias=True)
        self.w_k = nn.Linear(d_model, inner, bias=False)
        self.w_v = nn.Linear(d_model, inner, bias=False)
        self.w_o = nn.Linear(inner, d_model, bias=False)
        self.rel_bias = nn.Parameter(torch.zeros(n_heads, 2))  # slope, self

    def forward(self, x, return_attn=False):
        B, T, _ = x.shape
        H, K = self.n_heads, self.d_head
        q = self.w_q(x).view(B, T, H, K).transpose(1, 2)
        k = self.w_k(x).view(B, T, H, K).transpose(1, 2)
        v = self.w_v(x).view(B, T, H, K).transpose(1, 2)
        scores = q @ k.transpose(-2, -1)
        idx = torch.arange(T, device=x.device)
        delta = idx.view(T, 1) - idx.view(1, T)
        d = delta.clamp(min=0).to(scores.dtype)
        scores = scores + (self.rel_bias[:, 0].view(H, 1, 1) * d
                           + self.rel_bias[:, 1].view(H, 1, 1) * (d == 0).to(d.dtype))
        scores = scores.masked_fill(delta < 0, float("-inf"))
        attn = scores.softmax(dim=-1)
        y = (attn @ v).transpose(1, 2).reshape(B, T, H * K)
        y = self.w_o(y)
        if return_attn:
            return y, attn
        return y


class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden, bias=True)
        self.fc2 = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class AdderTransformer(nn.Module):
    """A single Macaron-style transformer block: FFN -> self-attention -> FFN.

    Input is a (B, 10, 2) integer tensor of digits; the token at sequence
    position t is embedded as emb[a_t] + emb[b_t].  Output is (B, 10, 10)
    logits over the sum digit produced at each place.  Pre-normalisation is
    parameter-free RMS; the unembedding is tied to the input embedding.
    """

    def __init__(self, d_model, d_ff_in, n_heads, d_head, d_ff_out):
        super().__init__()
        self.emb = nn.Parameter(torch.zeros(10, d_model))
        self.ffn_in = FeedForward(d_model, d_ff_in)
        self.attn = SelfAttention(d_model, n_heads, d_head)
        self.ffn_out = FeedForward(d_model, d_ff_out)
        self.logit_scale = nn.Parameter(torch.ones(1))

    def forward(self, digits, return_attn=False):
        x = self.emb[digits[..., 0]] + self.emb[digits[..., 1]]
        x = x + self.ffn_in(_rms(x))
        if return_attn:
            a, amap = self.attn(_rms(x), return_attn=True)
        else:
            a, amap = self.attn(_rms(x)), None
        x = x + a
        x = x + self.ffn_out(_rms(x))
        logits = _rms(x) @ self.emb.t() * self.logit_scale
        if return_attn:
            return logits, [amap]
        return logits


def encode_pair(a, b, device=None):
    """Tokenise two integers into the (1, 10, 2) digit-pair sequence."""
    da = [0] * SEQ_LEN
    db = [0] * SEQ_LEN
    for i in range(8):
        da[i + 1] = (a // 10 ** i) % 10
        db[i + 1] = (b // 10 ** i) % 10
    rows = [[da[t], db[t]] for t in range(SEQ_LEN)]
    return torch.tensor([rows], dtype=torch.long, device=device)


def decode_digits(digit_list):
    """Assemble predicted per-place digits (LSB first) back into an integer."""
    total = 0
    for i, d in enumerate(digit_list):
        total += int(d) * 10 ** i
    return total
# --- END MODEL SOURCE ---


KEY_MAP = {
    "emb": "emb",
    "logit_scale": "logit_scale",
    "blocks.0.mlp.fc1.weight": "ffn_in.fc1.weight",
    "blocks.0.mlp.fc1.bias": "ffn_in.fc1.bias",
    "blocks.0.mlp.fc2.weight": "ffn_in.fc2.weight",
    "blocks.1.attn.w_q.weight": "attn.w_q.weight",
    "blocks.1.attn.w_q.bias": "attn.w_q.bias",
    "blocks.1.attn.w_k.weight": "attn.w_k.weight",
    "blocks.1.attn.w_v.weight": "attn.w_v.weight",
    "blocks.1.attn.w_o.weight": "attn.w_o.weight",
    "blocks.1.attn.rel_bias": "attn.rel_bias",
    "blocks.1.mlp.fc1.weight": "ffn_out.fc1.weight",
    "blocks.1.mlp.fc1.bias": "ffn_out.fc1.bias",
    "blocks.1.mlp.fc2.weight": "ffn_out.fc2.weight",
}


def convert(ckpt):
    """Training checkpoint (generic block list) -> final single-block model."""
    cfg, state = ckpt["cfg"], ckpt["state"]
    assert cfg.get("rel_mode") == "ramp", "final model uses the ramp relative bias"
    (h0, k0, m1), (h1, k1, m2) = [tuple(b) for b in cfg["blocks"]]
    assert h0 == 0, "final model has exactly one attention layer"
    model = AdderTransformer(cfg["d_model"], m1, h1, k1, m2)
    new = {KEY_MAP[k]: v for k, v in state.items()}
    model.load_state_dict(new)
    model.eval()
    arch = {"d_model": cfg["d_model"], "d_ff_in": m1, "n_heads": h1,
            "d_head": k1, "d_ff_out": m2}
    return model, arch
