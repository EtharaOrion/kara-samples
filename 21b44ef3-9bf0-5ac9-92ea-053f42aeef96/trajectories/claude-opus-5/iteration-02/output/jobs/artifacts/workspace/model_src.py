# --- BEGIN MODEL SOURCE (inlined verbatim into submission.py) ---
import torch
import torch.nn as nn

SEQ_LEN = 10  # pos 0 = sink, pos 1..8 = digit places 0..7 (LSB first), pos 9 = carry-out slot


def _rms(x, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class SelfAttention(nn.Module):
    """Causal multi-head self-attention with a learned relative-position bias.

    Attention logits are  (W_q x_t + b_q) . (W_k x_j)  +  rel_bias[t - j],
    so the pattern is driven by key/query content; the bias only supplies a
    generic recency prior shared by every input.
    """

    def __init__(self, d_model, n_heads, d_head, max_len=SEQ_LEN, rel_mode="full"):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.rel_mode = rel_mode
        inner = n_heads * d_head
        self.w_q = nn.Linear(d_model, inner, bias=True)
        self.w_k = nn.Linear(d_model, inner, bias=False)
        self.w_v = nn.Linear(d_model, inner, bias=False)
        self.w_o = nn.Linear(inner, d_model, bias=False)
        # "full": one bias per relative offset.  "ramp": an ALiBi-style linear
        # slope in the offset plus a separate term for attending to self.
        self.rel_bias = nn.Parameter(torch.zeros(n_heads, max_len if rel_mode == "full" else 2))

    def rel(self, delta):
        if self.rel_mode == "full":
            return self.rel_bias[:, delta.clamp(min=0)]
        d = delta.clamp(min=0).to(self.rel_bias.dtype)
        return (self.rel_bias[:, 0].view(-1, 1, 1) * d
                + self.rel_bias[:, 1].view(-1, 1, 1) * (d == 0).to(d.dtype))

    def forward(self, x, return_attn=False):
        B, T, _ = x.shape
        H, K = self.n_heads, self.d_head
        q = self.w_q(x).view(B, T, H, K).transpose(1, 2)
        k = self.w_k(x).view(B, T, H, K).transpose(1, 2)
        v = self.w_v(x).view(B, T, H, K).transpose(1, 2)
        scores = q @ k.transpose(-2, -1)
        idx = torch.arange(T, device=x.device)
        delta = idx.view(T, 1) - idx.view(1, T)
        causal = delta >= 0
        scores = scores + self.rel(delta)
        scores = scores.masked_fill(~causal, float("-inf"))
        attn = scores.softmax(dim=-1)
        y = (attn @ v).transpose(1, 2).reshape(B, T, H * K)
        y = self.w_o(y)
        if return_attn:
            return y, attn
        return y


class MLP(nn.Module):
    def __init__(self, d_model, d_mlp):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_mlp, bias=True)
        self.fc2 = nn.Linear(d_mlp, d_model, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_head, d_mlp, use_attn=True, rel_mode="full"):
        super().__init__()
        self.attn = (SelfAttention(d_model, n_heads, d_head, rel_mode=rel_mode)
                     if use_attn else None)
        self.mlp = MLP(d_model, d_mlp) if d_mlp > 0 else None


class AdderTransformer(nn.Module):
    """Decoder-only transformer over per-place digit-pair tokens.

    Input is a (B, 10, 2) integer tensor of digits; the token at sequence
    position t embeds as emb[a] + emb[b].  Output is (B, 10, 10) logits over
    the sum digit produced at each place.  The unembedding is tied to the
    input embedding.
    """

    def __init__(self, d_model, blocks, norm="rms", rel_mode="full"):
        super().__init__()
        self.d_model = d_model
        self.norm_kind = norm
        self.emb = nn.Parameter(torch.randn(10, d_model) * 0.6)
        self.blocks = nn.ModuleList(
            [Block(d_model, h, k, m, use_attn=(h > 0), rel_mode=rel_mode)
             for (h, k, m) in blocks]
        )
        self.logit_scale = nn.Parameter(torch.ones(1))

    def _norm(self, x):
        return _rms(x) if self.norm_kind == "rms" else x

    def forward(self, digits, return_attn=False):
        x = self.emb[digits[..., 0]] + self.emb[digits[..., 1]]
        attns = []
        for blk in self.blocks:
            if blk.attn is not None:
                if return_attn:
                    a_out, a_map = blk.attn(self._norm(x), return_attn=True)
                    attns.append(a_map)
                else:
                    a_out = blk.attn(self._norm(x))
                x = x + a_out
            if blk.mlp is not None:
                x = x + blk.mlp(self._norm(x))
        logits = self._norm(x) @ self.emb.t() * self.logit_scale
        if return_attn:
            return logits, attns
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
