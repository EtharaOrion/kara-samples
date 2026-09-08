"""Minimal transformer that adds two 8-digit integers.

Architecture: decoder-only transformer over per-place digit-pair tokens.
The sequence is 10 positions long, least-significant place first:

    pos 0      sink / no-carry anchor
    pos 1..8   the digit pair (a_i, b_i) of place i, embedded as emb[a]+emb[b]
    pos 9      the carry-out slot

Every position predicts the sum digit produced at its place, so one forward
pass yields all nine output digits.  Causal self-attention with a learned
relative-position bias performs the carry lookup: a query place attends to
the most recent earlier place that is not carry-transparent and reads whether
that place generated a carry.  Which place that is depends entirely on the
operands, so the attention pattern is different for different inputs.

The unembedding is tied to the input embedding.  All learned values are
registered nn.Parameters; the weights below were produced by training this
architecture on synthetic addition examples (see train.py).
"""
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


_CONFIG = {
    "d_model": 4,
    "blocks": [[1, 1, 4], [1, 1, 6]],
    "norm": 'rms',
    "rel_mode": 'full',
}

# Trained parameter values, written out as ordinary floats.
_WEIGHTS = {
  'emb': [
   [0.06498263031244278, 2.50968599319458, -1.2177553176879883, 3.8467350006103516],
   [-0.8138808012008667, 0.5670536160469055, -0.405518114566803, 2.639512062072754],
   [-0.5055749416351318, -0.17584091424942017, -0.6402933597564697, 1.9520819187164307],
   [-0.23017673194408417, -0.6071619987487793, -0.6518740653991699, 1.2205209732055664],
   [6.609787669731304e-05, -0.7916700839996338, -0.465200811624527, 0.47882965207099915],
   [0.18866503238677979, -0.7726288437843323, -0.10640374571084976, -0.20850269496440887],
   [0.4735848307609558, -0.6062261462211609, 0.3011491894721985, -0.7747029066085815],
   [0.9229277968406677, -0.3212711811065674, 0.7395311594009399, -1.2072018384933472],
   [1.5256825685501099, 0.08391501754522324, 1.2090961933135986, -1.4369583129882812],
   [2.5772173404693604, 0.6538451313972473, 1.8460947275161743, -0.9999896287918091]
  ],
  'logit_scale': [54.63899230957031],
  'blocks.0.attn.rel_bias': [
   [4.61940860748291, -0.7928602695465088, -2.9199628829956055, -5.347681045532227, -8.980572700500488, -12.05256462097168, -12.26019287109375, -11.077301979064941, -8.32656192779541, 0.11054902523756027]
  ],
  'blocks.0.attn.w_q.weight': [
   [-0.7552801966667175, -0.07525546848773956, -0.7653866410255432, 0.7153263092041016]
  ],
  'blocks.0.attn.w_q.bias': [0.6242672204971313],
  'blocks.0.attn.w_k.weight': [
   [-0.7478020191192627, 0.248660147190094, -1.5118119716644287, 1.149281620979309]
  ],
  'blocks.0.attn.w_v.weight': [
   [0.31008684635162354, -0.5429869890213013, 0.3503189980983734, -0.5947684645652771]
  ],
  'blocks.0.attn.w_o.weight': [
   [0.44874489307403564],
   [1.405910849571228],
   [0.9213293194770813],
   [0.6281914114952087]
  ],
  'blocks.0.mlp.fc1.weight': [
   [-1.1298445463180542, 0.5730088353157043, 0.5438998341560364, -0.09640699625015259],
   [0.4581245481967926, 1.229620337486267, -0.2870566248893738, -0.2696523666381836],
   [-0.4799129068851471, -0.07022155076265335, -0.45598581433296204, 0.670208752155304],
   [-0.380482017993927, 0.212651789188385, -1.7244987487792969, -0.08742755651473999]
  ],
  'blocks.0.mlp.fc1.bias': [1.048937439918518, 0.6508557796478271, 0.4116468131542206, 0.700870156288147],
  'blocks.0.mlp.fc2.weight': [
   [1.2390879392623901, -0.40519165992736816, 0.05102448910474777, -1.4173833131790161],
   [-0.7984243631362915, -1.2226980924606323, -0.027885213494300842, -0.28814467787742615],
   [-0.25125956535339355, 0.09534502774477005, -0.8535048365592957, -0.39285314083099365],
   [1.4675366878509521, -0.6310096979141235, -1.189850926399231, -0.7117405533790588]
  ],
  'blocks.1.attn.rel_bias': [
   [-7.493399143218994, 7.141839027404785, 4.284543037414551, 2.053917169570923, 0.21624286472797394, -1.3569436073303223, -2.7509548664093018, -4.1405029296875, -5.735383033752441, -11.969701766967773]
  ],
  'blocks.1.attn.w_q.weight': [
   [0.17091266810894012, 0.12272938340902328, -0.20510627329349518, 0.19373413920402527]
  ],
  'blocks.1.attn.w_q.bias': [5.257021903991699],
  'blocks.1.attn.w_k.weight': [
   [-1.2279775142669678, 0.3129821717739105, 1.129278540611267, 0.33019304275512695]
  ],
  'blocks.1.attn.w_v.weight': [
   [-0.0736682340502739, 0.021032819524407387, -0.5861411094665527, -0.18744970858097076]
  ],
  'blocks.1.attn.w_o.weight': [
   [-0.40401819348335266],
   [0.6053751111030579],
   [-0.6452645659446716],
   [0.11390206962823868]
  ],
  'blocks.1.mlp.fc1.weight': [
   [-0.8216368556022644, -0.8212539553642273, 0.043609619140625, -0.8355533480644226],
   [0.004361292812973261, 0.4953506886959076, -0.1117432489991188, -0.12834610044956207],
   [0.8590355515480042, -1.0341873168945312, 0.060465168207883835, -0.5202308893203735],
   [1.086266279220581, 0.16451795399188995, 1.4928382635116577, -1.0337682962417603],
   [0.378589391708374, -0.5198825001716614, -1.355467438697815, 0.17125895619392395],
   [0.45890817046165466, -0.8220374584197998, -0.7802985906600952, -1.91368567943573]
  ],
  'blocks.1.mlp.fc1.bias': [-0.18902426958084106, -0.6296045780181885, -0.15957948565483093, -0.4236891269683838, 0.7402995228767395, -1.4456077814102173],
  'blocks.1.mlp.fc2.weight': [
   [1.8964918851852417, 0.16102346777915955, -0.42931240797042847, -1.0296404361724854, -0.05097288265824318, -1.910498857498169],
   [0.24259546399116516, 0.5169273018836975, 1.3116264343261719, -0.9536343812942505, -0.1925678700208664, 2.4971418380737305],
   [0.21500323712825775, 0.0844634473323822, -0.41575247049331665, -0.891967236995697, 1.4879610538482666, 1.7207881212234497],
   [0.042168356478214264, 0.11021734029054642, 0.7899444699287415, -0.38757914304733276, -0.2997913360595703, -0.009758373722434044]
  ],
}

_METADATA = {
    "name": "tiny-digitpair-adder",
    "architecture": "causal transformer, tied embedding, learned relative-position bias",
    "n_parameters": 185,
    "d_model": 4,
    "blocks": [[1, 1, 4], [1, 1, 6]],  # (n_heads, d_head, d_mlp) per block
    "seq_len": SEQ_LEN,
    "vocab": 10,
    "input_range": [10000000, 99999999],
}



def build_model():
    """Construct the trained model and its metadata."""
    model = AdderTransformer(
        _CONFIG["d_model"],
        [tuple(b) for b in _CONFIG["blocks"]],
        norm=_CONFIG["norm"],
        rel_mode=_CONFIG["rel_mode"],
    )
    shapes = {k: v.shape for k, v in model.state_dict().items()}
    loaded = {}
    for name, shape in shapes.items():
        loaded[name] = torch.tensor(_WEIGHTS[name], dtype=torch.float32).reshape(shape)
    model.load_state_dict(loaded)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, dict(_METADATA)


def add(model, a, b):
    """Return a + b, as decoded from a single forward pass of the model."""
    device = next(model.parameters()).device
    tokens = encode_pair(int(a), int(b), device=device)
    with torch.no_grad():
        logits = model(tokens)
    digits = logits[0, 1:, :].argmax(dim=-1).tolist()
    return decode_digits(digits)


if __name__ == "__main__":
    m, meta = build_model()
    print(meta)
    print(add(m, 19999995, 80000005))
