"""A minimal transformer that adds two 8-digit integers.

The model is a single Macaron-style transformer block -- feed-forward, causal
self-attention, feed-forward -- over a sequence of per-place digit-pair tokens.
The sequence is 10 positions long, least-significant place first:

    pos 0      a sink that anchors "no carry from below"
    pos 1..8   digit place i, embedded as emb[a_i] + emb[b_i]
    pos 9      the carry-out slot

Every position predicts the sum digit produced at its own place, so a single
forward pass yields all nine output digits.

The self-attention is what performs carry propagation.  The first
feed-forward marks each place as carry-generating (a_i + b_i >= 10) or
carry-transparent (a_i + b_i == 9); the attention head then makes each place
attend to the nearest earlier place that is *not* transparent and read off
whether that place generated a carry.  Which position that is depends entirely
on the operands -- the head jumps over runs of 9s of whatever length the input
happens to contain -- so the attention pattern is genuinely input-dependent.
Replacing it with its average over inputs destroys the model's accuracy.

The unembedding is tied to the input embedding, normalisation is
parameter-free RMS, and every learned value is a registered nn.Parameter.  The
weights below are the result of training this architecture on synthetic
addition examples; the training code is not part of this file.
"""
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


_ARCH = {
    'd_model': 4,
    'd_ff_in': 4,
    'n_heads': 1,
    'd_head': 1,
    'd_ff_out': 6,
}

# The trained parameter values, written out as ordinary floats.
_WEIGHTS = {
  'emb': [
   [-0.8065592646598816, 1.9426178932189941, -2.138421058654785, 1.8766058683395386],
   [0.4016250967979431, 1.8913878202438354, -1.307731032371521, 0.8320208787918091],
   [0.7292963266372681, 1.3990795612335205, -0.9161604046821594, 0.4512636661529541],
   [0.7835478186607361, 0.7698032259941101, -0.6744435429573059, 0.09909788519144058],
   [0.6601124405860901, 0.0771762952208519, -0.3866344690322876, -0.16521582007408142],
   [0.34373098611831665, -0.5330843329429626, -0.0322130024433136, -0.39988017082214355],
   [-0.15860435366630554, -1.0521068572998047, 0.3583212196826935, -0.5978211760520935],
   [-0.8513932228088379, -1.435135006904602, 0.8266668915748596, -0.8200594782829285],
   [-1.6730685234069824, -1.6994855403900146, 1.36728835105896, -0.9340057969093323],
   [-2.3700053691864014, -2.072028160095215, 1.791011095046997, -0.2929013669490814]
  ],
  'logit_scale': [82.5674057006836],
  'ffn_in.fc1.weight': [
   [0.18525215983390808, -1.4421700239181519, 2.359896183013916, 0.5401780605316162],
   [0.34281662106513977, -0.6754319667816162, 0.14552724361419678, 1.266525149345398],
   [-0.48283109068870544, -0.5161293745040894, -1.4106323719024658, 0.8610032200813293],
   [0.30071893334388733, -1.3090413808822632, -0.5696192979812622, 0.547963559627533]
  ],
  'ffn_in.fc1.bias': [0.09260125458240509, 0.025374291464686394, 0.16237080097198486, -0.29467859864234924],
  'ffn_in.fc2.weight': [
   [0.27301105856895447, -0.9173036813735962, 1.5427652597427368, -1.1529332399368286],
   [-0.056298356503248215, 0.18808422982692719, 0.1823459267616272, 0.6372008919715881],
   [2.9158236980438232, -0.300973504781723, 0.17987759411334991, 0.3925992548465729],
   [0.32481011748313904, -1.8677922487258911, -0.3651599586009979, 0.31770259141921997]
  ],
  'attn.rel_bias': [
   [-2.106961727142334, -18.12830924987793]
  ],
  'attn.w_q.weight': [
   [0.15236584842205048, -0.1591271311044693, 0.15223990380764008, -0.13248223066329956]
  ],
  'attn.w_q.bias': [-2.8259799480438232],
  'attn.w_k.weight': [
   [-1.2477999925613403, 0.9305930137634277, -0.3373156785964966, -2.1485767364501953]
  ],
  'attn.w_v.weight': [
   [-0.3128780424594879, -0.5821660161018372, 0.5411127209663391, 0.5001657009124756]
  ],
  'attn.w_o.weight': [
   [0.13569402694702148],
   [-0.2880164682865143],
   [0.6974902153015137],
   [0.19308622181415558]
  ],
  'ffn_out.fc1.weight': [
   [-1.7526180744171143, 0.02593085542321205, 2.6048028469085693, 1.0309697389602661],
   [-0.9786758422851562, 0.11553587764501572, -0.34222084283828735, -1.052245855331421],
   [1.371751308441162, 0.29276880621910095, -0.46481043100357056, -0.22057616710662842],
   [0.4130743145942688, 0.7157840728759766, 1.0383273363113403, -1.3986209630966187],
   [1.0625286102294922, 1.1576825380325317, 0.8027310371398926, -1.843855619430542],
   [0.28735390305519104, -1.7500330209732056, -0.5150654911994934, -1.0634217262268066]
  ],
  'ffn_out.fc1.bias': [-0.8200923800468445, 0.8010567426681519, 0.9959651827812195, -0.2568804919719696, -1.181618571281433, 0.09693819284439087],
  'ffn_out.fc2.weight': [
   [1.451786994934082, -1.1338186264038086, 0.4605080187320709, -1.0070773363113403, -1.9900641441345215, -1.1038217544555664],
   [-0.005606775172054768, -1.5676512718200684, -1.1986526250839233, 1.4288570880889893, -0.7157392501831055, 1.566400408744812],
   [-2.4431252479553223, -0.4033815860748291, 0.4430953860282898, -0.5790883302688599, -1.2710367441177368, -0.9742757678031921],
   [-0.7630313038825989, -0.005861518904566765, -0.5026131868362427, 1.760231375694275, -1.229607343673706, 1.1959896087646484]
  ],
}

_METADATA = {
    "name": "macaron-digitpair-adder",
    "architecture": "1 block: feed-forward -> causal self-attention -> feed-forward; tied embedding; parameter-free RMS norm",
    "n_parameters": 150,
    "n_attention_layers": 1,
    'd_model': 4,
    'd_ff_in': 4,
    'n_heads': 1,
    'd_head': 1,
    'd_ff_out': 6,
    "seq_len": SEQ_LEN,
    "vocab": 10,
    "tokenisation": "one token per decimal place, LSB first; emb[a_i] + emb[b_i]",
    "input_range": [10000000, 99999999],
}


def build_model():
    """Construct the trained model and return it with its metadata."""
    model = AdderTransformer(
        _ARCH["d_model"],
        _ARCH["d_ff_in"],
        _ARCH["n_heads"],
        _ARCH["d_head"],
        _ARCH["d_ff_out"],
    )
    target = model.state_dict()
    loaded = {}
    for name, ref in target.items():
        loaded[name] = torch.tensor(_WEIGHTS[name], dtype=torch.float32).reshape(ref.shape)
    model.load_state_dict(loaded)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, dict(_METADATA)


def add(model, a, b):
    """Return a + b, decoded from a single forward pass of the model."""
    device = next(model.parameters()).device
    tokens = encode_pair(int(a), int(b), device=device)
    with torch.no_grad():
        logits = model(tokens)
    digits = logits[0, 1:, :].argmax(dim=-1).tolist()
    return decode_digits(digits)


if __name__ == "__main__":
    model, metadata = build_model()
    print(metadata)
    for x, y in [(19999995, 80000005), (12345678, 87654321), (10000000, 10000000)]:
        print(x, "+", y, "=", add(model, x, y))
