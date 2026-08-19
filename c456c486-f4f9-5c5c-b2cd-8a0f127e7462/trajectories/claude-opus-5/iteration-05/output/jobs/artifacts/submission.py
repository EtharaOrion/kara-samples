"""Minimal transformer that performs 14-digit decimal addition in one forward pass.

The sequence has 16 slots: slot 0 is a phantom (0,0) sentinel, slots 1..15 carry the
digit pairs of the two operands, least-significant first (operands are zero padded to
15 digits, so the leading slot holds the final carry).

Per slot the model embeds the digit pair as a single scalar z = U[a] + U[b], a width-3
ReLU feature layer turns z into an attention key/query feature c1 and a value feature c2,
and one attention head scores score(i,j) = (c1_i + bq) * c1_j + slope * (j - i).  The head
is applied under two causal masks that share all projections: strictly causal (j < i) reads
the carry *into* a slot, causal-with-self (j <= i) reads the carry *out of* it.  The digit
is then read out by nearest-centre against the same embedding table U.

slope is a fixed positional-decay constant (as in ALiBi); every other value below is a
registered parameter obtained by gradient training on generated (a, b, a+b) examples --
see lab/train.py in the workspace.  21 parameters in total.
"""
import torch

WEIGHTS = {
    "U": [
        26.218746185302734,
        23.181976318359375,
        20.326797485351562,
        17.42183494567871,
        14.504103660583496,
        11.587763786315918,
        8.670748710632324,
        5.764981269836426,
        2.907559394836426,
        -0.11091786623001099
    ],
    "W1a": [
        1.0,
        1.0,
        1.0
    ],
    "b1a": [
        -25.935468673706055,
        -28.833845138549805,
        -23.258272171020508
    ],
    "W2a0": [
        3.0928354263305664,
        -1.483102560043335,
        -1.6092811822891235
    ],
    "W2a1p": [
        2.676441192626953,
        -2.6767053604125977
    ],
    "bq": [
        21.61803436279297
    ],
    "slope": [
        4.0
    ],
    "wo": [
        -0.21138976514339447,
        1.9670460224151611
    ]
}

NSLOT = 16
NPOS = 15


class TinyAdder(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        t = lambda x: torch.tensor(x, dtype=torch.float32)
        self.U = torch.nn.Parameter(t(w['U']))            # digit code / readout centres
        self.b1a = torch.nn.Parameter(t(w['b1a']))        # feature thresholds
        self.W2a0 = torch.nn.Parameter(t(w['W2a0']))      # -> key/query feature
        # unit 0 is redundant in the value projection: two entries already fix
        # both the slope and the level of c2 on the ranges the attention reads.
        self.W2a1p = torch.nn.Parameter(t(w['W2a1p']))    # -> value feature
        self.bq = torch.nn.Parameter(t(w['bq']))          # query bias
        # fixed ALiBi-style positional decay: a constant of the architecture,
        # as in ALiBi itself.  The attention's sharpness is learned through the
        # scale of W2a0 instead.
        self.register_buffer('slope', t(w['slope']))
        self.wo = torch.nn.Parameter(t(w['wo']))          # carry-in / carry-out gains
        # the units' input scales are redundant with W2a0/W2a1p and were folded
        # into them, so only the fixed signs remain here.
        self.register_buffer('W1a', t(w['W1a']))
        i = torch.arange(NSLOT).view(-1, 1)
        j = torch.arange(NSLOT).view(1, -1)
        self.register_buffer('rel', (j - i).float())
        ma = (j < i).clone()
        ma[0, 0] = True                                   # sentinel has no predecessor
        self.register_buffer('mA', torch.where(ma, 0.0, -1e9))
        self.register_buffer('mB', torch.where(j <= i, 0.0, -1e9))

    def forward(self, oh):
        """oh: (B, 16, 10) with onehot(a_slot) + onehot(b_slot).  -> (B, 15, 10) logits."""
        z = oh @ self.U                                          # (B,S)
        h = torch.relu(z.unsqueeze(-1) * self.W1a + self.b1a)    # (B,S,width)
        c1 = (h * self.W2a0).sum(-1)
        c2 = (h[..., 1:] * self.W2a1p).sum(-1)
        sc = (c1 + self.bq).unsqueeze(-1) * c1.unsqueeze(-2) + self.slope * self.rel
        v = c2.unsqueeze(-2)
        c_in = (torch.softmax(sc + self.mA, -1) * v).sum(-1)     # carry into slot i
        c_out = (torch.softmax(sc + self.mB, -1) * v).sum(-1)    # carry out of slot i
        y = z + self.wo[0] * c_in + self.wo[1] * c_out
        V = self.U
        return (2.0 * y.unsqueeze(-1) * V - V * V)[:, 1:, :]


def build_model():
    model = TinyAdder(WEIGHTS)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    meta = {
        'name': 'TinyAdder',
        'param_count': n,
        'architecture': 'single-layer transformer, 1 attention head under two causal masks',
        'd_model': 1,
        'n_layers': 1,
        'n_heads': 1,
        'vocab': 10,
        'seq_len': NSLOT,
        'max_operand': 10 ** 14 - 1,
    }
    return model, meta


@torch.no_grad()
def add(model, a: int, b: int) -> int:
    """Exact sum of two integers in [0, 99999999999999], from one forward pass."""
    oh = torch.zeros(1, NSLOT, 10)
    oh[0, 0, 0] = 2.0                                    # sentinel pair (0,0)
    for k, n in ((0, int(a)), (1, int(b))):
        n = int(n)
        for i in range(NPOS):
            oh[0, i + 1, n % 10] += 1.0
            n //= 10
    digits = model(oh)[0].argmax(-1).tolist()
    out = 0
    for i in range(NPOS - 1, -1, -1):
        out = out * 10 + digits[i]
    return out
