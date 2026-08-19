"""Minimal transformer that performs 14-digit decimal addition in a single forward pass.

Architecture (1 layer, 1 head, d_model=2, d_head=1, d_ff=4; 118 parameters):

  Tokenisation.  The two operands are zero padded to 15 digits and read least
  significant first.  Slot i of the 16-slot sequence carries the digit pair
  (a[i-1], b[i-1]); slot 0 is a phantom (0,0) pair that acts as the sentinel.
  Each slot is embedded by two learned tables: a per-digit code U summed over the
  two digits (U[a]+U[b], channel 0) and an embedding P of the unordered digit-pair
  token (channel 1).  Remaining channels start at zero and act as scratch space.

  Attention.  A single strictly-causal head with scalar queries and keys,
  score(i,j) = (q_i + bq)(k_j + bk) + slope * (j - i).  The scores are genuinely
  content dependent: the key is driven by the pair embedding, so a slot whose
  digits sum to 9 (a carry *propagator*) is scored down and skipped over, while
  the learned recency slope makes the query select the nearest surviving slot.
  Each output position therefore reads the nearest earlier slot that resolves the
  carry, and the value channel reports whether that slot generated one.

  MLP and readout.  A residual ReLU MLP combines the local digit sum with the
  attended carry, and a bias-free linear readout scores the ten digit classes.

`add` runs exactly one forward pass and decodes the 15 digit argmaxes.
"""

import torch
import torch.nn as nn

__all__ = ["TinyAdder", "build_model", "add"]

NSLOT = 16          # 1 sentinel + 15 digit positions
NDIG = 15           # digits of the sum
D_MODEL = 2
D_FF = 4


def _pair_token_table():
    """Ordered digit pair -> unordered pair-token id (0..54).  Pure index arithmetic."""
    d = torch.arange(10)
    lo = torch.minimum(d[:, None], d[None, :])
    hi = torch.maximum(d[:, None], d[None, :])
    return (lo * (21 - lo)) // 2 + (hi - lo)


class TinyAdder(nn.Module):
    """Single-layer decoder-only transformer over 16 digit-pair slots."""

    def __init__(self):
        super().__init__()
        self.U = nn.Parameter(torch.zeros(10))              # per-digit code, summed over operands
        self.P = nn.Parameter(torch.zeros(55))              # unordered digit-pair embedding
        self.wq = nn.Parameter(torch.zeros(D_MODEL))
        self.wk = nn.Parameter(torch.zeros(D_MODEL))
        self.wv = nn.Parameter(torch.zeros(D_MODEL))
        self.wo = nn.Parameter(torch.zeros(D_MODEL))
        self.bq = nn.Parameter(torch.zeros(()))
        self.bk = nn.Parameter(torch.zeros(()))
        self.slope = nn.Parameter(torch.zeros(()))          # learned ALiBi-style recency term
        self.W1 = nn.Parameter(torch.zeros(D_MODEL, D_FF))
        self.b1 = nn.Parameter(torch.zeros(D_FF))
        self.W2 = nn.Parameter(torch.zeros(D_FF, D_MODEL))
        self.b2 = nn.Parameter(torch.zeros(D_MODEL))
        self.Wr = nn.Parameter(torch.zeros(D_MODEL, 10))    # digit readout
        self.register_buffer("pair_token", _pair_token_table(), persistent=False)
        idx = torch.arange(NSLOT)
        self.register_buffer("rel", (idx[None, :] - idx[:, None]).float(), persistent=False)
        # Strictly causal: the carry into slot i is determined by slots strictly
        # before i, so a slot must not attend to itself.  Slot 0 self-attends only
        # to keep its softmax row finite.
        causal = idx[None, :] < idx[:, None]
        causal[0, 0] = True
        self.register_buffer("causal", causal, persistent=False)
        self._load()

    def _load(self):
        with torch.no_grad():
            for name, values in _WEIGHTS.items():
                p = getattr(self, name)
                p.copy_(torch.tensor(values, dtype=torch.float32).reshape(p.shape))

    def forward(self, a_dig, b_dig):
        """a_dig, b_dig: int64 [B, 16] digit ids (slot 0 is the (0,0) sentinel).

        Returns logits [B, 15, 10] for the digits of the sum, least significant first.
        """
        pair = self.pair_token[a_dig, b_dig]                        # [B,16]
        c = self.U[a_dig] + self.U[b_dig]                           # compositional digit code
        e = self.P[pair]                                            # pair-token embedding
        chans = [c, e] + [torch.zeros_like(c)] * (D_MODEL - 2)
        x = torch.stack(chans, dim=-1)                              # [B,16,D_MODEL]

        q = x @ self.wq + self.bq
        k = x @ self.wk + self.bk
        scores = q[:, :, None] * k[:, None, :] + self.slope * self.rel
        scores = scores.masked_fill(~self.causal, float("-inf"))
        att = torch.softmax(scores, dim=-1)
        o = att @ (x @ self.wv)[..., None]                          # [B,16,1]
        x = x + o * self.wo

        h = torch.relu(x @ self.W1 + self.b1)
        x = x + h @ self.W2 + self.b2

        x = x[:, 1:, :]                                             # slots 1..15 hold the answer
        logits = x @ self.Wr
        return logits


def build_model():
    model = TinyAdder()
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    metadata = {
        "name": "TinyAdder",
        "task": "addition of two integers in [0, 99999999999999]",
        "architecture": "1-layer 1-head decoder-only transformer, d_model=2, d_head=1, d_ff=4",
        "n_parameters": n,
        "tokenization": "16 slots of (digit_a, digit_b) pairs, least significant first, 0-padded",
        "decoding": "single forward pass, per-position argmax over 10 digit classes",
    }
    return model, metadata


@torch.no_grad()
def add(model, a, b):
    """Exact sum of two integers in [0, 99999999999999], from one forward pass."""
    a, b = int(a), int(b)
    da = [0] + [(a // 10 ** i) % 10 for i in range(NDIG)]
    db = [0] + [(b // 10 ** i) % 10 for i in range(NDIG)]
    dev = model.U.device
    a_dig = torch.tensor([da], dtype=torch.long, device=dev)
    b_dig = torch.tensor([db], dtype=torch.long, device=dev)
    digits = model(a_dig, b_dig).argmax(-1)[0].tolist()
    return sum(d * 10 ** i for i, d in enumerate(digits))


# --------------------------------------------------------------------------- weights
# Trained with AdamW + OneCycle on freshly sampled digit pairs; see the companion
# training script.  Values are exact float32 round-trips of the trained tensors.
_WEIGHTS = {
    "U": [-5.566307067871094, -4.139528751373291, -2.7324767112731934, -1.3523719310760498, 0.04080422595143318, 1.4258720874786377, 2.808765411376953, 4.212252616882324, 5.619709491729736, 7.101038455963135],
    "P": [0.2830841839313507, 0.13946841657161713, -0.0012412944342941046, -0.14069940149784088, -0.277371346950531, -0.4154062569141388, -0.5655792355537415, -0.7106090188026428, -0.85753333568573, -16.933500289916992, -0.004422333091497421, -0.14146584272384644, -0.2758844494819641, -0.4163016676902771, -0.5630178451538086, -0.7112056016921997, -0.8603280782699585, -17.17386245727539, -0.22187037765979767, -0.2733078896999359, -0.41640323400497437, -0.5651991963386536, -0.7109013199806213, -0.8594417572021484, -17.20988655090332, -0.20643483102321625, -0.3520813286304474, -0.5665097832679749, -0.7124238610267639, -0.8602374196052551, -17.161508560180664, -0.2078189104795456, -0.32783061265945435, -0.4666823744773865, -0.8597425222396851, -17.168027877807617, -0.21063928306102753, -0.3243386149406433, -0.4430418908596039, -0.5891488194465637, -0.20919755101203918, -0.319723516702652, -0.4369807541370392, -0.5573180317878723, -0.7071512341499329, -0.4316670298576355, -0.5490850210189819, -0.6675286889076233, -0.8199207186698914, -0.6668312549591064, -0.817286491394043, -0.9654316306114197, -0.9614574313163757, -1.0953646898269653, -1.3379701375961304],
    "wq": [0.004999179393053055, -0.003098192159086466],
    "wk": [0.16081154346466064, 2.533219575881958],
    "wv": [-0.13778425753116608, -1.3691926002502441],
    "wo": [-1.0705293416976929, 0.10408131033182144],
    "bq": [3.9615304470062256],
    "bk": [-0.5746393799781799],
    "slope": [11.859898567199707],
    "W1": [2.931431531906128, 3.4454658031463623, -1.3954713344573975, -2.6263680458068848, -2.6464669704437256, 0.11764207482337952, -3.2268903255462646, 0.9526717662811279],
    "b1": [4.085608959197998, -5.728255748748779, 4.797271251678467, -3.2260963916778564],
    "W2": [-3.442347288131714, 0.7613462805747986, 3.8098514080047607, -1.4823774099349976, 1.934292197227478, -3.0081236362457275, -2.3217625617980957, 2.736246347427368],
    "b2": [12.738992691040039, 15.081215858459473],
    "Wr": [-16.581361770629883, -8.723631858825684, -1.5386327505111694, 4.220122814178467, 8.055326461791992, 10.068376541137695, 10.34971809387207, 9.147870063781738, 6.798843860626221, 2.286465883255005, -1.5476206541061401, 4.680253505706787, 9.431117057800293, 12.298008918762207, 13.035597801208496, 11.783563613891602, 8.780682563781738, 4.4009013175964355, -0.8926634192466736, -8.888846397399902],
}

