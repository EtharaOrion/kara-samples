"""Minimal transformer that performs 14-digit decimal addition in a single forward pass.

Architecture (1 layer, 1 head, d_model=2, d_head=1, d_ff=4; 115 parameters):

  Tokenisation.  The two operands are zero padded to 15 digits and read least
  significant first.  Slot i of the 16-slot sequence carries the digit pair
  (a[i-1], b[i-1]); slot 0 is a phantom (0,0) pair that acts as the sentinel.
  Each slot is embedded by two learned tables: a per-digit code U summed over the
  two digits (U[a]+U[b], channel 0) and an embedding P of the unordered digit-pair
  token (channel 1).  Remaining channels start at zero and act as scratch space.

  Attention.  A single strictly-causal head with scalar queries and keys,
  score(i,j) = (q_i + bq) * k_j + slope * (j - i).  The scores are genuinely
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
        self.slope = nn.Parameter(torch.zeros(()))          # learned ALiBi-style recency term
        self.W1 = nn.Parameter(torch.zeros(D_MODEL, D_FF))
        self.b1 = nn.Parameter(torch.zeros(D_FF))
        self.W2 = nn.Parameter(torch.zeros(D_FF, D_MODEL))
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
        k = x @ self.wk
        scores = q[:, :, None] * k[:, None, :] + self.slope * self.rel
        scores = scores.masked_fill(~self.causal, float("-inf"))
        att = torch.softmax(scores, dim=-1)
        o = att @ (x @ self.wv)[..., None]                          # [B,16,1]
        x = x + o * self.wo

        h = torch.relu(x @ self.W1 + self.b1)
        x = x + h @ self.W2

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
    "U": [-8.343558311462402, -6.724874496459961, -5.2392168045043945, -3.704023599624634, -2.1910228729248047, -0.6785832047462463, 0.8456299901008606, 2.3639328479766846, 3.8603174686431885, 5.504703521728516],
    "P": [-1.1578829288482666, -0.16126245260238647, 1.3623021841049194, 1.4594392776489258, 1.5685895681381226, 1.6856971979141235, 1.8087334632873535, 1.9418991804122925, 2.586289882659912, -54.474430084228516, 1.3547145128250122, 1.4537111520767212, 1.5625180006027222, 1.679220199584961, 1.8035569190979004, 1.9359761476516724, 2.5790019035339355, -54.77750778198242, -1.1797550916671753, 1.5648503303527832, 1.6799486875534058, 1.8043683767318726, 1.9384346008300781, 2.5807712078094482, -54.82191848754883, -1.1683608293533325, -1.0881884098052979, 1.8029131889343262, 1.9375728368759155, 2.5805909633636475, -54.634918212890625, -1.1710349321365356, -1.0792020559310913, 0.2181704342365265, 2.5799646377563477, -54.62383270263672, -1.1690157651901245, -1.0790573358535767, 0.22781455516815186, 0.33339789509773254, -1.1679099798202515, -1.0802701711654663, 0.22658005356788635, 0.34502097964286804, 0.45878443121910095, 0.22573700547218323, 0.3399081826210022, 0.46748512983322144, 0.5851531028747559, 0.4658583402633667, 0.5958439111709595, 0.6959906220436096, 0.7065021395683289, 0.7924202084541321, -2.681791305541992],
    "wq": [0.009848960675299168, 0.0022296386305242777],
    "wk": [-0.06341558694839478, -1.2698979377746582],
    "wv": [0.7684674263000488, 0.01947709731757641],
    "wo": [-0.45946741104125977, 0.051927629858255386],
    "bq": [-2.2321434020996094],
    "slope": [12.320496559143066],
    "W1": [-0.7400578260421753, 0.666149914264679, 0.4418938457965851, -0.031836554408073425, -1.3828794956207275, 9.684469223022461, 3.143301010131836, 6.070512294769287],
    "b1": [5.4546613693237305, -2.4706859588623047, 4.703057765960693, 9.204039573669434],
    "W2": [-0.228187695145607, -3.886263847351074, 11.814202308654785, -7.843623638153076, -5.027837753295898, 6.761507987976074, -0.8028497695922852, 7.729037761688232],
    "Wr": [-8.444341659545898, -15.347987174987793, -15.657344818115234, -11.805630683898926, -8.176868438720703, -4.678203582763672, -1.2882049083709717, 1.8004918098449707, 4.918967247009277, 7.3142828941345215, -2.3190784454345703, 0.18659240007400513, 4.51993989944458, 5.7465596199035645, 6.1434125900268555, 5.772164821624756, 4.661032676696777, 2.941821336746216, 0.4775196313858032, -3.186591863632202],
}

