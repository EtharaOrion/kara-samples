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
    "U": [-9.656909942626953, -8.125764846801758, -6.93660831451416, -5.2815022468566895, -3.820465326309204, -2.487891674041748, -0.8691200017929077, 0.2913004755973816, 1.9954936504364014, 3.510091781616211],
    "P": [0.16540135443210602, 0.7864788770675659, 1.0482077598571777, 3.249743700027466, 2.89495849609375, 2.561525583267212, 2.1813604831695557, 1.877702236175537, 1.3670854568481445, -36.00458908081055, 1.0221116542816162, 1.0577797889709473, 2.882810115814209, 2.5282108783721924, 2.1950440406799316, 1.816853642463684, 1.4285863637924194, -35.91093444824219, -0.3003193140029907, 1.0843842029571533, 2.57273006439209, 2.2182722091674805, 1.886610984802246, 1.4238903522491455, -37.638694763183594, -0.28920942544937134, -0.12830345332622528, 2.1867380142211914, 1.8340381383895874, 1.417853832244873, -35.97780990600586, 2.681558847427368, 1.8659391403198242, 1.2243475914001465, 1.368808627128601, -36.476165771484375, -0.2982572019100189, 1.9057978391647339, 1.2331180572509766, 0.8621413707733154, 2.679237127304077, 1.877535343170166, 1.2934366464614868, 0.8916281461715698, 0.5290442109107971, 1.2187100648880005, 0.9055333733558655, 0.5119947195053101, 0.14980250597000122, 0.600674569606781, 0.20772218704223633, -0.1546757072210312, -0.18580293655395508, -0.5957649350166321, -2.1523847579956055],
    "wq": [0.012788199819624424, 0.00442701717838645],
    "wk": [-0.11382125318050385, -0.914337694644928],
    "wv": [0.25346845388412476, 0.007242351770401001],
    "wo": [-0.7803341150283813, 0.09718849509954453],
    "bq": [-2.143275737762451],
    "bk": [0.45898276567459106],
    "slope": [6.909710884094238],
    "W1": [-0.09071887284517288, 1.3781312704086304, 0.5410044193267822, 1.7650564908981323, 6.017016410827637, 6.14725399017334, 8.606654167175293, 11.848278045654297],
    "b1": [5.012845516204834, 3.3914501667022705, 3.6932437419891357, -3.3904144763946533],
    "W2": [0.15001659095287323, 6.804939270019531, 6.071864128112793, 5.224061489105225, 6.758073806762695, -7.688946723937988, -16.095394134521484, 6.990910530090332],
    "b2": [-5.724803924560547, 0.7061372995376587],
    "Wr": [-17.233278274536133, -9.51552963256836, -5.214763164520264, -1.4521920680999756, 1.7478145360946655, 4.546708583831787, 6.818654537200928, 8.735512733459473, 10.433760643005371, 11.907002449035645, -0.9394837021827698, 2.843334197998047, 4.034899711608887, 3.933699607849121, 3.357527256011963, 2.328322649002075, 0.9096689224243164, -0.998384952545166, -3.663300037384033, -9.683022499084473],
}

