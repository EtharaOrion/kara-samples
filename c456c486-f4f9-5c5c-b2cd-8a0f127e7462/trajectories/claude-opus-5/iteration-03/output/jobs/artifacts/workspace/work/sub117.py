"""Minimal transformer that performs 14-digit decimal addition in a single forward pass.

Architecture (1 layer, 1 head, d_model=2, d_head=1, d_ff=4; 117 parameters):

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
        k = x @ self.wk
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
    "U": [-2.362666606903076, -2.1896603107452393, -1.999707579612732, -1.8338655233383179, -1.6574528217315674, -1.4820317029953003, -1.3127095699310303, -1.1322662830352783, -0.9710729122161865, -0.7625499963760376],
    "P": [-0.06951562315225601, -0.7576082348823547, -1.1017342805862427, -1.4088956117630005, -1.5478343963623047, -1.6937410831451416, -1.8498907089233398, -2.007960319519043, -2.179875373840332, 26.522785186767578, -1.106256127357483, -1.4058914184570312, -1.5502665042877197, -1.694648027420044, -1.8485108613967896, -2.0104987621307373, -2.1771740913391113, 26.988691329956055, -0.2828466594219208, -1.5441113710403442, -1.6906507015228271, -1.8468116521835327, -2.0074028968811035, -2.1759092807769775, 26.57698631286621, -0.2846108078956604, -0.5769819617271423, -1.8497294187545776, -2.0113680362701416, -2.178074359893799, 26.77513885498047, -0.2853560447692871, -0.5849617123603821, -0.7711068391799927, -2.178800106048584, 26.65511131286621, -0.28550490736961365, -0.5822858810424805, -0.777627170085907, -1.068610429763794, -0.2852828800678253, -0.583271861076355, -0.7746279835700989, -1.0752147436141968, -1.2255278825759888, -0.7761545777320862, -1.0735292434692383, -1.2317498922348022, -1.3749186992645264, -1.2273786067962646, -1.3811676502227783, -1.5307362079620361, -1.5395102500915527, -1.6930489540100098, -1.8548812866210938],
    "wq": [0.21327587962150574, 0.0021083380561321974],
    "wk": [-0.08430510014295578, 1.8175709247589111],
    "wv": [-1.1316801309585571, 0.005193536169826984],
    "wo": [0.8010532855987549, 0.23525886237621307],
    "bq": [-3.7936148643493652],
    "slope": [17.66219139099121],
    "W1": [-1.100215196609497, 1.3802517652511597, -2.428112745285034, 0.0014405278488993645, 6.288822650909424, -8.190847396850586, 1.5404832363128662, 0.006481992080807686],
    "b1": [3.076580286026001, -2.5036022663116455, -1.2888262271881104, -0.1829552948474884],
    "W2": [-0.11759600043296814, 6.0615339279174805, 13.521803855895996, 1.444807767868042, -2.210205078125, -0.48078393936157227, -1.1957190036773682, -0.336350679397583],
    "b2": [-10.577238082885742, -23.294897079467773],
    "Wr": [-2.834425449371338, -11.085433006286621, -14.520356178283691, -11.41836166381836, -6.254008769989014, -2.061246871948242, 0.5605664849281311, 2.4214577674865723, 3.815979242324829, 4.899481773376465, 1.6586146354675293, -3.7729132175445557, -11.54345703125, -20.37186622619629, -23.303165435791016, -21.29881477355957, -17.178430557250977, -11.719626426696777, -5.160294055938721, 2.5694785118103027],
}

