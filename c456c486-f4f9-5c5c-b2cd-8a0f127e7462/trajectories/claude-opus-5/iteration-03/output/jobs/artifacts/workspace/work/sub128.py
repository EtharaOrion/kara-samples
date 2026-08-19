"""Minimal transformer that performs 14-digit decimal addition in a single forward pass.

Architecture (1 layer, 1 head, d_model=2, d_head=1, d_ff=4; 128 parameters):

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
  attended carry, and a linear readout scores the ten digit classes.

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
        self.br = nn.Parameter(torch.zeros(10))
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
        logits = x @ self.Wr + self.br
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
    "U": [-5.68878173828125, -4.239926815032959, -2.8671255111694336, -1.4479188919067383, -0.055331069976091385, 1.3313837051391602, 2.728001117706299, 4.080044746398926, 5.477969646453857, 6.93590784072876],
    "P": [-0.44703930616378784, -0.4613710641860962, -0.4737197160720825, -0.48760801553726196, -0.5098130702972412, -0.5436858534812927, -0.581820011138916, -0.5645303726196289, -0.5771659016609192, 10.49273681640625, -0.470255583524704, -0.48605868220329285, -0.5084051489830017, -0.5419065356254578, -0.581133246421814, -0.5822845101356506, -0.5798633694648743, 10.303267478942871, 0.01209630724042654, -0.5108709335327148, -0.5401704907417297, -0.5806487798690796, -0.5745919346809387, -0.5856516361236572, 10.587259292602539, 0.00521053746342659, -0.00663695577532053, -0.5809794664382935, -0.5826478004455566, -0.588458240032196, 10.270331382751465, 0.003945042844861746, -0.00680704927071929, -0.02522452548146248, -0.5874510407447815, 10.518036842346191, 0.010285306721925735, -0.0072809308767318726, -0.027042776346206665, -0.046765126287937164, 0.009808086790144444, -0.004538772162050009, -0.025710545480251312, -0.051606059074401855, -0.06298957020044327, -0.025947539135813713, -0.06296829879283905, -0.0728023499250412, -0.11348362267017365, -0.0758698433637619, -0.10033159703016281, -0.12083003669977188, -0.1103055402636528, -0.11134930700063705, -0.08619628846645355],
    "wq": [-0.007636331953108311, 0.005309744738042355],
    "wk": [0.09281099587678909, -2.981351613998413],
    "wv": [-0.020084328949451447, -1.6730369329452515],
    "wo": [-1.3957335948944092, 0.016039932146668434],
    "bq": [3.4410717487335205],
    "bk": [0.9536207318305969],
    "slope": [7.281537055969238],
    "W1": [0.2931363880634308, 3.926568031311035, 1.3899530172348022, 2.0210108757019043, -8.201118469238281, -0.24565207958221436, -0.8617476224899292, 0.5334596633911133],
    "b1": [0.08147090673446655, -0.0335644967854023, 5.477319240570068, -7.952719211578369],
    "W2": [8.0814847946167, -4.04525089263916, -1.1865426301956177, -3.100471258163452, -1.9165394306182861, 3.190160036087036, 6.770740985870361, 4.692516803741455],
    "b2": [-5.107141017913818, 0.8379776477813721],
    "Wr": [-10.574966430664062, -9.713415145874023, -7.919412136077881, -5.9883012771606445, -3.788022756576538, -0.9151780009269714, 2.844069004058838, 4.044814109802246, 5.181034088134766, 7.574574947357178, 7.68084716796875, -0.09071161597967148, -3.8592634201049805, -6.221817970275879, -7.204277038574219, -5.841116428375244, -1.8604329824447632, 1.8533871173858643, 5.758701324462891, 16.500675201416016],
    "br": [-6.383931636810303, 2.8262033462524414, 11.499031066894531, 9.929206848144531, 1.0035327672958374, -7.802231311798096, -14.08976936340332, 4.734177112579346, 9.090649604797363, 1.1066009998321533],
}

