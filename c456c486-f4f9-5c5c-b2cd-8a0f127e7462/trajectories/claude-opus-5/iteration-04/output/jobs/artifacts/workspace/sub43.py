"""Minimal transformer for exact addition of two 14-digit integers (43 parameters).

One attention layer, one head.  Slot 0 is a phantom (0, 0) sentinel; slots 1..15
hold the digit pair (a_i, b_i) of the two operands, least significant digit
first, and slot i predicts digit i of the sum.

  embedding   z_i = U[a_i] + U[b_i], a learned additive digit code.
  feature mlp a width-3 ReLU MLP turns z into two features, c1 (attention key)
              and c2 (attention value).  A linear key cannot express "these two
              digits sum to 9" from an additive code -- with an additive code
              h, h(a)+h(9-a) > h(a)+h(8) and h(1)+h(8) > h(1)+h(9) contradict --
              so this nonlinearity is what keeps the digit code 10 parameters
              wide instead of a 55-row pair table.
              Training always spends one unit of this MLP as a constant, so the
              constant is kept directly instead: in the key it would only shift
              the score by terms that cancel in the softmax, so it is folded
              into bq and gone; in the value it survives the attention average
              and is the single scalar k2.
  attention   score(i, j) = (c1_i + bq) * c1_j + slope * (j - i), strictly
              causal, slot 0 attending to itself.  Training drives c1 to a
              V shape whose minimum sits on the carry-*propagate* digit sums, so
              those slots are invisible as keys and every slot attends to the
              nearest earlier slot that settles the carry, reading that slot's
              carry-generate feature c2.  That is carry lookahead, and which
              slot is attended depends on the digits, not on position alone.
              The usual query scale and value scale are omitted because they are
              redundant: c1 appears only in the score, so a query scale is the
              same as rescaling W2a0, and an output scale on the attention is
              the same as rescaling W1b1 and the second readout column.
  output mlp  a width-2 ReLU MLP over (z_i, carry_i) whose output is added
              back onto the digit-code channel: that sum-plus-carry fold, and
              the wrap at ten, is the only place the answer digit is formed.
              The residual carried alongside it is the attention's carry
              channel, so the readout sees two numbers per slot.
  readout     bias-free linear map from those 2 channels to 10 logits, with its
              first column tied to the input digit code U (tied embeddings):
              scoring answer digit d against the folded channel is exactly what
              the input code already means, so only the second column is free.

Every scalar the answer depends on is a registered parameter above, and all of
them were produced by gradient training (AdamW + OneCycle) that I ran on
randomly generated operand pairs; the training code lives in train.py.
"""
import torch
import torch.nn as nn

NPOS = 15                     # digit slots: 14-digit operands -> 15-digit sum
L = NPOS + 1                  # + sentinel slot

_W = {
    'U': [-20.434972763061523, -16.587312698364258, -12.942315101623535, -9.33958911895752, -5.722752571105957, -2.1019489765167236, 1.5246251821517944, 5.161464691162109, 8.812618255615234, 12.516902923583984],
    'W1a': [0.3338991701602936, 0.5570055842399597, 0.5771393775939941],
    'b1a': [2.5061964988708496, 5.402791500091553, 2.410477876663208],
    'W2a0': [10.575891494750977, -3.8048858642578125, -2.447216510772705],
    'W2a1': [0.1511678397655487, 0.6614193320274353, -0.7256924510002136],
    'bq': 28.47103500366211,
    'W1b0': [4.524225234985352, 6.568779945373535],
    'W1b1': [2.6027333736419678, 3.7894368171691895],
    'b1b': [-8.029740333557129, -7.812430381774902],
    'W2b': [13.338190078735352, -9.184112548828125],
    'slope': 6.731841087341309,
    'Wout1': [-28.89763069152832, -19.452699661254883, -11.321048736572266, -4.066714286804199, 2.4073243141174316, 8.084413528442383, 12.96985149383545, 17.068161010742188, 20.378921508789062, 22.922645568847656],
    'k2': 15.54938793182373,
}


class TinyAdder(nn.Module):
    def __init__(self):
        super().__init__()
        for name, value in _W.items():
            self.register_parameter(name, nn.Parameter(torch.tensor(value)))
        pos = torch.arange(L)
        self.register_buffer('rel', (pos[None, :] - pos[:, None]).float())
        allow = pos[None, :] < pos[:, None]
        allow[0, 0] = True
        self.register_buffer('blocked', ~allow)

    def forward(self, a_idx, b_idx):
        """a_idx, b_idx: (B, L) digit tokens -> (B, L, 10) digit logits."""
        onehot = (nn.functional.one_hot(a_idx, 10) + nn.functional.one_hot(b_idx, 10)).float()
        z = onehot @ self.U


        ha = torch.relu(z[..., None] * self.W1a + self.b1a)
        c1 = (ha * self.W2a0).sum(-1)
        c2 = (ha * self.W2a1).sum(-1) + self.k2
        score = (c1[..., :, None] + self.bq) * c1[..., None, :] + self.slope * self.rel
        attn = torch.softmax(score.masked_fill(self.blocked, float('-inf')), -1)
        carry = (attn @ c2[..., None]).squeeze(-1)

        hb = torch.relu(z[..., None] * self.W1b0 + carry[..., None] * self.W1b1 + self.b1b)
        x = torch.stack([z + hb @ self.W2b, carry], -1)
        return x @ torch.stack([self.U, self.Wout1], -1).t()



def build_model():
    model = TinyAdder().eval()
    meta = {
        'name': 'TinyAdder',
        'parameters': sum(p.numel() for p in model.parameters()),
        'layers': 1, 'heads': 1, 'd_head': 1,
        'max_operand': 10 ** 14 - 1,
        'description': 'single-head causal transformer whose attention performs '
                       'carry lookahead over digit-pair slots',
    }
    return model, meta


@torch.no_grad()
def add(model, a, b):
    """Exact sum of two integers in [0, 99999999999999], from one forward pass."""
    device = next(model.parameters()).device
    a_tok = [0] * L
    b_tok = [0] * L
    for i in range(NPOS):
        a_tok[i + 1] = (a // 10 ** i) % 10
        b_tok[i + 1] = (b // 10 ** i) % 10
    logits = model(torch.tensor([a_tok], device=device), torch.tensor([b_tok], device=device))
    digits = logits[0, 1:].argmax(-1).tolist()
    return sum(d * 10 ** i for i, d in enumerate(digits))


if __name__ == '__main__':
    m, meta = build_model()
    print(meta)
    print(add(m, 99999999999999, 1), add(m, 12345678901234, 98765432109876))
