"""Minimal transformer for exact addition of two 14-digit integers (36 parameters).

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
  output mlp  a width-3 ReLU MLP over (z_i, carry_i) whose output is added
              back onto the digit-code channel: that sum-plus-carry fold, and
              the wrap at ten, is the only place the answer digit is formed, so
              the slot's residual y is meant to *be* the code of the answer
              digit.
  readout     nearest-neighbour decode of y against the same learned digit code:
              logit_d = -(y - U[d])^2, expanded and with the class-independent
              -y^2 dropped.  It costs no parameters of its own, and it is what
              lets the residual be one channel wide: a single readout column
              would make the logits linear in U[d], so the argmax could only
              ever be the largest or smallest code, never an interior digit.

Every scalar the answer depends on is a registered parameter above, and all of
them were produced by gradient training (AdamW + OneCycle) that I ran on
randomly generated operand pairs; the training code lives in train.py.
"""
import torch
import torch.nn as nn

NPOS = 15                     # digit slots: 14-digit operands -> 15-digit sum
L = NPOS + 1                  # + sentinel slot

_W = {
    'U': [-18.51323890686035, -15.11837100982666, -11.826911926269531, -8.505044937133789, -5.182139873504639, -1.8591010570526123, 1.463804006576538, 4.785670757293701, 8.077130317687988, 11.47199821472168],
    'W1a': [0.3435690402984619, 0.5742140412330627, 0.5778563022613525],
    'b1a': [2.533165693283081, 5.567855358123779, 2.251861572265625],
    'W2a0': [10.581645011901855, -3.7979178428649902, -2.5174219608306885],
    'W2a1': [0.18399649858474731, 0.7933167219161987, -0.8977132439613342],
    'bq': 27.42870330810547,
    'W1b0': [4.8783488273620605, 7.183454513549805, -0.00021300211665220559],
    'W1b1': [2.359010696411133, 3.471771478652954, 0.26350247859954834],
    'b1b': [31.86701011657715, 50.57632827758789, 4.276106357574463],
    'W2b': [13.419179916381836, -9.11309814453125, 4.336572647094727],
    'slope': 7.524038791656494,
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
        c2 = (ha * self.W2a1).sum(-1)
        score = (c1[..., :, None] + self.bq) * c1[..., None, :] + self.slope * self.rel
        attn = torch.softmax(score.masked_fill(self.blocked, float('-inf')), -1)
        carry = (attn @ c2[..., None]).squeeze(-1)

        hb = torch.relu(z[..., None] * self.W1b0 + carry[..., None] * self.W1b1 + self.b1b)
        y = z + hb @ self.W2b
        return 2.0 * y[..., None] * self.U - self.U * self.U



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
