"""Minimal transformer for exact addition of two 14-digit integers (33 parameters).

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
  attn out    the head's output is written into the residual stream through a
              single scalar wo -- the output projection of a one-channel
              residual -- so the carry reaches the answer directly as well as
              through the MLP.
  output mlp  a width-2 ReLU MLP over (z_i, carry_i) added onto the same
              channel.  What is left for it is the part that needs a
              nonlinearity: the wrap at ten, a step in the digit code whose
              threshold moves by one digit when a carry arrives.
  readout     nearest-neighbour decode of the resulting y against the same
              learned digit code: logit_d = -(y - U[d])^2, expanded and with the
              class-independent -y^2 dropped.  It costs no parameters of its
              own, and it is what lets the residual be one channel wide: a
              single linear readout column would make the logits linear in U[d],
              so the argmax could only ever be the largest or smallest code,
              never an interior digit.

Every scalar the answer depends on is a registered parameter above, and all of
them were produced by gradient training (AdamW + OneCycle) that I ran on
randomly generated operand pairs; the training code lives in train.py.
"""
import torch
import torch.nn as nn

NPOS = 15                     # digit slots: 14-digit operands -> 15-digit sum
L = NPOS + 1                  # + sentinel slot

_W = {
    'U': [-0.18868215382099152, 3.2295446395874023, 6.477260112762451, 9.767143249511719, 13.11572265625, 16.399497985839844, 19.74803924560547, 23.037837982177734, 26.28547477722168, 29.704029083251953],
    'W1a': [0.34715113043785095, 0.5797257423400879, 0.5851418972015381],
    'b1a': [-10.2053861618042, -15.72268009185791, -19.174856185913086],
    'W2a0': [10.581526756286621, -3.796772003173828, -2.5169677734375],
    'W2a1': [0.1870310753583908, 0.8046388626098633, -0.9078782200813293],
    'bq': 27.34415054321289,
    'W1b0': [4.877799987792969, 7.182590961456299],
    'W1b1': [2.3569610118865967, 3.4766364097595215],
    'b1b': [-149.0514678955078, -215.84622192382812],
    'W2b': [13.419677734375, -9.112606048583984],
    'slope': 7.6143879890441895,
    'wo': 1.2163795232772827,
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
        y = z + self.wo * carry + hb @ self.W2b
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
