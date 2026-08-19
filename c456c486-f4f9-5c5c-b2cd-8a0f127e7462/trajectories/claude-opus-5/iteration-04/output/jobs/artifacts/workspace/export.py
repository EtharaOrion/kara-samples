"""Write a trained checkpoint into /workspace/submission.py (model + weights only)."""
import argparse, torch

HEADER = '''"""Minimal transformer for exact addition of two 14-digit integers ({NP} parameters).

One attention layer, one head.  Slot 0 is a phantom (0, 0) sentinel; slots 1..15
hold the digit pair (a_i, b_i) of the two operands, least significant digit
first, and slot i predicts digit i of the sum.

  embedding   z_i = U[a_i] + U[b_i], a learned additive digit code.
  feature mlp a width-{DA} ReLU MLP turns z into two features, c1 (attention key)
              and c2 (attention value).  A linear key cannot express "these two
              digits sum to 9" from an additive code -- with an additive code
              h, h(a)+h(9-a) > h(a)+h(8) and h(1)+h(8) > h(1)+h(9) contradict --
              so this nonlinearity is what keeps the digit code 10 parameters
              wide instead of a 55-row pair table.
{FEATDOC}{ATTNDOC}{OUTDOC}
Every scalar the answer depends on is a registered parameter above, and all of
them were produced by gradient training (AdamW + OneCycle) that I ran on
randomly generated operand pairs.  None of the training code is here: train.py
in the same directory lists and runs the pipeline that produced these weights.
"""
import torch
import torch.nn as nn

NPOS = 15                     # digit slots: 14-digit operands -> 15-digit sum
L = NPOS + 1                  # + sentinel slot

_W = {W}


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

{FEATCODE}{ATTNCODE}
        attn = torch.softmax(score.masked_fill(self.blocked, float('-inf')), -1)
        carry = {CARRY}
{OUTCODE}

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
'''

FEAT_FREE = """
        ha = torch.relu(z[..., None] * self.W1a + self.b1a)
        c1 = (ha * self.W2a0).sum(-1)
        c2 = (ha * self.W2a1).sum(-1)"""
FEAT_G = """
        ha = torch.relu(z[..., None] * self.W1a + self.b1a)
        c1 = (ha * self.W2a0).sum(-1)
        c2 = (ha * self.W2a1).sum(-1) + self.k2"""
FEATDOC_G = """              Training always spends one unit of this MLP as a constant, so the
              constant is kept directly instead: in the key it would only shift
              the score by terms that cancel in the softmax, so it is folded
              into bq and gone; in the value it survives the attention average
              and is the single scalar k2.
"""

# ---------------------------------------------------------------- attention --
ATTNDOC_B = '''  attention   score(i, j) = (wq * c1_i + bq) * c1_j + slope * (j - i), strictly
              causal, slot 0 attending to itself.  Training drives c1 to a
              V shape whose minimum sits on the carry-*propagate* digit sums, so
              those slots are invisible as keys and every slot attends to the
              nearest earlier slot that settles the carry, reading that slot's
              carry-generate feature c2.  That is carry lookahead, and which
              slot is attended depends on the digits, not on position alone.
'''
ATTNCODE_B = '''
        q = c1 * self.wq + self.bq
        score = q[..., :, None] * c1[..., None, :] + self.slope * self.rel'''
CARRY_B = 'self.wv * (attn @ c2[..., None]).squeeze(-1)'

ATTNDOC_E = '''  attention   score(i, j) = (c1_i + bq) * c1_j + slope * (j - i), strictly
              causal, slot 0 attending to itself.  Training drives c1 to a
              V shape whose minimum sits on the carry-*propagate* digit sums, so
              those slots are invisible as keys and every slot attends to the
              nearest earlier slot that settles the carry, reading that slot's
              carry-generate feature c2.  That is carry lookahead, and which
              slot is attended depends on the digits, not on position alone.
              The usual query scale and value scale are omitted because they are
              redundant: c1 appears only in the score, so a query scale is the
              same as rescaling W2a0, and a value scale is the same as rescaling
              everything the attention output feeds into.
'''
ATTNCODE_E = '''
        score = (c1[..., :, None] + self.bq) * c1[..., None, :] + self.slope * self.rel'''
CARRY_E = '(attn @ c2[..., None]).squeeze(-1)'

# ------------------------------------------------------------------- output --
OUTDOC_FREE = '''  output mlp  residual ReLU MLP of width {DB} over (z_i, carry_i): it folds
              digit sum + carry into the answer digit mod 10.
  readout     bias-free linear map from the 2-channel residual to 10 logits.
'''
OUTCODE_FREE = '''
        x = torch.stack([z, carry], -1)
        hb = torch.relu(z[..., None] * self.W1b0 + carry[..., None] * self.W1b1 + self.b1b)
        x = x + hb @ self.W2b.t()
        return x @ self.Wout.t()

'''

OUTDOC_TIED = '''  output mlp  residual ReLU MLP of width {DB} over (z_i, carry_i): it folds
              digit sum + carry into the answer digit mod 10.
  readout     bias-free linear map from the 2-channel residual to 10 logits,
              with its first column tied to the input digit code U (tied
              embeddings): the readout has to score answer digit d by how well
              the folded channel matches U[d], which is exactly what the input
              code already means, so only the second column stays free.
'''
OUTCODE_TIED = '''
        x = torch.stack([z, carry], -1)
        hb = torch.relu(z[..., None] * self.W1b0 + carry[..., None] * self.W1b1 + self.b1b)
        x = x + hb @ self.W2b.t()
        return x @ torch.stack([self.U, self.Wout1], -1).t()

'''


OUTDOC_TIED_F = """  output mlp  a width-{DB} ReLU MLP over (z_i, carry_i) whose output is added
              back onto the digit-code channel: that sum-plus-carry fold, and
              the wrap at ten, is the only place the answer digit is formed.
              The residual carried alongside it is the attention's carry
              channel, so the readout sees two numbers per slot.
  readout     bias-free linear map from those 2 channels to 10 logits, with its
              first column tied to the input digit code U (tied embeddings):
              scoring answer digit d against the folded channel is exactly what
              the input code already means, so only the second column is free.
"""
OUTCODE_TIED_F = """
        hb = torch.relu(z[..., None] * self.W1b0 + carry[..., None] * self.W1b1 + self.b1b)
        x = torch.stack([z + hb @ self.W2b, carry], -1)
        return x @ torch.stack([self.U, self.Wout1], -1).t()

"""


OUTDOC_DIST = """  output mlp  a width-{DB} ReLU MLP over (z_i, carry_i) whose output is added
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
"""
OUTCODE_DIST = """
        hb = torch.relu(z[..., None] * self.W1b0 + carry[..., None] * self.W1b1 + self.b1b)
        y = z + hb @ self.W2b
        return 2.0 * y[..., None] * self.U - self.U * self.U

"""


OUTDOC_PROJ = """  attn out    the head's output is written into the residual stream through a
              single scalar wo -- the output projection of a one-channel
              residual -- so the carry reaches the answer directly as well as
              through the MLP.
  output mlp  a width-{DB} ReLU MLP over (z_i, carry_i) added onto the same
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
"""
OUTCODE_PROJ = """
        hb = torch.relu(z[..., None] * self.W1b0 + carry[..., None] * self.W1b1 + self.b1b)
        y = z + self.wo * carry + hb @ self.W2b
        return 2.0 * y[..., None] * self.U - self.U * self.U

"""


def fmt(x):
    if isinstance(x, list):
        return '[' + ', '.join(fmt(v) for v in x) + ']'
    return repr(float(x))


def db_units(W):
    return len(W['b1b'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='/workspace/ckpt_b2.pt')
    ap.add_argument('--arch', default='b', choices=['b', 'e', 'f', 'g', 'h', 'i'])
    ap.add_argument('--index', type=int, default=-1)
    ap.add_argument('--out', default='/workspace/submission.py')
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location='cpu')
    i = ck['best'] if args.index < 0 else args.index
    W, total = {}, 0
    for k, v in ck['params'].items():
        t = v[i]
        total += t.numel()
        W[k] = t.tolist() if t.dim() > 0 else float(t)
    tied = 'Wout1' in W
    body = '{\n' + ''.join(f"    {k!r}: {fmt(v)},\n" for k, v in W.items()) + '}'
    attndoc, attncode, carry = ((ATTNDOC_B, ATTNCODE_B, CARRY_B) if args.arch == 'b'
                                else (ATTNDOC_E, ATTNCODE_E, CARRY_E))
    if args.arch == 'i':
        assert 'wo' in W and 'Wout1' not in W, 'arch i: output projection, weightless readout'
        outdoc, outcode = OUTDOC_PROJ, OUTCODE_PROJ
    elif args.arch == 'h':
        assert 'Wout1' not in W and 'Wout' not in W, 'arch h: the readout has no weights of its own'
        outdoc, outcode = OUTDOC_DIST, OUTCODE_DIST
    elif args.arch in ('f', 'g'):
        assert tied and len(W['W2b']) == db_units(W), 'arch f: tied readout, one residual write row'
        outdoc, outcode = OUTDOC_TIED_F, OUTCODE_TIED_F
    else:
        outdoc, outcode = (OUTDOC_TIED, OUTCODE_TIED) if tied else (OUTDOC_FREE, OUTCODE_FREE)
    da, db = len(W['b1a']), len(W['b1b'])
    featcode = FEAT_G if 'k2' in W else FEAT_FREE
    featdoc = FEATDOC_G if 'k2' in W else ''
    src = (HEADER.replace('{FEATCODE}', featcode).replace('{FEATDOC}', featdoc)
           .replace('{ATTNDOC}', attndoc).replace('{ATTNCODE}', attncode)
           .replace('{CARRY}', carry).replace('{OUTDOC}', outdoc).replace('{OUTCODE}', outcode)
           .replace('{W}', body).replace('{NP}', str(total))
           .replace('{DA}', str(da)).replace('{DB}', str(db)))
    with open(args.out, 'w') as f:
        f.write(src)
    print('wrote', args.out, 'model', i, 'params', total,
          'acc', float(ck['acc'][i]), float(ck['acc_stress'][i]))


if __name__ == '__main__':
    main()
