"""Bake a trained cell into /workspace/submission.py (model + inference only)."""
import sys, os, json, torch
sys.path.insert(0, os.path.dirname(__file__))
from core import Ens

TEMPLATE = '''"""Minimal transformer that performs 14-digit decimal addition in one forward pass.

The sequence has 16 slots: slot 0 is a phantom (0,0) sentinel, slots 1..15 carry the
digit pairs of the two operands, least-significant first (operands are zero padded to
15 digits, so the leading slot holds the final carry).

Per slot the model embeds the digit pair as a single scalar z = U[a] + U[b], a width-{WIDTH}
ReLU feature layer turns z into an attention key/query feature c1 and a value feature c2,
and one attention head scores score(i,j) = (c1_i + bq) * c1_j + slope * (j - i).  The head
is applied under two causal masks that share all projections: strictly causal (j < i) reads
the carry *into* a slot, causal-with-self (j <= i) reads the carry *out of* it.  The digit
is then read out by nearest-centre against the same embedding table U.

All weights below were obtained by gradient training (see train.py in the workspace).
"""
import torch

WEIGHTS = {WEIGHTS}

NSLOT = 16
NPOS = 15


class TinyAdder(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        t = lambda x: torch.tensor(x, dtype=torch.float32)
        self.U = torch.nn.Parameter(t(w['U']))            # digit code / readout centres
        self.b1a = torch.nn.Parameter(t(w['b1a']))        # feature thresholds
        self.W2a0 = torch.nn.Parameter(t(w['W2a0']))      # -> key/query feature
{W2A1_DECL}
        self.bq = torch.nn.Parameter(t(w['bq']))          # query bias
{SLOPE_DECL}
        self.wo = torch.nn.Parameter(t(w['wo']))          # carry-in / carry-out gains
{V_DECL}{W1A_DECL}
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
        c2 = {C2_EXPR}
        sc = (c1 + self.bq).unsqueeze(-1) * c1.unsqueeze(-2) + self.slope * self.rel
        v = c2.unsqueeze(-2)
        c_in = (torch.softmax(sc + self.mA, -1) * v).sum(-1)     # carry into slot i
        c_out = (torch.softmax(sc + self.mB, -1) * v).sum(-1)    # carry out of slot i
        y = z + self.wo[0] * c_in + self.wo[1] * c_out
        V = {VREF}
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
'''


def bake(ckpt_path, out_path='/workspace/submission.py', cell=0):
    ck = torch.load(ckpt_path, map_location='cpu')
    P = {k: v[cell] for k, v in ck['params'].items()}
    fold = ck.get('fold_sign')
    drop = ck.get('drop') or ()
    W = P['b1a'].shape[0]
    if fold is not None:
        P['W1a'] = torch.tensor(fold)
    untied = 'V' in P
    dropped = 'W2a1_0' in drop
    vname = 'W2a1p' if dropped else 'W2a1'
    order = ['U', 'W1a', 'b1a', 'W2a0', vname, 'bq', 'slope', 'wo'] + (['V'] if untied else [])
    P.setdefault('slope', torch.tensor([float(ck['fix_slope'])])) if ck.get('fix_slope') is not None else None
    w = {k: [float(x) for x in P[k].flatten()] for k in order}
    fixsl = ck.get('fix_slope')
    if fixsl is None:
        slope_decl = "        self.slope = torch.nn.Parameter(t(w['slope']))    # learned ALiBi-style decay"
    else:
        P['slope'] = torch.tensor([float(fixsl)])
        slope_decl = ("        # fixed ALiBi-style positional decay: a constant of the architecture,\n"
                      "        # as in ALiBi itself.  The attention's sharpness is learned through the\n"
                      "        # scale of W2a0 instead.\n"
                      "        self.register_buffer('slope', t(w['slope']))")
    if fold is None:
        w1a_decl = "        self.W1a = torch.nn.Parameter(t(w['W1a']))        # feature input weights"
    else:
        w1a_decl = ("        # unit input scales are redundant with W2a0/W2a1 and were folded\n"
                    "        # into them; only the fixed signs remain.\n"
                    "        self.register_buffer('W1a', t(w['W1a']))")
    if dropped:
        w2a1_decl = ("        # unit 0 is redundant in the value projection: two entries already fix\n"
                     "        # both the slope and the level of c2 on the ranges the attention reads.\n"
                     "        self.W2a1p = torch.nn.Parameter(t(w['W2a1p']))    # -> value feature")
        c2_expr = '(h[..., 1:] * self.W2a1p).sum(-1)'
    else:
        w2a1_decl = "        self.W2a1 = torch.nn.Parameter(t(w['W2a1']))      # -> value feature"
        c2_expr = '(h * self.W2a1).sum(-1)'
    v_decl = ("        self.V = torch.nn.Parameter(t(w['V']))            # readout centres\n"
              if untied else '')
    src = (TEMPLATE.replace('{WEIGHTS}', json.dumps(w, indent=4))
                   .replace('{V_DECL}', v_decl)
                   .replace('{VREF}', 'self.V' if untied else 'self.U')
                   .replace('{W1A_DECL}', w1a_decl)
                   .replace('{SLOPE_DECL}', slope_decl)
                   .replace('{W2A1_DECL}', w2a1_decl)
                   .replace('{C2_EXPR}', c2_expr)
                   .replace('{WIDTH}', str(W)))
    with open(out_path, 'w') as f:
        f.write(src)
    return out_path


if __name__ == '__main__':
    p = bake(sys.argv[1], cell=int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    print('wrote', p)
