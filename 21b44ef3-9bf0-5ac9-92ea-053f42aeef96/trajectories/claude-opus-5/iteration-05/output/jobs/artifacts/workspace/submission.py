"""Minimal transformer that adds two 8-digit numbers.

One Macaron-style block over per-place digit-pair tokens:

    x  = code[a_i] + code[b_i]        shared 10-entry code table
    x += FFN1(x)                      pre-attention, token-wise
    x += w_o * Attention(x)           1 head, learned relative-position bias
    x += FFN2(x)                      post-attention, token-wise
    logits[d] = -|x - code[d]|^2      readout tied to the same code table

Sequence layout is LSB-first, length 10: position 0 holds a (0,0) pair that
supplies the "no carry in" default, positions 1..8 hold decimal places 0..7,
and position 9 is the carry-out slot.  Position p predicts sum digit p-1, so
all nine answer digits come from a single forward pass.

The carry chain is what the attention is for: place p has to look back past a
run of transparent places (a_i + b_i == 9) to the nearest place that either
generates a carry (a_i + b_i >= 10) or absorbs it, and that lookup depends on
the digits, not on position alone.

Weights below were produced by training this architecture from scratch
(see train_ens.py / cascade.py in the workspace).
"""

import torch
import torch.nn as nn


N_POS = 10

PIN_ROWS = [1, 0]


class Adder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        cfg = dict(cfg)
        self.cfg = cfg
        d, u1, u2 = cfg['d'], cfg['u1'], cfg['u2']
        nfix = cfg['code_fix']

        # ---- code table: a few rows are pinned constants, the rest learned ----
        cd = cfg['code_dim'] or d
        pinned = PIN_ROWS[:nfix]
        pin = torch.zeros(max(nfix, 1), cd)
        for i, row in enumerate(pinned):
            if row == 1:
                pin[i, 0] = 1.0
        self.register_buffer('code_pin', pin)
        free = [i for i in range(10) if i not in pinned]
        self.code_p = nn.Parameter(torch.randn(len(free), cd) * 0.6)
        order = torch.zeros(10, dtype=torch.long)
        for j, row in enumerate(pinned):
            order[row] = j
        for j, row in enumerate(free):
            order[row] = nfix + j
        self.register_buffer('code_order', order)

        # ---- pre-attention FFN ----
        if not cfg['f1_id_in']:
            self.f1_w = nn.Parameter(torch.randn(d, u1) * 0.8)
        self.f1_b = nn.Parameter(torch.randn(u1) * 0.5)
        if cfg['f1_out_map'] is not None:
            self.register_buffer('f1_map', self._axis_map(cfg['f1_out_map'], d))
            self.f1_o = nn.Parameter(torch.randn(u1) * 0.5)
        else:
            self.f1_o = nn.Parameter(
                torch.randn(u1, 1 if cfg['f1_out_axis'] else d) * 0.5)

        # ---- attention ----
        if cfg['q_weight']:
            self.q_w = nn.Parameter(torch.randn(d, 1) * 0.5)
        if cfg['q_bias']:
            self.q_b = nn.Parameter(torch.randn(1) * 0.5)
        if cfg['k_weight']:
            self.k_w = nn.Parameter(torch.randn(d, 1) * 0.5)
        if cfg['alibi_fix'] == 0.0:
            self.alibi = nn.Parameter(torch.tensor([-1.0]))
        else:
            self.register_buffer('alibi_c', torch.tensor([float(cfg['alibi_fix'])]))
        if cfg['self_bias']:
            self.s_b = nn.Parameter(torch.zeros(1))
        if cfg['w_o']:
            self.w_o = nn.Parameter(torch.tensor([0.5]))
        if cfg['attn_v']:
            self.v_w = nn.Parameter(torch.randn(d, 1) * 0.7)
            self.o_w = nn.Parameter(torch.randn(1, d) * 0.7)
        if cfg['logit_scale']:
            self.tau = nn.Parameter(torch.tensor([1.0]))

        # ---- post-attention FFN ----
        if not cfg['f2_id_in']:
            self.f2_w = nn.Parameter(torch.randn(d, u2) * 0.8)
        self.f2_b = nn.Parameter(torch.randn(u2) * 0.5)
        if cfg['f2_out_map'] is not None:
            self.register_buffer('f2_map', self._axis_map(cfg['f2_out_map'], d))
            self.f2_o = nn.Parameter(torch.randn(u2) * 0.5)
        else:
            self.f2_o = nn.Parameter(
                torch.randn(u2, 1 if cfg['f2_out_axis'] else d) * 0.5)

        pos = torch.arange(N_POS)
        self.register_buffer('dist', (pos[:, None] - pos[None, :]).float())
        eye = torch.eye(N_POS, dtype=torch.bool)
        m = pos[:, None] >= pos[None, :]
        if cfg['strict_causal']:
            m = (m & ~eye)
            m[0, 0] = True          # position 0 has no history; let it see itself
        self.register_buffer('mask', m)
        self.register_buffer('eye', eye.float())

    # ------------------------------------------------------------------
    @staticmethod
    def _axis_map(axes, d):
        m = torch.zeros(len(axes), d)
        for i, a in enumerate(axes):
            m[i, a] = 1.0
        return m

    def code(self):
        if self.cfg['code_fix']:
            allc = torch.cat([self.code_pin[:self.cfg['code_fix']], self.code_p], 0)
            c = allc.index_select(0, self.code_order)
        else:
            c = self.code_p
        pad = self.cfg['d'] - c.shape[-1]
        if pad > 0:
            c = torch.cat([c, c.new_zeros(c.shape[0], pad)], -1)
        return c

    def _norm(self, x, where=''):
        t = float(self.cfg['prenorm'])
        if self.cfg['norm_t'] is not None:
            t *= float(self.cfg['norm_t'].get(where, 1.0))
        if t == 0.0 or where in self.cfg['norm_skip']:
            return x
        r = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        return x * (r if t == 1.0 else t * r + (1.0 - t))

    def _ffn(self, x, w, b, o, id_in, out_axis, d, amap=None):
        h = torch.relu((x[..., :1] if id_in else x @ w) + b)
        if amap is not None:
            return (h * o) @ amap
        y = h @ o
        if out_axis and d > 1:
            y = torch.cat([y, y.new_zeros(y.shape[:-1] + (d - 1,))], -1)
        return y

    def forward(self, da, db, want_attn=False):
        """da, db: (B, N_POS) long digit tensors -> logits (B, N_POS, 10)."""
        cfg = self.cfg
        d = cfg['d']
        code = self.code()
        x = code[da] + code[db]

        x = x + self._ffn(self._norm(x, '1'), getattr(self, 'f1_w', None), self.f1_b,
                          self.f1_o, cfg['f1_id_in'], cfg['f1_out_axis'], d,
                          getattr(self, 'f1_map', None))

        xn = self._norm(x, 'a')
        ka = cfg['k_axis']
        k = xn @ self.k_w if cfg['k_weight'] else xn[..., ka:ka + 1]
        if cfg['q_weight']:
            q = xn @ self.q_w
            if cfg['q_bias']:
                q = q + self.q_b
        else:
            q = self.q_b.expand_as(k)
        alibi = self.alibi if cfg['alibi_fix'] == 0.0 else self.alibi_c
        att = q * k.transpose(-1, -2) + alibi * self.dist
        if cfg['self_bias']:
            att = att + self.s_b * self.eye
        att = att.masked_fill(~self.mask, -1e9)
        p = torch.softmax(att, -1)
        a_out = (p @ (xn @ self.v_w)) @ self.o_w if cfg['attn_v'] else p @ xn
        if cfg['w_o']:
            a_out = a_out * self.w_o
        x = x + a_out

        x = x + self._ffn(self._norm(x, '2'), getattr(self, 'f2_w', None), self.f2_b,
                          self.f2_o, cfg['f2_id_in'], cfg['f2_out_axis'], d,
                          getattr(self, 'f2_map', None))

        logits = -((x[..., None, :] - code) ** 2).sum(-1)
        if cfg['logit_scale']:
            logits = logits * self.tau
        return (logits, p) if want_attn else logits



_CFG = {'d': 2, 'u1': 4, 'u2': 4, 'code_fix': 1, 'code_dim': 0, 'f1_id_in': False, 'f2_id_in': False, 'f1_out_axis': False, 'f2_out_axis': False, 'f1_out_map': None, 'f2_out_map': None, 'norm_skip': '', 'norm_t': None, 'q_weight': False, 'q_bias': True, 'k_weight': False, 'k_axis': 1, 'alibi_fix': 0.0, 'self_bias': False, 'strict_causal': True, 'w_o': True, 'logit_scale': False, 'attn_v': False, 'prenorm': True}

_META = {'name': 'tiny-8digit-adder', 'architecture': 'single Macaron transformer block, 1 attention head', 'residual_width': 2, 'sequence_length': 10, 'trained': 'from scratch on synthetic 8-digit addition', 'note': 'restore verified 61-param model'}

_W = {
    'code_p': [[1.1615049839019775, 0.41143545508384705], [0.8402999043464661, -0.35296395421028137], [0.6573815941810608, -0.696467936038971], [0.4368191361427307, -1.0186429023742676], [0.2052554041147232, -1.2981001138687134], [-0.03513237461447716, -1.5308618545532227], [-0.2768687903881073, -1.7218998670578003], [-0.5038529634475708, -1.86318838596344], [-0.7219760417938232, -1.95317542552948]],
    'f1_w': [[16.241960525512695, -39.38740158081055, 1.6833323240280151, 8.775093078613281], [8.43957805633545, -2.069721221923828, 25.589929580688477, 16.280397415161133]],
    'f1_b': [5.183365345001221, 7.90153694152832, 4.004864692687988, 2.2195558547973633],
    'f1_o': [[27.395225524902344, -0.6262986660003662], [-28.700496673583984, -1.0231395959854126], [27.512968063354492, 2.7357394695281982], [18.236469268798828, -3.2640700340270996]],
    'q_b': [19.29569435119629],
    'alibi': [-3.016254425048828],
    'w_o': [35.20893096923828],
    'f2_w': [[0.953157901763916, 0.7984033226966858, 16.9450626373291, -21.64040184020996], [-6.569928169250488, -14.925140380859375, 17.611888885498047, -29.929685592651367]],
    'f2_b': [-2.6434895992279053, -1.8204457759857178, 14.234868049621582, 25.71873664855957],
    'f2_o': [[-17.574861526489258, 0.7505522966384888], [-16.663244247436523, 6.726356029510498], [6.163936138153076, -19.741870880126953], [18.08563995361328, -6.436252593994141]],
}


def build_model():
    """Returns (model, metadata)."""
    model = Adder(_CFG)
    sd = dict(model.named_parameters())
    for name, flat in _W.items():
        p = sd[name]
        t = torch.tensor(flat, dtype=torch.float32).reshape(p.shape)
        with torch.no_grad():
            p.copy_(t)
    model.eval()
    meta = dict(_META)
    meta['parameters'] = sum(p.numel() for p in model.parameters())
    return model, meta


def add(model, a, b):
    """Exact sum of two 8-digit integers, decoded from one forward pass."""
    da = [0] + [(int(a) // 10 ** i) % 10 for i in range(8)] + [0]
    db = [0] + [(int(b) // 10 ** i) % 10 for i in range(8)] + [0]
    dev = next(model.parameters()).device
    ta = torch.tensor([da], dtype=torch.long, device=dev)
    tb = torch.tensor([db], dtype=torch.long, device=dev)
    with torch.no_grad():
        pred = model(ta, tb).argmax(-1)[0].tolist()
    return sum(pred[p] * 10 ** (p - 1) for p in range(1, N_POS))
