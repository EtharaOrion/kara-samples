"""Architecture for the 8-digit adder.

One Macaron-style transformer block over per-place digit-pair tokens:

    x  = code[a_i] + code[b_i]                (shared 10-entry code table)
    x += FFN1(x)                              (pre-attention, token-wise)
    x += w_o * SelfAttention(x)               (1 head, learned relative bias)
    x += FFN2(x)                              (post-attention, token-wise)
    logits[d] = -|x - code[d]|^2              (readout tied to the code table)

Sequence layout (LSB first), length 10:
    pos 0      : (0,0)   -- leading pair, supplies the "no carry in" default
    pos 1..8   : (a_p, b_p) for decimal place p = 0..7
    pos 9      : (0,0)   -- carry-out slot
Position p predicts sum digit p-1, so all 9 answer digits come out of one
forward pass.

Every flag below removes parameters; they exist so that a trained model can be
re-fit on a smaller architecture (warm-start cascade).
"""

import torch
import torch.nn as nn

N_POS = 10

# code rows pinned by the gauge fixes, in the order they get applied.
#   code[1] = e0        uses up the residual-scale freedom
#   code[0] = 0         would need a constant on the answer axis (known to fail)
PIN_ROWS = [1, 0]


def default_cfg(**kw):
    cfg = dict(
        d=2,             # residual width
        u1=4,            # pre-attention FFN units
        u2=4,            # post-attention FFN units
        code_fix=0,      # number of code rows pinned to constants (gauge fix)
        code_dim=0,      # code writes only this many residual axes (0 -> all d)
        f1_id_in=False,  # pre-FFN reads residual axis 0 with weight 1 (no W_in)
        f2_id_in=False,  # post-FFN reads residual axis 0 with weight 1
        f1_out_axis=False,   # pre-FFN writes only to residual axis 0
        f2_out_axis=False,   # post-FFN writes only to residual axis 0
        f1_out_map=None,     # per-unit output axis, e.g. [1] (one gain per unit)
        f2_out_map=None,
        norm_skip='',    # sublayers that see the raw residual: '1', 'a', '2'
        norm_t=None,     # per-sublayer norm blend, e.g. {'1': 0.3} (training aid)
        q_weight=True,   # query = x @ Wq + bq  (False -> query = bq, constant)
        q_bias=True,
        k_weight=True,   # key = x @ Wk    (False -> key = x[..., k_axis])
        k_axis=0,
        alibi_fix=0.0,   # != 0 -> relative-position slope pinned to this value
        self_bias=True,  # learned logit bias on the diagonal
        strict_causal=False,  # True -> position i cannot attend to itself
        w_o=True,        # scalar gain on the attention output
        logit_scale=False,   # learned readout temperature (argmax-irrelevant)
        attn_v=False,    # True -> head reads x@W_v (d,1) and writes through W_o (1,d)
        prenorm=False,   # parameter-free RMSNorm on sublayer inputs; a float in
                         # [0,1] blends it with the identity so training can
                         # anneal the norm away (it degenerates to sign at d=1)
    )
    for k in kw:
        if k not in cfg:
            raise KeyError(f'unknown cfg key {k}')
    cfg.update(kw)
    return cfg


class Adder(nn.Module):
    def __init__(self, cfg=None, **kw):
        super().__init__()
        cfg = default_cfg(**kw) if cfg is None else default_cfg(**cfg)
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

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
