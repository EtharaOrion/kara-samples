"""Minimal transformer for exact 8-digit decimal addition.

One Macaron block -- FFN, single-head causal self-attention with a learned
relative-position bias, FFN -- over per-place digit-pair tokens.

Token layout (length 10, least-significant place first):

    pos 0 : (0, 0)                  a virtual place with digit sum 0; it neither
                                    generates nor propagates a carry, so it is
                                    the natural terminator for the carry lookahead
    pos i : (a[i-1], b[i-1])        places 0..7 of the two operands
    pos 9 : (0, 0)                  a virtual place 8, which holds the final carry

Position i predicts the output digit of the place it holds, so positions 1..9
emit all 9 digits of the sum in a single forward pass.  Attention is what
carries the carry: each position looks back for the most recent place that does
not propagate (a + b != 9) and reads off whether that place generates
(a + b >= 10).  Those keys and values are computed from the token contents, so
the attention pattern is a function of the input, not a fixed template.

The whole vocabulary is one learned table of ten numbers.  It is the input
embedding -- a token embeds as the sum of its two digits' entries -- and it is
also the set of output prototypes: a position predicts the digit whose entry
its residual stream ends up nearest to.  Training is free to put those ten
numbers anywhere; what it converges on is an evenly spaced ramp, which is what
makes summing the two embeddings mean anything at all.

All weights below were produced by gradient training (see train_ens.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rms(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


class Adder(nn.Module):
    """One Macaron transformer block: FFN -> causal self-attention -> FFN."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = c = dict(cfg)
        d, T, cd = c["d"], c["T"], c["code_dim"]

        # ---- digit code: shared input embedding / output prototypes ----
        # The answer axis has no natural unit or origin: rescaling it and
        # rescaling the code together leaves every prediction unchanged, and so
        # does shifting both.  `code_fix` spends that freedom by naming the
        # coordinates instead of paying for them -- digit 1 sits at 1 (the
        # unit) and, at code_fix 2, digit 0 sits at 0 (the origin).  The other
        # eight entries are learned and nothing constrains their shape.
        nfix = c.get("code_fix", 0)
        self.code_p = nn.Parameter(torch.randn(10 * cd - nfix) * 0.6)

        # ---- pre-attention FFN ----
        f1 = c["f1"]
        if f1:
            k1 = c["f1_in"][1] - c["f1_in"][0]
            m1 = c["f1_out"][1] - c["f1_out"][0]
            # A ReLU unit reading a single axis has one weight too many:
            # (w, b, o) and (cw, cb, o/c) are the same function for any c > 0,
            # so `*_id_in` lets the unit read the axis as it stands and leaves
            # the knee to the bias and the amplitude to the output weight.
            if not (c.get("f1_fixed_in", False) or c.get("f1_id_in", False)):
                self.f1_w = nn.Parameter(torch.randn(f1, k1) / k1 ** 0.5)
            self.f1_b = nn.Parameter(torch.zeros(f1))
            self.f1_o = nn.Parameter(torch.randn(f1, m1) * 0.5)
            if c.get("f1_ob", False):
                self.f1_ob = nn.Parameter(torch.zeros(m1))

        # ---- attention (single head, d_head = 1) ----
        # w_q, w_v and w_o are pure scale factors once the query, value and
        # output each touch a single residual axis: any factor put on them can
        # equally be put on the FFN weights that write those axes.  The
        # `*_fixed` flags pin the redundant ones to 1 instead of paying for
        # them twice.
        kq = c["qk_in"][1] - c["qk_in"][0]
        if not c.get("q_fixed", False):
            self.w_q = nn.Parameter(torch.randn(kq) / kq ** 0.5)
        if c.get("sep_qk", False):
            self.w_k = nn.Parameter(torch.randn(kq) / kq ** 0.5)
        if c.get("q_bias", False):
            self.b_q = nn.Parameter(torch.zeros(()))
        if c.get("k_bias", False):
            self.b_k = nn.Parameter(torch.zeros(()))
        # The relative-position term is a slope on (i - j).  Its size only
        # matters against the content score, whose scale the FFN that writes
        # the key axis already owns, so `alibi_fix` pins the slope to a
        # constant the way the original ALiBi does and lets the FFN set the
        # ratio instead of paying for it twice.
        if c.get("alibi", True) and c.get("alibi_fix") is None:
            self.alibi = nn.Parameter(torch.zeros(()))
        # A place's own digits say nothing about the carry coming into it, so
        # what a position needs is the most recent *earlier* place that settles
        # the carry.  `strict` masks the diagonal instead of paying a learned
        # scalar to push it down; position 0 keeps its diagonal so that its row
        # is not empty (its output is a sink and is never read).
        if c.get("self_bias", True):
            self.self_bias = nn.Parameter(torch.zeros(()))
        kv = c["v_in"][1] - c["v_in"][0]
        if not c.get("v_fixed", False):
            self.w_v = nn.Parameter(torch.randn(kv) / kv ** 0.5)
        mo = c["o_out"][1] - c["o_out"][0]
        if not c.get("o_fixed", False):
            self.w_o = nn.Parameter(torch.randn(mo) * 0.5)

        # ---- post-attention FFN ----
        f2 = c["f2"]
        if f2:
            k2 = c["f2_in"][1] - c["f2_in"][0]
            m2 = c["f2_out"][1] - c["f2_out"][0]
            if not c.get("f2_id_in", False):
                self.f2_w = nn.Parameter(torch.randn(f2, k2) / k2 ** 0.5)
            self.f2_b = nn.Parameter(torch.zeros(f2))
            self.f2_o = nn.Parameter(torch.randn(f2, m2) * 0.5)
            if c.get("f2_ob", False):
                self.f2_ob = nn.Parameter(torch.zeros(m2))

        # ---- readout ----
        od = c.get("out_dim", cd)
        if c.get("out_mode", "tied") == "free":
            self.out_w = nn.Parameter(torch.randn(10, od) * 0.6)
        kr = c["r_in"][1] - c["r_in"][0]
        if not c.get("read_id", False):
            self.r_w = nn.Parameter(torch.randn(kr, od) / kr ** 0.5)
        if c.get("r_bias", False):
            self.r_b = nn.Parameter(torch.zeros(od))
        if c.get("logit_scale", True):
            self.lsc = nn.Parameter(torch.ones(()) * c.get("lsc0", 1.0))

        # ---- constant tables (no learned values) ----
        pos = torch.arange(T)
        rel = (pos[:, None] - pos[None, :]).float()
        ok = (rel >= 0)
        if c.get("strict", False):
            ok = (rel > 0).clone()
            ok[0, 0] = True                  # keep row 0 from being empty
        self.register_buffer("rel", rel, persistent=False)
        self.register_buffer("causal", ok, persistent=False)
        self.register_buffer("eye", torch.eye(T), persistent=False)
        if c.get("f1_fixed_in", False):
            self.register_buffer("f1_sgn", torch.tensor(c["f1_signs"]).float(),
                                 persistent=False)

    # ------------------------------------------------------------------
    def _code(self):
        c = self.cfg
        cd, nfix = c["code_dim"], c.get("code_fix", 0)
        p = self.code_p
        if nfix:
            o, e = p.new_zeros(1), p.new_ones(1)
            if cd == 1:
                # digit 1 at 1 fixes the unit; digit 0 at 0 fixes the origin
                p = torch.cat([o, e, p] if nfix > 1 else [p[:1], e, p[1:]])
            else:
                canon = [e, o, o, e, o, o]
                p = torch.cat(canon[:nfix] + [p])
        return p if cd == 1 else p.view(10, cd)

    def forward(self, x):
        """x: (..., T, 2) long digit pairs -> (..., T, 10) digit logits."""
        c = self.cfg
        d, cd = c["d"], c["code_dim"]
        code = self._code()

        # embedding: token = sum of the two operand digits' codes
        counts = F.one_hot(x, 10).sum(-2).to(code.dtype)          # (..., T, 10)
        z = counts @ (code if cd > 1 else code.unsqueeze(-1))     # (..., T, cd)
        h = F.pad(z, (0, d - cd))

        # ---- pre-attention FFN ----
        if c["f1"]:
            lo, hi = c["f1_in"]
            u = h[..., lo:hi]
            if c.get("f1_id_in", False):
                p = F.relu(u + self.f1_b)
            else:
                w = self.f1_sgn.unsqueeze(-1) if c.get("f1_fixed_in", False) \
                    else self.f1_w
                p = F.relu(u @ w.T + self.f1_b)
            o = p @ self.f1_o
            if c.get("f1_ob", False):
                o = o + self.f1_ob
            lo, hi = c["f1_out"]
            h = h + F.pad(o, (lo, d - hi))

        # ---- causal self-attention, one head ----
        a = _rms(h) if "a" in c.get("norm", "") else h
        lo, hi = c["qk_in"]
        w_q = a.new_ones(hi - lo) if c.get("q_fixed", False) else self.w_q
        q = a[..., lo:hi] @ w_q
        if c.get("q_bias", False):
            q = q + self.b_q
        if c.get("sep_qk", False):
            k = a[..., lo:hi] @ self.w_k
        else:
            k = q if not c.get("k_bias", False) else q
        if c.get("k_bias", False):
            k = k + self.b_k
        att = q.unsqueeze(-1) * k.unsqueeze(-2)                   # (..., T, T)
        if c.get("alibi", True):
            sl = self.alibi if c.get("alibi_fix") is None else c["alibi_fix"]
            att = att + sl * self.rel
        if c.get("self_bias", True):
            att = att + self.self_bias * self.eye
        att = att.masked_fill(~self.causal, float("-inf")).softmax(-1)
        lo, hi = c["v_in"]
        w_v = a.new_ones(hi - lo) if c.get("v_fixed", False) else self.w_v
        v = a[..., lo:hi] @ w_v
        o = att @ v.unsqueeze(-1)                                 # (..., T, 1)
        w_o = self.w_o if not c.get("o_fixed", False) else o.new_ones(
            c["o_out"][1] - c["o_out"][0])
        lo, hi = c["o_out"]
        h = h + F.pad(o * w_o, (lo, d - hi))

        # ---- post-attention FFN ----
        if c["f2"]:
            b = _rms(h) if "f" in c.get("norm", "") else h
            lo, hi = c["f2_in"]
            u = b[..., lo:hi]
            p = F.relu(u + self.f2_b) if c.get("f2_id_in", False) \
                else F.relu(u @ self.f2_w.T + self.f2_b)
            o = p @ self.f2_o
            if c.get("f2_ob", False):
                o = o + self.f2_ob
            lo, hi = c["f2_out"]
            h = h + F.pad(o, (lo, d - hi))

        # ---- readout: nearest learned digit prototype ----
        lo, hi = c["r_in"]
        r = h[..., lo:hi]
        qv = r if c.get("read_id", False) else r @ self.r_w       # (..., T, cd)
        if c.get("r_bias", False):
            qv = qv + self.r_b
        mode = c.get("out_mode", "tied")
        if mode == "proto":
            lg = -(qv - code).pow(2)
        elif mode == "free":
            lg = qv @ self.out_w.T
        else:
            lg = qv @ code.T
        return lg * self.lsc if c.get("logit_scale", True) else lg

_CFG = {'d': 2, 'T': 10, 'code_dim': 1, 'code_fix': 1, 'f1': 3, 'f1_in': [0, 1], 'f1_out': [1, 2], 'f1_ob': False, 'f1_fixed_in': False, 'qk_in': [1, 2], 'sep_qk': False, 'q_bias': True, 'k_bias': False, 'alibi': True, 'self_bias': False, 'v_in': [1, 2], 'o_out': [0, 1], 'o_fixed': False, 'f2': 2, 'f2_in': [0, 1], 'f2_out': [0, 1], 'f2_ob': False, 'r_in': [0, 1], 'read_id': True, 'r_bias': False, 'norm': '', 'logit_scale': False, 'lsc0': 1.0, 'out_mode': 'proto', 'out_dim': 1, 'q_fixed': True, 'v_fixed': True, 'strict': True, 'alibi_fix': -4.0, 'f1_id_in': True, 'f2_id_in': True}

_W = {
    'code_p': [1.1267646551132202, 0.8741064071655273, 0.7482900023460388, 0.6224234700202942, 0.4965442419052124, 0.3707185983657837, 0.24484629929065704, 0.11883068829774857, -0.008700908161699772],
    'f1_b': [-0.9955086708068848, -1.244898796081543, -1.1001861095428467],
    'f1_o': [-18.56531524658203, -14.03720474243164, 32.604373931884766],
    'b_q': [22.152545928955078],
    'w_o': [1.5111526250839233],
    'f2_b': [-1.248042106628418, -1.1373040676116943],
    'f2_o': [11.411001205444336, -11.410008430480957],
}


def build_model():
    """Returns (model, metadata)."""
    model = Adder(_CFG)
    sd = model.state_dict()
    for k, v in _W.items():
        t = torch.tensor(v, dtype=torch.float32).reshape(sd[k].shape)
        sd[k] = t
    model.load_state_dict(sd)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    meta = {
        "n_params": sum(p.numel() for p in model.parameters()),
        "d_model": _CFG["d"],
        "n_layers": 1,
        "n_heads": 1,
        "d_head": 1,
        "seq_len": _CFG["T"],
        "vocab": 10,
        "task": "8-digit decimal addition",
        "description": "single Macaron transformer block; positions 1..9 emit "
                       "the 9 sum digits in one forward pass",
    }
    return model, meta


def _digits(n):
    return [(n // 10 ** i) % 10 for i in range(8)]


@torch.no_grad()
def add(model, a, b):
    """Exact sum of two 8-digit integers, read off one forward pass."""
    da, db = _digits(int(a)), _digits(int(b))
    seq = [[0, 0]] + [[da[i], db[i]] for i in range(8)] + [[0, 0]]
    x = torch.tensor([seq], dtype=torch.long)
    logits = model(x)[0, 1:, :]
    out = 0
    for i, d in enumerate(logits.argmax(-1).tolist()):
        out += d * 10 ** i
    return out


@torch.no_grad()
def add_batch(model, pairs):
    """Vectorised form of `add` for a list of (a, b) pairs."""
    seq = []
    for a, b in pairs:
        da, db = _digits(int(a)), _digits(int(b))
        seq.append([[0, 0]] + [[da[i], db[i]] for i in range(8)] + [[0, 0]])
    x = torch.tensor(seq, dtype=torch.long)
    dig = model(x)[:, 1:, :].argmax(-1)
    pw = torch.tensor([10 ** i for i in range(9)], dtype=torch.long)
    return (dig * pw).sum(-1).tolist()
