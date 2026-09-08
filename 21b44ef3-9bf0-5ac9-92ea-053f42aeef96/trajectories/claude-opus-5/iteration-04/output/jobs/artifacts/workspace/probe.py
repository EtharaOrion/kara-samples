"""An instrumented copy of the block's forward pass, for tools that need to see
inside it: the audit (which intervenes on the attention) and the warm start
(which needs to know what each FFN actually sees).

`Adder.forward` in model_src.py is the shipped path and stays as small as it
can be, so the instrumentation lives here instead of being threaded through it.
The two must agree exactly; verify.py checks that on every run.
"""
import torch
import torch.nn.functional as F

from model_src import _rms


def run(m, x, attn_mode="normal", frozen=None):
    """Returns (logits, attention before any intervention, intermediates)."""
    c = m.cfg
    d, cd = c["d"], c["code_dim"]
    code = m._code()
    mid = {}
    counts = F.one_hot(x, 10).sum(-2).to(code.dtype)
    z = counts @ (code if cd > 1 else code.unsqueeze(-1))
    h = F.pad(z, (0, d - cd))
    if c["f1"]:
        lo, hi = c["f1_in"]
        mid["f1_in"] = h[..., lo:hi]
        if c.get("f1_id_in", False):
            p = F.relu(h[..., lo:hi] + m.f1_b)
        else:
            w = m.f1_sgn.unsqueeze(-1) if c.get("f1_fixed_in", False) else m.f1_w
            p = F.relu(h[..., lo:hi] @ w.T + m.f1_b)
        o = p @ m.f1_o
        if c.get("f1_ob", False):
            o = o + m.f1_ob
        lo, hi = c["f1_out"]
        h = h + F.pad(o, (lo, d - hi))
    a = _rms(h) if "a" in c.get("norm", "") else h
    lo, hi = c["qk_in"]
    w_q = a.new_ones(hi - lo) if c.get("q_fixed", False) else m.w_q
    q = a[..., lo:hi] @ w_q
    if c.get("q_bias", False):
        q = q + m.b_q
    k = a[..., lo:hi] @ m.w_k if c.get("sep_qk", False) else q
    if c.get("k_bias", False):
        k = k + m.b_k
    att = q.unsqueeze(-1) * k.unsqueeze(-2)
    if c.get("alibi", True):
        att = att + (m.alibi if c.get("alibi_fix") is None
                     else c["alibi_fix"]) * m.rel
    if c.get("self_bias", True):
        att = att + m.self_bias * m.eye
    att = att.masked_fill(~m.causal, float("-inf")).softmax(-1)
    raw = att
    if attn_mode == "frozen":
        att = frozen.expand_as(att)
    elif attn_mode == "uniform":
        att = m.causal.float() / m.causal.float().sum(-1, keepdim=True)
    lo, hi = c["v_in"]
    w_v = a.new_ones(hi - lo) if c.get("v_fixed", False) else m.w_v
    v = a[..., lo:hi] @ w_v
    o = att @ v.unsqueeze(-1)
    w_o = m.w_o if not c.get("o_fixed", False) else o.new_ones(
        c["o_out"][1] - c["o_out"][0])
    lo, hi = c["o_out"]
    h = h + F.pad(o * w_o, (lo, d - hi))
    if c["f2"]:
        b = _rms(h) if "f" in c.get("norm", "") else h
        lo, hi = c["f2_in"]
        mid["f2_in"] = b[..., lo:hi]
        p = F.relu(b[..., lo:hi] + m.f2_b) if c.get("f2_id_in", False) \
            else F.relu(b[..., lo:hi] @ m.f2_w.T + m.f2_b)
        o = p @ m.f2_o
        if c.get("f2_ob", False):
            o = o + m.f2_ob
        lo, hi = c["f2_out"]
        h = h + F.pad(o, (lo, d - hi))
    lo, hi = c["r_in"]
    r = h[..., lo:hi]
    qv = r if c.get("read_id", False) else r @ m.r_w
    if c.get("r_bias", False):
        qv = qv + m.r_b
    mode = c.get("out_mode", "tied")
    if mode == "proto":
        lg = -(qv - code).pow(2)
    elif mode == "free":
        lg = qv @ m.out_w.T
    else:
        lg = qv @ code.T
    if c.get("logit_scale", True):
        lg = lg * m.lsc
    return lg, raw, mid
