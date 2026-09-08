"""Instrumented re-implementations of the block, for auditing only.

`attn_maps` and `frozen_forward` mirror `DigitPairAdder.forward` exactly (there
is an assertion for that in `check_mirror`), but expose the attention map so it
can be inspected or replaced.  Nothing here is imported by the submission.
"""
import torch

from model_src import DigitPairAdder


def _pre_attention(model, tok):
    code = model.code_table()
    h = code[tok[..., 0]] + code[tok[..., 1]]
    z = (h @ model.W1 if model.f1_in == "full" else h[..., :1]) + model.b1
    u = torch.relu(z)
    if model.f1_out == "full":
        h = h + u @ model.O1
    else:
        h = h + (u @ model.o1_vec()).unsqueeze(-1) * model.ekv
    return code, h


def attn_maps(model, tok, mode="full"):
    """The block's attention map for a batch of inputs: (..., P, P).

    ``mode`` selects which half of the attention logit survives:
      "full"      q.k + lam*distance          (what the model actually computes)
      "nocontent" lam*distance only           (the model's own fixed pattern)
      "nodist"    q.k only                    (content, no positional bias)
    """
    with torch.no_grad():
        _, h = _pre_attention(model, tok)
        k = (h @ model.Wk) if model.kv == "full" else h[..., model.KV:model.KV + 1]
        q = (h @ model.Wq + model.bq) if model.q == "proj" else model.bq
        lam = model.lam if model.lam_mode == "learn" else model.lam_c
        content = q * k.transpose(-2, -1)
        pos = lam * model.dist
        if mode == "nocontent":
            logit = pos.expand(content.shape)
        elif mode == "nodist":
            logit = content
        else:
            logit = content + pos
        return torch.softmax(logit + model.mask_bias, dim=-1)


def frozen_forward(model, tok, fixed_att):
    """Forward pass with the attention map replaced by `fixed_att`."""
    with torch.no_grad():
        code, h = _pre_attention(model, tok)
        v = (h @ model.Wv) if model.kv == "full" else h[..., model.KV:model.KV + 1]
        o = fixed_att @ v
        if model.wo == "full":
            h = h + o @ model.Wo
        elif model.wo == "axis":
            h = h + (o * model.wo_s) * model.e0
        else:
            h = h + o * model.e0
        z2 = (h @ model.W2 if model.f2_in == "full" else h[..., :1]) + model.b2
        u2 = torch.relu(z2)
        if model.f2_out == "full":
            h = h + u2 @ model.O2
        else:
            h = h + (u2 @ model.o2_vec()).unsqueeze(-1) * model.e0
        return model.temp * (2.0 * (h @ code.t()) - (code * code).sum(-1))


def ablate_acc(model, tok, tgt, mode, chunk=100000):
    """Accuracy with the attention logit restricted to `mode` (see attn_maps)."""
    good = 0
    for s in range(0, tok.shape[0], chunk):
        t = tok[s:s + chunk]
        lg = frozen_forward(model, t, attn_maps(model, t, mode))
        good += int((lg[:, 1:].argmax(-1) == tgt[s:s + chunk, 1:]).all(-1).sum())
    return good / tok.shape[0]


def content_share(model, tok):
    """How much of the attention logit spread comes from content vs position.

    Returns (sd_content, sd_position) over the allowed (i, j) entries.  A model
    whose attention is a fixed pattern in disguise has sd_content ~ 0.
    """
    with torch.no_grad():
        _, h = _pre_attention(model, tok)
        k = (h @ model.Wk) if model.kv == "full" else h[..., model.KV:model.KV + 1]
        q = (h @ model.Wq + model.bq) if model.q == "proj" else model.bq
        lam = model.lam if model.lam_mode == "learn" else model.lam_c
        m = model.mask_bias > -1.0
        c = (q * k.transpose(-2, -1)).expand(tok.shape[0], -1, -1)[:, m]
        p = (lam * model.dist).expand_as(model.dist)[m]
        return float(c.std()), float(p.std())


def check_mirror(model, tok):
    """frozen_forward with the model's own attention must equal forward()."""
    with torch.no_grad():
        ref = model(tok)
    got = frozen_forward(model, tok, attn_maps(model, tok))
    return float((ref - got).abs().max())


def batched_acc(model, tok, tgt, chunk=200000):
    good = 0
    for s in range(0, tok.shape[0], chunk):
        with torch.no_grad():
            lg = model(tok[s:s + chunk])
        good += int((lg[:, 1:].argmax(-1) == tgt[s:s + chunk, 1:]).all(-1).sum())
    return good / tok.shape[0]


def frozen_acc(model, tok, tgt, fixed_att, chunk=200000):
    good = 0
    for s in range(0, tok.shape[0], chunk):
        lg = frozen_forward(model, tok[s:s + chunk], fixed_att)
        good += int((lg[:, 1:].argmax(-1) == tgt[s:s + chunk, 1:]).all(-1).sum())
    return good / tok.shape[0]


def load_ckpt(path, P=None, dev="cpu"):
    """Rebuild a checkpoint's model, optionally in a different position count.

    Every parameter is position-independent (the only position-dependent
    tensors are the causal mask and the integer distance matrix, both
    buffers), so the same trained weights can be run over any sequence length.
    """
    from model_src import default_cfg
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(default_cfg())
    cfg.update(ck["cfg"])
    if P is not None:
        cfg["P"] = int(P)
    m = DigitPairAdder(cfg)
    with torch.no_grad():
        for k, p in m.named_parameters():
            p.copy_(ck["state"][k].reshape(p.shape).float())
    return m.to(dev).eval(), ck
