"""Instantiate an AdderTransformer from a checkpoint's cfg dict.

Every checkpoint stores the architecture flags it was trained with; the
defaults here are the values those flags had before each one existed, so old
checkpoints keep loading as the architecture grows new options.
"""
from model_src import AdderTransformer

DEFAULTS = dict(act="relu", d_v=0, res_bias=True, share_qk=False, norm=True,
                q_bias=True, self_bias=True, out_scale=True, res_bias_in=None,
                res_bias_out=None, emb_rank=0, alibi=True, emb0_zero=False,
                emb_fixed_up=False, plane_io=False, ffn_in_axis=False)


def kwargs(cfg):
    kw = dict(d_model=cfg["d_model"], d_ff_in=cfg["d_ff_in"],
              d_ff_out=cfg["d_ff_out"])
    kw.update({k: cfg.get(k, v) for k, v in DEFAULTS.items()})
    return kw


def mk(cfg, state=None):
    m = AdderTransformer(**kwargs(cfg))
    if state is not None:
        m.load_state_dict(state)
    m.eval()
    return m


def count(cfg, **override):
    """Parameter count from the config alone, independent of any state dict."""
    from ens import single_param_count
    c = dict(cfg, **override)
    return single_param_count(
        c["d_model"], c["d_ff_in"], c["d_ff_out"],
        **{k: c.get(k, v) for k, v in DEFAULTS.items() if k not in ("act", "norm")})
