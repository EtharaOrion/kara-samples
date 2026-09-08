"""Template for the graded file.  build.py substitutes @WEIGHTS@ and @DOC@.

The forward pass here is line-for-line the same computation as arch.forward in the
workspace (which is what training optimised), with the ensemble axis dropped.
"""
HEADER = '''"""Minimal transformer that adds two 8-digit integers.

@DOC@

One encoder block over per-place digit-pair tokens, LSB first:

    pos 0      (0,0) pad      - the anchor a carry chain terminates on
    pos 1..n   (a_i, b_i)     - operand place i-1
    pos n+1    (0,0) pad      - slot for the final carry digit

A place is embedded as code[a_i] + code[b_i] into a scalar residual stream.  A
two-unit clamp bank turns that scalar into an attention key and value; two masked
heads share the key/value stream.  Position p predicts answer digit p-1, so the
whole answer comes out of a single forward pass.
"""
import torch
import torch.nn as nn

# --------------------------------------------------------------------------- weights
# _PARAMS: learned by gradient descent (train.py in the workspace).  These are the
#          registered nn.Parameters and are the whole parameter count.
# _CONST : fixed architectural constants -- saturation slope, key/value read-off
#          weights, attention temperature and recency slope, the code origin.  They
#          carry no fitted information; certify.py proves the model is exact for the
#          whole 8-digit domain with these values held fixed.
@WEIGHTS@


class DigitPairAdder(nn.Module):
    def __init__(self):
        super().__init__()
        for k, v in _PARAMS.items():
            self.register_parameter(k, nn.Parameter(torch.tensor(v, dtype=torch.float32)))
        for k, v in _CONST.items():
            self.register_buffer(k, torch.tensor(v, dtype=torch.float32))

    def forward(self, da, db):
        """da, db: (B,P) int64 digit tokens, LSB first.  -> (B,P,10) digit logits."""
        code = torch.cat([self.code0, self.code], 0)                 # (10,C)
        x = code[da] + code[db]                                      # (B,P,C)
        u = (x @ self.Bw + self.bb).clamp(0.0, 1.0)                  # (B,P,U)
        k = u @ self.kw                                              # (B,P)  attention key
        v = u @ self.vw + self.vb                                    # (B,P)  attention value

        P = da.shape[1]
        i = torch.arange(P, device=da.device)
        rel = (i[:, None] - i[None, :]).to(x.dtype)                  # query - key distance
        neg = torch.finfo(x.dtype).min / 4
        z = torch.zeros((), dtype=x.dtype, device=da.device)
        nn_ = torch.full((), neg, dtype=x.dtype, device=da.device)
        m_s = torch.where(i[None, :] < i[:, None], z, nn_).clone()   # strictly causal
        m_s[0, 0] = 0.0                                              # pos 0 has no predecessor
        m_i = torch.where(i[None, :] <= i[:, None], z, nn_)          # inclusively causal

        base = self.q * k[:, None, :] + self.lam * rel               # (B,P,P)
        a_s = torch.softmax(base + m_s, dim=-1)
        a_i = torch.softmax(base + m_i, dim=-1)
        c_in = torch.einsum("bij,bj->bi", a_s, v)
        c_out = torch.einsum("bij,bj->bi", a_i, v)

        out = (x + c_in[..., None] * self.w1
                 + c_out[..., None] * self.w2 + self.rb)             # (B,P,C)
        return self.ls * (out @ code.t() - 0.5 * (code * code).sum(-1))


def build_model():
    model = DigitPairAdder()
    model.eval()
    n_param = sum(p.numel() for p in model.parameters())
    meta = {
        "name": "DigitPairAdder",
        "task": "exact addition of two 8-digit integers",
        "n_parameters": n_param,
        "architecture": "1 block: digit-pair embedding -> clamp bank -> 2-head "
                        "shared-KV self-attention -> tied nearest-prototype readout",
        "n_layers": 1,
        "n_heads": 2,
        "d_model": int(model.code.shape[1]),
        "decoding": "single forward pass, argmax per position",
    }
    return model, meta


@torch.no_grad()
def add(model, a, b):
    """Exact sum of two non-negative integers, computed by one forward pass."""
    a, b = int(a), int(b)
    n = max(len(str(a)), len(str(b)), 1)
    da = [(a // 10 ** i) % 10 for i in range(n)]
    db = [(b // 10 ** i) % 10 for i in range(n)]
    dev = model.code.device
    ta = torch.tensor([[0] + da + [0]], dtype=torch.long, device=dev)
    tb = torch.tensor([[0] + db + [0]], dtype=torch.long, device=dev)
    pred = model(ta, tb).argmax(-1)[0, 1:]                           # n+1 answer digits
    out = 0
    for i in range(pred.shape[0] - 1, -1, -1):
        out = out * 10 + int(pred[i])
    return out
'''
