"""Source of truth for the shipped model class.

`build.py` copies the text between the BEGIN/END markers verbatim into
/workspace/submission.py, so this file and the graded file always define exactly the
same module. No training code lives here.
"""

# ---- BEGIN SHIPPED SOURCE ----
import torch
import torch.nn as nn


class DigitPairAdder(nn.Module):
    """A one-block transformer that adds two base-10 numbers, digit-pair by digit-pair.

    Tokens are digit pairs, least-significant first, with a ``(0, 0)`` pad at each end::

        pos 0        pos 1 .. pos n         pos n+1
        (0, 0)       (a_k, b_k)             (0, 0)

    Everything the block computes lives on a single scalar residual channel.  A token's
    value is ``x = code[a] + code[b]`` for one learned 10-entry code table, which is also
    the set of read-out prototypes.

    A 2-unit clamp bank turns ``x`` into one key/value stream::

        u_j  = clamp(bank_w * (x - knee_j), 0, 1)      j = 0, 1
        key  = key_w * (u_1 - u_0)
        val  = u_1

    Two heads read that stream through different causal masks, with attention logits
    ``key_j + lam * (i - j)``.  The strictly-causal head (``j < i``) supplies position
    ``i`` with its carry-in; the inclusively-causal head (``j <= i``) reports position
    ``i``'s carry-out, which gates the mod-10 fold::

        y = x + carry_w * attn_strict + fold * attn_incl
        logits[d] = -(y - code[d]) ** 2

    Entries 0 and 1 of the code pin the origin and the unit of the residual axis; the
    other eight, the weight a carry is worth and the size of the mod-10 fold are learned.

    Position ``p`` predicts answer digit ``p - 1``, so the whole sum is produced by a
    single forward pass.  Nothing here is specific to eight digits: the parameters do not
    depend on position or on sequence length.
    """

    # Fixed architectural constants (chosen a priori, not fitted).
    BANK_W = 8.0    # gate sharpness: transition width is 1/8 of a code step
    KEY_W = 400.0   # depth of the notch that makes a carry-transparent place unattendable
    LAM = -12.0     # per-step recency bias, so a head lands on the *nearest* legal key

    def __init__(self):
        super().__init__()
        # --- the 12 learned values ---
        self.code_free = nn.Parameter(torch.zeros(8))   # code[2] .. code[9]
        self.carry_w = nn.Parameter(torch.zeros(1))     # what a carry is worth
        self.knee = nn.Parameter(torch.zeros(2))        # the two bank thresholds
        self.fold = nn.Parameter(torch.zeros(1))        # mod-10 write-back
        # --- constants, held as buffers so every number the model uses is inspectable ---
        self.register_buffer("code01", torch.tensor([0.0, 1.0]))  # origin and unit
        self.register_buffer("bank_w", torch.tensor(self.BANK_W))
        self.register_buffer("key_w", torch.tensor(self.KEY_W))
        self.register_buffer("lam", torch.tensor(self.LAM))

    def codes(self):
        """The 10 digit codes.  Entries 0 and 1 fix the origin and the unit of the
        residual axis (the one scale freedom the block has); 2..9 are learned."""
        return torch.cat([self.code01, self.code_free])

    def forward(self, tok):
        """tok: integer tensor (..., P, 2) of digit pairs.  Returns (..., P, 10) logits."""
        code = self.codes()
        x = code[tok[..., 0]] + code[tok[..., 1]]                       # (..., P)

        u = torch.clamp(self.bank_w * (x.unsqueeze(-1) - self.knee), 0.0, 1.0)
        key = self.key_w * (u[..., 1] - u[..., 0])                      # (..., P)
        val = u[..., 1]                                                 # (..., P)

        p = x.shape[-1]
        idx = torch.arange(p, device=x.device)
        rel = self.lam * (idx[:, None] - idx[None, :]).to(x.dtype)      # (P, P)
        logit = key.unsqueeze(-2) + rel                                 # (..., i, j)

        after = idx[:, None] > idx[None, :]
        at_or_after = idx[:, None] >= idx[None, :]
        # Row 0 has no strictly-earlier token; let it look at the leading pad (val 0) so the
        # softmax is well defined.  That row's prediction is never read.
        strict = after | ((idx[:, None] == 0) & (idx[None, :] == 0))

        neg = torch.tensor(-1e30, dtype=x.dtype, device=x.device)
        a_strict = torch.softmax(torch.where(strict, logit, neg), dim=-1)
        a_incl = torch.softmax(torch.where(at_or_after, logit, neg), dim=-1)

        carry_in = (a_strict * val.unsqueeze(-2)).sum(-1)                # (..., P)
        carry_out = (a_incl * val.unsqueeze(-2)).sum(-1)                 # (..., P)

        y = x + self.carry_w * carry_in + self.fold * carry_out
        return -(y.unsqueeze(-1) - code) ** 2                            # (..., P, 10)


def _digits(v, n):
    return [(v // (10 ** k)) % 10 for k in range(n)]


def tokenize(a, b, n):
    """Digit-pair tokens for a + b at width n, with a (0, 0) pad at each end."""
    da, db = _digits(a, n), _digits(b, n)
    return [[0, 0]] + [[da[k], db[k]] for k in range(n)] + [[0, 0]]


def build_model():
    model = DigitPairAdder()
    state = {k: torch.tensor(v, dtype=torch.float32) for k, v in _WEIGHTS.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, unexpected
    assert set(missing) <= {"code01", "bank_w", "key_w", "lam"}, missing
    model.eval()
    n_param = sum(p.numel() for p in model.parameters())
    meta = {
        "name": "DigitPairAdder",
        "architecture": "1-block transformer, 2 heads over 1 content key/value stream",
        "parameters": n_param,
        "parameter_groups": {"code_free": 8, "carry_w": 1, "knee": 2, "fold": 1},
        "buffers_are_constants": ["code01", "bank_w", "key_w", "lam"],
        "residual_channels": 1,
        "vocab": "digit pairs (a, b), least-significant first",
        "trained_on": "synthetic digit-pair addition, widths 1-8",
        "notes": "position p predicts answer digit p-1; whole sum from one forward pass",
    }
    return model, meta


@torch.no_grad()
def add(model, a, b):
    """Return a + b, read off one forward pass of `model`."""
    a, b = int(a), int(b)
    n = max(len(str(abs(a))), len(str(abs(b))))
    device = next(model.parameters()).device
    tok = torch.tensor([tokenize(a, b, n)], dtype=torch.long, device=device)
    digits = model(tok)[0].argmax(-1).tolist()          # position p -> answer digit p-1
    return sum(int(d) * (10 ** k) for k, d in enumerate(digits[1:]))
# ---- END SHIPPED SOURCE ----


# Placeholder so this module is importable on its own; submission.py carries real numbers.
_WEIGHTS = {"code_free": [0.0] * 8, "carry_w": [0.0], "knee": [0.0, 0.0], "fold": [0.0]}
