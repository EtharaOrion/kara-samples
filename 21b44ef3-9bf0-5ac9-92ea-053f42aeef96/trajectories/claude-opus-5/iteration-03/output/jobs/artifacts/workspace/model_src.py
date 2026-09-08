import torch
import torch.nn as nn
import torch.nn.functional as F


def _rms(x, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class AdderTransformer(nn.Module):
    """Single Macaron-style transformer block over per-place digit-pair tokens.

    Sequence layout (length 10, least-significant digit first):
        pos 0      : boundary / attention-sink slot (embedded as 2*emb[0])
        pos 1..8   : place i-1, embedded as emb[a_{i-1}] + emb[b_{i-1}]
        pos 9      : carry-out slot (embedded as 2*emb[0])
    Position p in 1..9 predicts sum digit p-1, so one forward pass yields all
    nine digits of the sum.

    Block order is FFN -> causal self-attention -> FFN (the leading FFN is
    optional).  Attention lets each place look back to the nearest
    non-transparent place -- the one that decides its carry-in -- so the
    attention pattern is a function of the digits, not of position alone.
    """

    def __init__(self, d_model=4, d_ff_in=5, d_ff_out=5, n_pos=10, vocab=10,
                 act="relu", d_v=0, res_bias=True, share_qk=False,
                 norm=True, q_bias=True, self_bias=True, out_scale=True,
                 res_bias_in=None, res_bias_out=None, emb_rank=0,
                 alibi=True, emb0_zero=False, emb_fixed_up=False,
                 plane_io=False, ffn_in_axis=False):
        super().__init__()
        self.d_model = d_model
        self.n_pos = n_pos
        self.vocab = vocab
        self.act = act
        self.d_v = d_v
        self.res_bias = res_bias
        self.rb_in = res_bias if res_bias_in is None else res_bias_in
        self.rb_out = res_bias if res_bias_out is None else res_bias_out
        self.d_ff_in = d_ff_in
        self.share_qk = share_qk
        self.norm = norm
        self.emb_rank = emb_rank
        self.use_alibi = alibi
        self.emb0_zero = emb0_zero
        self.emb_fixed_up = emb_fixed_up
        # With `plane_io` the two FFNs talk to the residual stream only through
        # the `emb_rank` coordinates the embedding occupies -- the "digit
        # plane".  Neither restriction costs anything, given `emb_fixed_up`:
        #   * the pre-attention FFN is the first thing in the block, so the
        #     stream it reads is the embedding alone and the other coordinates
        #     are identically zero there;
        #   * the tied readout also lives in the plane, so what the last FFN
        #     writes outside it reaches the logits only through the RMSNorm
        #     denominator -- one positive factor common to all ten classes,
        #     which cannot move the arg max.
        # The remaining coordinates are still a real channel: the pre-attention
        # FFN writes them and attention and the last FFN read them.
        self.plane_io = plane_io
        self.d_io = emb_rank if plane_io else d_model
        # With `ffn_in_axis` the pre-attention FFN has no read matrix: it takes
        # the first coordinate of the residual stream as its pre-activation.
        # Fixing `emb_up` to [I | 0] does not use up the whole O(d) gauge -- an
        # O(emb_rank) rotation *within* the digit plane still leaves [I | 0]
        # invariant -- and that leftover rotation is exactly enough to align the
        # plane's first axis with whichever in-plane direction this FFN reads.
        # The remaining scale is absorbed downstream, since relu(t*z) =
        # t*relu(z) for t > 0.  See axis.py, which performs both steps and
        # checks that no logit moves.
        self.ffn_in_axis = ffn_in_axis
        assert not ffn_in_axis or d_ff_in == 1, "ffn_in_axis needs d_ff_in == 1"
        self.q_bias = q_bias
        self.use_self_bias = self_bias
        self.out_scale = out_scale

        # With `emb0_zero` the digit-0 row is not a parameter at all: it is a
        # structural zero, so the boundary slots (2*emb[0]) are the zero vector
        # and the class-0 logit is pinned to 0.  Cheaper, but a real constraint.
        n_emb = vocab - 1 if emb0_zero else vocab
        if emb_rank:
            # Factorised tied embedding: the 10 digit vectors are constrained
            # to a rank-`emb_rank` subspace, which is cheaper than a full table
            # once vocab > d_model.  Both factors are learned.
            self.emb_lo = nn.Parameter(torch.zeros(n_emb, emb_rank))
            if not emb_fixed_up:
                self.emb_up = nn.Parameter(torch.zeros(emb_rank, d_model))
        else:
            self.emb = nn.Parameter(torch.zeros(n_emb, d_model))

        if d_ff_in:
            if not ffn_in_axis:
                self.w_in1 = nn.Parameter(torch.zeros(self.d_io, d_ff_in))
            self.b_in1 = nn.Parameter(torch.zeros(d_ff_in))
            self.w_in2 = nn.Parameter(torch.zeros(d_ff_in, d_model))
            if self.rb_in:
                self.b_in2 = nn.Parameter(torch.zeros(d_model))

        self.w_q = nn.Parameter(torch.zeros(d_model, 1))
        if q_bias:
            self.b_q = nn.Parameter(torch.zeros(1))
        if not share_qk:
            self.w_k = nn.Parameter(torch.zeros(d_model, 1))
        if alibi:
            self.alibi = nn.Parameter(torch.zeros(1))
        if self_bias:
            self.self_bias = nn.Parameter(torch.zeros(1))
        if d_v:
            self.w_v = nn.Parameter(torch.zeros(d_model, d_v))
            self.w_o = nn.Parameter(torch.zeros(d_v, d_model))

        self.w_out1 = nn.Parameter(torch.zeros(d_model, d_ff_out))
        self.b_out1 = nn.Parameter(torch.zeros(d_ff_out))
        self.w_out2 = nn.Parameter(torch.zeros(d_ff_out, self.d_io))
        if self.rb_out:
            self.b_out2 = nn.Parameter(torch.zeros(self.d_io))

        if out_scale:
            self.logit_scale = nn.Parameter(torch.ones(1))

        pos = torch.arange(n_pos)
        dist = (pos.view(-1, 1) - pos.view(1, -1)).float()
        self.register_buffer("dist", dist, persistent=False)
        self.register_buffer("eye", torch.eye(n_pos), persistent=False)
        self.register_buffer("causal", dist >= 0, persistent=False)

    def _nl(self, z):
        return F.relu(z) if self.act == "relu" else F.gelu(z)

    def _n(self, x):
        return _rms(x) if self.norm else x

    def _emb(self):
        """The 10 digit vectors, as one (vocab, d_model) table.

        With `emb_fixed_up` the up-projection is the constant injection
        [I | 0] rather than a parameter: the embedding writes into the first
        `emb_rank` coordinates of the residual stream and reads back out of
        them.  This costs nothing in expressivity.  RMSNorm is equivariant
        under orthogonal maps of the residual stream (||xQ|| = ||x||), so the
        whole block has an O(d) gauge freedom, and a rank-r factorisation has a
        GL(r) one; together they act transitively on rank-r up-projections, so
        any trained `emb_up` can be rotated to [I | 0] exactly (see gauge.py,
        which does the conversion and checks the predictions are unchanged).
        """
        if self.emb_rank:
            e = (F.pad(self.emb_lo, (0, self.d_model - self.emb_rank))
                 if self.emb_fixed_up else self.emb_lo @ self.emb_up)
        else:
            e = self.emb
        return F.pad(e, (0, 0, 1, 0)) if self.emb0_zero else e

    def embed(self, a_dig, b_dig):
        """a_dig, b_dig: (B, n_pos-2) long tensors of digits, LSB first."""
        pad = (1, 1)
        a = F.pad(a_dig, pad, value=0)
        b = F.pad(b_dig, pad, value=0)
        e = self._emb()
        return e[a] + e[b]

    def _rel(self, n):
        if n == self.n_pos:
            return self.dist, self.eye, self.causal
        dev = self.w_q.device
        pos = torch.arange(n, device=dev)
        dist = (pos.view(-1, 1) - pos.view(1, -1)).float()
        return dist, torch.eye(n, device=dev), dist >= 0

    def attend(self, h):
        dist, eye, causal = self._rel(h.shape[1])
        q = h @ self.w_q
        if self.q_bias:
            q = q + self.b_q
        k = h @ (self.w_q if self.share_qk else self.w_k)
        scores = q * k.transpose(1, 2)
        if self.use_alibi:
            scores = scores + self.alibi * dist
        if self.use_self_bias:
            scores = scores + self.self_bias * eye
        scores = scores.masked_fill(~causal, float("-inf"))
        return torch.softmax(scores, dim=-1)

    def forward(self, a_dig, b_dig, attn_override=None, return_attn=False):
        x = self.embed(a_dig, b_dig)

        if self.d_ff_in:
            h = self._n(x)
            z = (h[..., :1] if self.ffn_in_axis
                 else h[..., :self.d_io] @ self.w_in1)
            d = self._nl(z + self.b_in1) @ self.w_in2
            x = x + (d + self.b_in2 if self.rb_in else d)

        h = self._n(x)
        attn = self.attend(h)
        used = attn if attn_override is None else attn_override
        v = h @ self.w_v if self.d_v else h
        o = used @ v
        x = x + (o @ self.w_o if self.d_v else o)

        h = self._n(x)
        d = self._nl(h @ self.w_out1 + self.b_out1) @ self.w_out2
        d = d + self.b_out2 if self.rb_out else d
        x = x + F.pad(d, (0, self.d_model - self.d_io))

        logits = _rms(x) @ self._emb().t()
        if self.out_scale:
            logits = self.logit_scale * logits
        if return_attn:
            return logits, attn
        return logits
