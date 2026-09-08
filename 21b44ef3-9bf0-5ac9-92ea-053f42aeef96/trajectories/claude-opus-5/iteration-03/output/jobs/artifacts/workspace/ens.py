"""E-way batched copy of AdderTransformer.

Seed variance dominates at these parameter counts, so instead of training one
model we train E independently-initialised models simultaneously on a leading
"member" axis.  Adam is elementwise, so the members stay independent; only the
learning-rate schedule and the data stream are shared.  Gradient clipping is
done per member.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _rms(x, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class EnsembleAdder(nn.Module):
    def __init__(self, E, d_model=4, d_ff_in=5, d_ff_out=5, n_pos=10, vocab=10,
                 act="relu", d_v=0, res_bias=True, share_qk=False, norm=True,
                 q_bias=True, self_bias=True, out_scale=True,
                 res_bias_in=None, res_bias_out=None, emb_rank=0, alibi=True, emb0_zero=False, emb_fixed_up=False, plane_io=False, ffn_in_axis=False, alibi_mean=-1.0, seed=0):
        super().__init__()
        self.E, self.d_model, self.n_pos, self.vocab = E, d_model, n_pos, vocab
        self.d_ff_in, self.d_ff_out, self.act = d_ff_in, d_ff_out, act
        self.d_v, self.res_bias, self.share_qk = d_v, res_bias, share_qk
        self.rb_in = res_bias if res_bias_in is None else res_bias_in
        self.rb_out = res_bias if res_bias_out is None else res_bias_out
        self.norm = norm
        self.emb_rank = emb_rank
        self.use_alibi = alibi
        self.emb0_zero = emb0_zero
        self.emb_fixed_up = emb_fixed_up
        self.plane_io = plane_io
        self.d_io = emb_rank if plane_io else d_model
        self.ffn_in_axis = ffn_in_axis
        self.q_bias, self.use_self_bias, self.out_scale = q_bias, self_bias, out_scale
        g = torch.Generator().manual_seed(seed)

        def rnd(*shape, std=1.0, mean=0.0):
            return nn.Parameter(torch.randn(*shape, generator=g) * std + mean)

        n_emb = vocab - 1 if emb0_zero else vocab
        if emb_rank:
            self.emb_lo = rnd(E, n_emb, emb_rank, std=0.9)
            if not emb_fixed_up:
                self.emb_up = rnd(E, emb_rank, d_model, std=0.9)
        else:
            self.emb = rnd(E, n_emb, d_model, std=0.8)
        if d_ff_in:
            if not ffn_in_axis:
                self.w_in1 = rnd(E, self.d_io, d_ff_in,
                                 std=1.0 / math.sqrt(self.d_io))
            self.b_in1 = rnd(E, d_ff_in, std=0.1)
            self.w_in2 = rnd(E, d_ff_in, d_model, std=1.0 / math.sqrt(d_ff_in))
            if self.rb_in:
                self.b_in2 = rnd(E, d_model, std=0.1)
        self.w_q = rnd(E, d_model, 1, std=1.0 / math.sqrt(d_model))
        if q_bias:
            self.b_q = rnd(E, 1, std=0.5)
        if not share_qk:
            self.w_k = rnd(E, d_model, 1, std=1.0 / math.sqrt(d_model))
        if alibi:
            self.alibi = rnd(E, 1, std=0.5, mean=alibi_mean)
        if self_bias:
            self.self_bias = rnd(E, 1, std=1.0)
        if d_v:
            self.w_v = rnd(E, d_model, d_v, std=1.0 / math.sqrt(d_model))
            self.w_o = rnd(E, d_v, d_model, std=1.0 / math.sqrt(d_v))
        self.w_out1 = rnd(E, d_model, d_ff_out, std=1.0 / math.sqrt(d_model))
        self.b_out1 = rnd(E, d_ff_out, std=0.1)
        self.w_out2 = rnd(E, d_ff_out, self.d_io,
                          std=1.0 / math.sqrt(d_ff_out))
        if self.rb_out:
            self.b_out2 = rnd(E, self.d_io, std=0.1)
        if out_scale:
            self.logit_scale = rnd(E, 1, std=0.2, mean=2.0)

        self.spec = [n for n, _ in self.named_parameters()]

        pos = torch.arange(n_pos)
        dist = (pos.view(-1, 1) - pos.view(1, -1)).float()
        self.register_buffer("dist", dist)
        self.register_buffer("eye", torch.eye(n_pos))
        self.register_buffer("causal", dist >= 0)

    def _nl(self, z):
        return F.relu(z) if self.act == "relu" else F.gelu(z)

    def _n(self, x):
        return _rms(x) if self.norm else x

    def _emb(self):
        if self.emb_rank and self.emb_fixed_up:
            e = F.pad(self.emb_lo, (0, self.d_model - self.emb_rank))
        elif self.emb_rank:
            e = torch.einsum("evr,erd->evd", self.emb_lo, self.emb_up)
        else:
            e = self.emb
        return F.pad(e, (0, 0, 1, 0)) if self.emb0_zero else e

    def forward(self, a_dig, b_dig, freeze_attn=None):
        a = F.pad(a_dig, (1, 1), value=0)
        b = F.pad(b_dig, (1, 1), value=0)
        # (E, V, d) indexed by (B, P) tokens -> (E, B, P, d)
        emb = self._emb()
        x = emb[:, a] + emb[:, b]

        if self.d_ff_in:
            h = self._n(x)
            pre = (h[..., :1] if self.ffn_in_axis else
                   torch.einsum("ebpd,edf->ebpf",
                                h[..., :self.d_io], self.w_in1))
            z = self._nl(pre + self.b_in1[:, None, None, :])
            x = x + torch.einsum("ebpf,efd->ebpd", z, self.w_in2)
            if self.rb_in:
                x = x + self.b_in2[:, None, None, :]

        h = self._n(x)
        q = torch.einsum("ebpd,edh->ebph", h, self.w_q)
        if self.q_bias:
            q = q + self.b_q[:, None, None, :]
        w_k = self.w_q if self.share_qk else self.w_k
        k = torch.einsum("ebpd,edh->ebph", h, w_k)
        scores = q * k.transpose(-1, -2)
        if self.use_alibi:
            scores = scores + self.alibi[:, None, :, None] * self.dist
        if self.use_self_bias:
            scores = scores + self.self_bias[:, None, :, None] * self.eye
        scores = scores.masked_fill(~self.causal, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        used = attn if freeze_attn is None else freeze_attn
        v = torch.einsum("ebpd,edv->ebpv", h, self.w_v) if self.d_v else h
        o = torch.einsum("ebpq,ebqv->ebpv", used, v)
        x = x + (torch.einsum("ebpv,evd->ebpd", o, self.w_o) if self.d_v else o)

        h = self._n(x)
        z = self._nl(torch.einsum("ebpd,edf->ebpf", h, self.w_out1)
                     + self.b_out1[:, None, None, :])
        d = torch.einsum("ebpf,efd->ebpd", z, self.w_out2)
        if self.rb_out:
            d = d + self.b_out2[:, None, None, :]
        x = x + F.pad(d, (0, self.d_model - self.d_io))

        logits = torch.einsum("ebpd,evd->ebpv", _rms(x), emb)
        if self.out_scale:
            logits = self.logit_scale[:, None, None, :] * logits
        return logits, attn

    # ---- bookkeeping -----------------------------------------------------
    def member_params(self):
        return sum(p[0].numel() for p in self.parameters())

    def extract(self, e):
        return {n: getattr(self, n)[e].detach().clone() for n in self.spec}


def single_param_count(d_model, d_ff_in, d_ff_out, vocab=10, d_v=0,
                       res_bias=True, share_qk=False, q_bias=True,
                       self_bias=True, out_scale=True, res_bias_in=None,
                       res_bias_out=None, emb_rank=0, alibi=True,
                       emb0_zero=False, emb_fixed_up=False,
                       plane_io=False, ffn_in_axis=False):
    d = d_model
    rb_in = res_bias if res_bias_in is None else res_bias_in
    rb_out = res_bias if res_bias_out is None else res_bias_out
    v = vocab - 1 if emb0_zero else vocab
    n = ((v * emb_rank + (0 if emb_fixed_up else emb_rank * d))
         if emb_rank else v * d)
    dio = emb_rank if plane_io else d
    n += (d * d_ff_out + d_ff_out + d_ff_out * dio
         + d + bool(alibi)
         + q_bias + self_bias + out_scale)
    if rb_out:
        n += dio
    if d_ff_in:
        n += ((0 if ffn_in_axis else dio * d_ff_in)
              + d_ff_in + d_ff_in * d + (d if rb_in else 0))
    if not share_qk:
        n += d
    if d_v:
        n += d * d_v + d_v * d
    return n
