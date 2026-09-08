"""Train the tiny addition transformer.

Usage:
    python train.py --d-model 6 --layers "0,0,6;1,3,14" --steps 40000 --out ckpt.pt

``--layers`` is a semicolon separated list of "n_heads,d_head,d_ff" per block;
n_heads/d_head of 0 means the block has no attention, d_ff of 0 means no MLP.
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

import data as D
import model_src
from model_src import NDIG, SEQ, AdderTransformer


def parse_layers(spec):
    layers = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        nh, dh, dff = (int(x) for x in part.split(","))
        layers.append(dict(n_heads=nh, d_head=dh, d_ff=dff))
    return layers


class SamplePool:
    """Amortises operand generation.

    The model is so small that a training step is dominated by CUDA launch
    overhead, and the sampler (rejection-resampling the held-out bucket, three
    times per step for the three mixture components) issues more kernels than
    the forward and backward passes combined.  Generating a large pool and
    serving random slices from it collapses that to two index operations.
    The pool is refilled every ``refresh`` steps and only a small fraction of it
    is consumed in between, so the data stays effectively fresh -- and it is
    still drawn by the same holdout-excluding sampler, so no held-out pair can
    leak in.
    """

    def __init__(self, size, maker, device, gen, refresh):
        self.size, self.maker, self.device = size, maker, device
        self.gen, self.refresh = gen, refresh
        self.a = self.b = None
        self.next_fill = 0

    def _fill(self):
        a, b = self.maker(self.size)
        self.a, self.b = a.to(torch.uint8), b.to(torch.uint8)

    def draw(self, n, step):
        if self.a is None or step >= self.next_fill:
            self._fill()
            self.next_fill = step + self.refresh
        idx = torch.randint(0, self.size, (n,), device=self.device, generator=self.gen)
        return self.a[idx].long(), self.b[idx].long()


def to_tokens(a, b):
    """(B, 8) digits -> (B, 10) token ids: zero pad, digits, zero pad."""
    n = a.shape[0]
    pad = torch.zeros((n, 1), dtype=torch.long, device=a.device)
    ta = torch.cat([pad, a.long(), pad], dim=1)
    tb = torch.cat([pad, b.long(), pad], dim=1)
    return ta, tb


@torch.no_grad()
def evaluate(model, a, b, batch=1 << 16):
    """Exact-match accuracy: every one of the 9 output digits must be right."""
    model.eval()
    total = correct = 0
    worst_digit = torch.zeros(NDIG + 1, device=a.device)
    for i in range(0, a.shape[0], batch):
        aa, bb = a[i:i + batch], b[i:i + batch]
        tgt = D.targets_from_values(D.digits_to_value(aa), D.digits_to_value(bb))
        ta, tb = to_tokens(aa, bb)
        pred = model(ta, tb)[:, 1:, :].argmax(-1)
        ok = (pred == tgt)
        worst_digit += (~ok).float().sum(0)
        correct += int(ok.all(-1).sum())
        total += aa.shape[0]
    model.train()
    return correct / total, (worst_digit / total).tolist()


def attn_entropy(model):
    """Mean entropy (nats) of the attention rows of the last forward pass.
    ~0 means a hard content look-up; ln(9)=2.2 would be uniform over the causal
    mask.  This is the sharpest single indicator of whether the carry circuit
    was actually found or merely approximated by a fixed geometric blur."""
    tot, n = 0.0, 0
    for blk in model.blocks:
        if blk.attn is not None and blk.attn.attn is not None:
            p = blk.attn.attn[:, :, 1:, :].detach()
            tot += float((-(p * p.clamp_min(1e-9).log()).sum(-1)).mean())
            n += 1
    return tot / max(n, 1)


def save_ckpt(cfg, state, nparams, acc, hacc):
    torch.save({
        "state_dict": state,
        "d_model": cfg.d_model,
        "layers": parse_layers(cfg.layers),
        "pos_mode": cfg.pos_mode,
        "tie_head": not cfg.untied,
        "head_bias": not cfg.no_head_bias,
        "q_bias": cfg.q_bias,
        "learn_scale": cfg.learn_scale,
        "n_params": nparams,
        "holdout_acc": acc,
        "hard_acc": hacc,
        "cfg": vars(cfg),
    }, cfg.out)


def train(cfg):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    model_src.STORE_ATTN = True   # needed for the entropy penalty and the crispness readout

    model = AdderTransformer(cfg.d_model, parse_layers(cfg.layers),
                             pos_mode=cfg.pos_mode, tie_head=not cfg.untied,
                             head_bias=not cfg.no_head_bias, q_bias=cfg.q_bias,
                             learn_scale=cfg.learn_scale).to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"[cfg] d_model={cfg.d_model} layers={cfg.layers} pos={cfg.pos_mode} tied={not cfg.untied} params={nparams}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98),
                            weight_decay=cfg.wd, eps=1e-9)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.steps, pct_start=0.05,
        anneal_strategy="cos", div_factor=10.0, final_div_factor=200.0)

    ho_a, ho_b = D.make_holdout(cfg.eval_n, dev, prop_p=0.0, seed=1234)
    hard_a, hard_b = D.holdout_chain(cfg.eval_n // 4, dev, seed=4321)

    gen = torch.Generator(device=dev)
    gen.manual_seed(cfg.seed + 777)

    n_chain = int(cfg.batch * cfg.chain_frac)
    n_prop = int(cfg.batch * cfg.prop_frac)
    n_uni = cfg.batch - n_chain - n_prop
    pools = [
        (n_uni, SamplePool(cfg.pool, lambda n: D.make_batch(n, dev, prop_p=0.0, gen=gen),
                           dev, gen, cfg.pool_refresh)),
        (n_prop, SamplePool(cfg.pool, lambda n: D.make_batch(n, dev, prop_p=cfg.prop_p, gen=gen),
                            dev, gen, cfg.pool_refresh)),
        (n_chain, SamplePool(cfg.pool, lambda n: D.make_chain(n, dev, gen=gen),
                             dev, gen, cfg.pool_refresh)),
    ]

    best = 0.0
    best_state = None
    t0 = time.time()
    for step in range(1, cfg.steps + 1):
        # Most of the batch matches the graded (uniform) distribution; the rest
        # is enriched with carry chains so the model must learn real propagation
        # rather than an exponentially-weighted blur that only spans two places.
        parts = [p.draw(k, step) for k, p in pools if k > 0]
        a = torch.cat([x for x, _ in parts])
        b = torch.cat([y for _, y in parts])
        tgt = D.targets_from_values(D.digits_to_value(a), D.digits_to_value(b))
        ta, tb = to_tokens(a, b)

        logits = model(ta, tb)[:, 1:, :]
        loss = F.cross_entropy(logits.reshape(-1, 10), tgt.reshape(-1))
        if cfg.attn_entropy > 0.0 and step >= cfg.entropy_warmup * cfg.steps:
            # Nudge the carry head towards a hard look-up rather than a blur.
            # Warmed up, because a sharpening pressure applied from step 0 just
            # freezes whatever arbitrary pattern the model starts with.
            # Pure training-time regularisation; it touches no inference code.
            ent = 0.0
            heads = [b.attn for b in model.blocks if b.attn is not None]
            if cfg.entropy_last_only:
                heads = heads[-1:]     # only the carry head; layer 0 is a shift
            for att in heads:
                p = att.attn[:, :, 1:, :]
                ent = ent + (-(p * torch.log(p.clamp_min(1e-9)))).sum(-1).mean()
            loss = loss + cfg.attn_entropy * ent
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % cfg.eval_every == 0 or step == cfg.steps:
            acc, per_digit = evaluate(model, ho_a, ho_b)
            hacc, _ = evaluate(model, hard_a, hard_b)
            score = min(acc, hacc)
            if score >= best:
                best = score
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                if cfg.out:
                    save_ckpt(cfg, best_state, nparams, acc, hacc)
            print(f"[{step:6d}] loss={loss.item():.5f} holdout={acc*100:.3f}% "
                  f"chain={hacc*100:.3f}% ent={attn_entropy(model):.3f} "
                  f"lr={sched.get_last_lr()[0]:.2e} ({time.time()-t0:.0f}s)", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    acc, per_digit = evaluate(model, ho_a, ho_b)
    hacc, hper = evaluate(model, hard_a, hard_b)
    print(f"[final] params={nparams} holdout={acc*100:.4f}% chain={hacc*100:.4f}% "
          f"ent={attn_entropy(model):.4f}", flush=True)
    print(f"[final] per-digit error rate (uniform): "
          f"{['%.5f' % x for x in per_digit]}", flush=True)

    if cfg.out:
        save_ckpt(cfg, model.state_dict(), nparams, acc, hacc)
        print(f"[saved] {cfg.out}", flush=True)
    return acc, nparams


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d-model", type=int, default=8)
    p.add_argument("--layers", type=str, default="1,4,16;1,4,16")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--prop-p", type=float, default=0.4)
    p.add_argument("--prop-frac", type=float, default=0.35)
    p.add_argument("--chain-frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-n", type=int, default=100000)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--pos-mode", type=str, default="rank1")
    p.add_argument("--untied", action="store_true")
    p.add_argument("--attn-entropy", type=float, default=0.0)
    p.add_argument("--q-bias", action="store_true")
    p.add_argument("--learn-scale", action="store_true")
    p.add_argument("--no-head-bias", action="store_true")
    p.add_argument("--entropy-warmup", type=float, default=0.25)
    p.add_argument("--entropy-last-only", action="store_true")
    p.add_argument("--pool", type=int, default=1 << 21)
    p.add_argument("--pool-refresh", type=int, default=400)
    p.add_argument("--out", type=str, default="")
    train(p.parse_args())


if __name__ == "__main__":
    main()
