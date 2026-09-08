"""Audit /workspace/submission.py: accuracy, parameter count, and whether the
self-attention is genuinely doing input-dependent work.

Run:  python verify.py [--n 200000]
"""

import argparse
import importlib.util
import sys

import torch

import data as D


def load_submission(path="/workspace/submission.py"):
    spec = importlib.util.spec_from_file_location("submission_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def to_tokens(a, b):
    pad = torch.zeros((a.shape[0], 1), dtype=torch.long, device=a.device)
    return (torch.cat([pad, a.long(), pad], 1), torch.cat([pad, b.long(), pad], 1))


@torch.no_grad()
def exact_match(model, a, b, chunk=1 << 15):
    ok = 0
    for i in range(0, a.shape[0], chunk):
        aa, bb = a[i:i + chunk], b[i:i + chunk]
        tgt = D.targets_from_values(D.digits_to_value(aa), D.digits_to_value(bb))
        pred = model(*to_tokens(aa, bb))[:, 1:, :].argmax(-1)
        ok += int((pred == tgt).all(-1).sum())
    return ok / a.shape[0]


def chain_cases(n, device, seed=31337):
    """Operand pairs built so a carry must propagate across a maximal run of
    carry-transparent places: place g generates, every place above g sums to 9.
    These are the cases uniform sampling essentially never produces."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    a = torch.randint(0, 10, (n, 8), device=device, generator=g)
    b = torch.randint(0, 10, (n, 8), device=device, generator=g)
    start = torch.randint(0, 8, (n,), device=device, generator=g)
    place = torch.arange(8, device=device).unsqueeze(0)
    above = place > start.unsqueeze(1)
    at = place == start.unsqueeze(1)
    # the generating place needs a_g >= 1 and b_g >= 10 - a_g
    a = torch.where(at, a.clamp(min=1), a)
    lo = 10 - a
    span = (10 - lo).clamp(min=1)
    rnd = torch.randint(0, 10, (n, 8), device=device, generator=g)
    b = torch.where(at, lo + rnd % span, b)
    # transparent places above it: every digit pair summing to 9, 9+0 included
    b = torch.where(above, 9 - a, b)
    # keep both operands a full 8 digits wide
    a[:, 7] = a[:, 7].clamp(min=1)
    b[:, 7] = torch.where(above[:, 7], 9 - a[:, 7], b[:, 7].clamp(min=1))
    # only the top place is constrained, so both operands stay 8 digits wide
    a = torch.where(above & (place == 7), a.clamp(min=1, max=8), a)
    b = torch.where(above & (place == 7), 9 - a, b)
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--path", default="/workspace/submission.py")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sub = load_submission(args.path)
    model, meta = sub.build_model()
    model = model.to(dev)

    n_params = sum(p.numel() for p in model.parameters())
    print("=" * 72)
    print("parameters (counted independently):", n_params)
    for k, v in meta.items():
        print("  meta.%-24s %s" % (k, v))
    named = [(n, tuple(p.shape), p.numel()) for n, p in model.named_parameters()]
    print("-" * 72)
    for n, s, c in named:
        print("  %-28s %-14s %5d" % (n, s, c))
    print("  %-28s %-14s %5d" % ("TOTAL", "", sum(c for _, _, c in named)))
    buffers = [(n, tuple(b.shape), b.dtype) for n, b in model.named_buffers()]
    print("  non-parameter buffers:", buffers or "none")
    print("  floating-point buffers:",
          [n for n, b in model.named_buffers() if b.is_floating_point()] or "none")

    # ---------------- accuracy on never-trained-on pairs ----------------
    ho_a, ho_b = D.make_holdout(args.n, dev, prop_p=0.0, seed=99991)
    acc = exact_match(model, ho_a, ho_b)
    hard_a, hard_b = D.make_holdout(args.n // 4, dev, prop_p=0.45, seed=77771)
    hacc = exact_match(model, hard_a, hard_b)
    # Same measurement without the hash restriction, to rule out any artefact
    # of the held-out bucket being distributionally special.
    gu = torch.Generator(device=dev)
    gu.manual_seed(555)
    ua, ub = D.sample_digits(args.n, 0.0, dev, gu)
    uacc = exact_match(model, ua, ub)

    ch_a, ch_b = chain_cases(100000, dev)
    cacc = exact_match(model, ch_a, ch_b)
    print("-" * 72)
    print("held-out exact match (uniform, n=%d): %.4f%%" % (args.n, 100 * acc))
    print("unrestricted uniform pairs  (n=%d): %.4f%%" % (args.n, 100 * uacc))
    print("held-out exact match (carry-chain enriched): %.4f%%" % (100 * hacc))
    print("maximal-carry-chain stress cases: %.4f%%" % (100 * cacc))

    # end-to-end through the public add() on CPU, exactly as the grader would
    cpu_model, _ = sub.build_model()
    idx = torch.randperm(ho_a.shape[0])[:2000]
    av = D.digits_to_value(ho_a[idx]).tolist()
    bv = D.digits_to_value(ho_b[idx]).tolist()
    good = sum(sub.add(cpu_model, x, y) == x + y for x, y in zip(av, bv))
    print("public add() on 2000 held-out pairs: %d/2000" % good)
    for x, y in [(99999999, 99999999), (10000000, 10000000), (19999999, 10000001),
                 (45454545, 54545455), (12345678, 87654321), (99999999, 10000001),
                 (50000000, 50000000), (11111111, 88888889)]:
        got = sub.add(cpu_model, x, y)
        print("   %8d + %8d -> %9d  (want %9d) %s"
              % (x, y, got, x + y, "ok" if got == x + y else "MISMATCH"))

    # ---------------- is the attention doing real work? ----------------
    sub.STORE_ATTN = True
    attn_layers = [b.attn for b in model.blocks if b.attn is not None]
    print("-" * 72)
    print("attention layers:", len(attn_layers))

    a, b = D.holdout_chain(4096, dev, seed=5150)
    _ = model(*to_tokens(a, b))

    # ground-truth carry into each place, for the read-out check below
    ssum = (a + b).to(torch.long)
    ssum = torch.cat([torch.zeros_like(ssum[:, :1]), ssum, torch.zeros_like(ssum[:, :1])], 1)
    generates = ssum >= 10
    transparent = ssum == 9
    carry_in = torch.zeros_like(ssum, dtype=torch.bool)
    cur = torch.zeros(ssum.shape[0], dtype=torch.bool, device=dev)
    for i in range(10):
        carry_in[:, i] = cur
        cur = generates[:, i] | (transparent[:, i] & cur)

    lookup = readout = 0.0
    for li, layer in enumerate(attn_layers):
        att = layer.attn                      # (B, H, T, T)
        spread = att.std(0).max().item()      # variation across inputs
        rows = att[:, :, 1:, :]
        ent = (-(rows * rows.clamp_min(1e-9).log()).sum(-1)).mean().item()
        print("  layer %d: max std of attention weight across inputs = %.4f" % (li, spread))
        print("     mean row entropy = %.4f nats (0 = hard look-up, 2.2 = uniform)" % ent)
        hit = tot = 0
        for i in range(1, 10):
            mass = (att[:, :, i, :i].mean(1) * generates[:, :i].float()).sum(-1)
            hit += int(((mass > 0.5) == carry_in[:, i]).sum())
            tot += a.shape[0]
        readout = hit / tot
        print("     (attention mass on generating places > 0.5) == true carry: %.2f%%"
              % (100 * readout))

        # The carry circuit: the query at place i must find the nearest lower
        # place that is not carry-transparent (a_j + b_j != 9) and report
        # whether it generates.  Only the rows whose answer is "carry" pin down
        # where the head has to look: when the nearest non-transparent place is
        # blocking, every non-generating place (the sink included) is an equally
        # correct target, so the argmax is genuinely free there.
        s = (a + b).to(torch.long)                                  # (B, 8) place sums
        s = torch.cat([torch.zeros_like(s[:, :1]), s, torch.zeros_like(s[:, :1])], 1)
        nonprop = (s != 9)                                          # (B, 10)
        agree = onreal = tot = 0
        for i in range(1, 10):
            j = torch.arange(i, device=dev)
            cand = nonprop[:, :i].float() * (j + 1).float()          # prefer largest j
            target = cand.argmax(-1)                                 # nearest non-transparent
            picked = att[:, :, i, :i].mean(1).argmax(-1)
            need = carry_in[:, i]                     # rows that force the look-up
            agree += int(((picked == target) & need).sum())
            onreal += int(generates[:, :i].gather(1, picked[:, None])[need].sum())
            tot += int(need.sum())
        nearest, lookup = agree / max(tot, 1), onreal / max(tot, 1)
        print("     where a carry must be found, argmax lands on a place that"
              " generates: %.2f%%" % (100 * lookup))
        print("     ... and on the *nearest* non-transparent one:        %.2f%%"
              % (100 * nearest))

    # ---------------- ablations ----------------
    print("-" * 72)
    import model_src as MS

    orig = MS.SelfAttention.forward
    sub_cls = type(attn_layers[0])
    orig_sub = sub_cls.forward

    def uniform_forward(self, x, mask):
        B, T, _ = x.shape
        H, Dh = self.n_heads, self.d_head
        v = self.wv(x).view(B, T, H, Dh).transpose(1, 2)
        att = mask.float() / mask.float().sum(-1, keepdim=True)
        y = torch.matmul(att.expand(B, H, T, T), v).transpose(1, 2).reshape(B, T, H * Dh)
        return self.wo(y)

    def frozen_forward(self, x, mask):
        """Replace content-based attention with the average pattern it produces
        (a fixed, input-independent pattern of the same shape)."""
        B, T, _ = x.shape
        H, Dh = self.n_heads, self.d_head
        v = self.wv(x).view(B, T, H, Dh).transpose(1, 2)
        att = self.frozen.to(x.dtype).expand(B, H, T, T)
        y = torch.matmul(att, v).transpose(1, 2).reshape(B, T, H * Dh)
        return self.wo(y)

    # Build the frozen pattern from the *same* distribution the ablation is
    # scored on, which is the strongest form of the "it is just a fixed
    # pattern" hypothesis: the best single input-independent pattern for the
    # graded distribution, with every weight and the value path left intact.
    _ = model(*to_tokens(ho_a[:8192], ho_b[:8192]))
    for layer in attn_layers:
        layer.frozen = layer.attn.mean(0, keepdim=True)

    # Two evaluation sets per ablation.  On *uniform* pairs a fixed pattern is
    # not very damaging, because the mean pattern is close to a one-place shift
    # and a purely local carry rule ("carry iff the place below generates")
    # already gets about half of all 9-digit sums exactly right -- that number
    # says more about the task than about the model.  The maximal carry chains
    # are the discriminating set: resolving them *requires* looking at a place
    # chosen by content, so any input-independent pattern must collapse there.
    scores = {}
    def both(name, tag):
        u = exact_match(model, ho_a[:50000], ho_b[:50000])
        c = exact_match(model, ch_a, ch_b)
        scores[tag] = c
        scores[tag + "/uniform"] = u
        print("  attention -> %-26s uniform %8.4f%%   carry chains %8.4f%%"
              % (name, 100 * u, 100 * c))

    for name, fn, tag in [("uniform-over-causal-mask", uniform_forward, "uniform-mask"),
                          ("frozen average pattern", frozen_forward, "frozen")]:
        sub_cls.forward = fn
        both(name, tag)
    sub_cls.forward = orig_sub
    MS.SelfAttention.forward = orig

    # Same ablation applied to one layer at a time.  Layer 0 is a learned local
    # shift, which is legitimately close to input-independent; the layer that
    # has to be content-addressed is the last one, where the carry chain is
    # resolved.  Freezing *that* alone is the sharp test.
    def only(idx):
        def fwd(self, x, mask):
            if self is attn_layers[idx]:
                return frozen_forward(self, x, mask)
            return orig_sub(self, x, mask)
        return fwd

    for li in range(len(attn_layers)):
        sub_cls.forward = only(li)
        both("frozen layer %d only" % li, "frozen%d" % li)
    sub_cls.forward = orig_sub
    both("(restored)", "restored")

    # ---------------- verdict ----------------
    # "frozen average pattern" replaces the computed attention with the fixed
    # pattern it produces on average, keeping every weight and the value path
    # intact.  If the model survives that, its attention *was* a fixed pattern.
    print("-" * 72)
    carry_layer = "frozen%d" % (len(attn_layers) - 1)
    checks = [
        ("held-out accuracy >= 99%", acc >= 0.99, "%.4f%%" % (100 * acc)),
        ("maximal carry chains >= 99%", cacc >= 0.99, "%.4f%%" % (100 * cacc)),
        ("carry head argmax lands on a generating place >= 95%",
         lookup >= 0.95, "%.2f%%" % (100 * lookup)),
        ("carry read-out from attention mass >= 99%",
         readout >= 0.99, "%.2f%%" % (100 * readout)),
        ("freezing the carry attention breaks chains (< 50%)",
         scores[carry_layer] < 0.50, "%.4f%%" % (100 * scores[carry_layer])),
        ("freezing all attention breaks chains (< 50%)",
         scores["frozen"] < 0.50, "%.4f%%" % (100 * scores["frozen"])),
    ]
    for label, passed, value in checks:
        print("  [%s] %-46s %s" % ("PASS" if passed else "FAIL", label, value))
    print("  => %s" % ("ACCEPT" if all(p for _, p, _ in checks) else "REJECT"))
    print("=" * 72)


if __name__ == "__main__":
    main()
