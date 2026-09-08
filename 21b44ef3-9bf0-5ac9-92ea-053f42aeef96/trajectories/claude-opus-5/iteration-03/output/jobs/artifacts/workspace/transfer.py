"""Warm-start an ensemble from an already-trained member.

Below ~75 parameters a random restart almost never finds the carry mechanism,
but a working model one size up is a good starting point.  Every ensemble
member is seeded with a copy of the donor, mapped onto the (usually narrower)
target shape, then perturbed by its own noise level -- member 0 is an exact
copy, the rest span a geometric ladder of perturbation sizes, so a single run
covers both fine-tuning and genuine re-exploration.

Width changes are handled by keeping a random subset of FFN hidden units,
different per member, which also searches over *which* unit to drop.
"""
import torch


def _ladder(E, smax, device):
    """Per-member noise scale: member 0 exact, then a geometric ramp."""
    t = torch.arange(E, device=device, dtype=torch.float32) / max(E - 1, 1)
    s = smax * (t ** 2)
    s[0] = 0.0
    return s


def warm_start(model, donor, sigma_max=0.6, seed=0, verbose=True):
    """Fill `model` (an EnsembleAdder) from the single-member dict `donor`."""
    device = next(model.parameters()).device
    g = torch.Generator(device=device).manual_seed(seed)
    E = model.E
    sig = _ladder(E, sigma_max, device)

    # Per-member choice of which FFN hidden units to keep, when narrowing.
    def unit_pick(name_w1, target_f):
        src_f = donor[name_w1].shape[1]
        if src_f == target_f:
            return torch.arange(target_f, device=device).expand(E, target_f)
        if src_f > target_f:
            keep = torch.stack([torch.randperm(src_f, generator=g,
                                               device=device)[:target_f]
                                for _ in range(E)])
            keep[0] = torch.arange(target_f, device=device)
            return keep
        pad = torch.randint(0, src_f, (E, target_f - src_f), device=device,
                            generator=g)
        base = torch.arange(src_f, device=device).expand(E, src_f)
        return torch.cat([base, pad], 1)

    # Dropping b_in2 is not the same as zeroing it.  Every token enters the
    # stream as emb[a] + emb[b], so folding b_in2/2 into every embedding row
    # reproduces the post-FFN residual exactly; the tied unembedding then adds
    # one constant to all ten logits at a position, which cancels in the
    # softmax.  Only the pre-attention FFN's own input shifts, so this is a far
    # better starting point than throwing the bias away.
    donor = dict(donor)
    if "b_in2" in donor and not getattr(model, "rb_in", True):
        donor["emb"] = donor["emb"] + donor["b_in2"].view(1, -1) / 2.0
        donor.pop("b_in2")
        if verbose:
            print("[warm_start] folded b_in2 into emb", flush=True)

    # Full table -> factorised table: the best rank-r starting point is the
    # truncated SVD of the donor's embedding, split evenly between the two
    # factors.
    if getattr(model, "emb_rank", 0) and "emb" in donor:
        r = model.emb_rank
        u, s, v = torch.linalg.svd(donor["emb"].float(), full_matrices=False)
        root = s[:r].sqrt()
        donor["emb_lo"] = u[:, :r] * root
        donor["emb_up"] = root.unsqueeze(1) * v[:r]
        kept = float((s[:r] ** 2).sum() / (s ** 2).sum())
        donor.pop("emb")
        if verbose:
            print(f"[warm_start] emb -> rank {r} via SVD "
                  f"({100 * kept:.2f}% of the energy kept)", flush=True)

    # Target pins the digit-0 row to a structural zero.  Shifting every row by
    # -emb[0] preserves all *differences* between the digit codes -- the part
    # that carries the information -- and moves every token by one constant,
    # which is a much smaller perturbation than truncating the row away.
    if getattr(model, "emb0_zero", False):
        key = "emb_lo" if "emb_lo" in donor else "emb"
        want = getattr(model, key, None)
        if key in donor and want is not None \
                and donor[key].shape[0] == want.shape[1] + 1:
            donor[key] = (donor[key] - donor[key][:1])[1:]
            if verbose:
                print(f"[warm_start] centred {key} on digit 0 and dropped "
                      "the row", flush=True)

    picks = {}
    if model.d_ff_in and "w_in1" in donor:
        picks["in"] = unit_pick("w_in1", model.d_ff_in)
    if "w_out1" in donor:
        picks["out"] = unit_pick("w_out1", model.d_ff_out)

    copied, skipped = [], []
    with torch.no_grad():
        for name in model.spec:
            tgt = getattr(model, name)
            if name not in donor:
                skipped.append(name)
                continue
            src = donor[name].to(device)
            which = "in" if "_in" in name else ("out" if "_out" in name else None)
            if which in picks:
                idx = picks[which]                      # (E, f')
                if name in ("w_in1", "w_out1"):         # (d, f) -> gather cols
                    val = src.unsqueeze(0).expand(E, *src.shape)
                    val = torch.gather(val, 2, idx.unsqueeze(1)
                                       .expand(E, src.shape[0], idx.shape[1]))
                elif name in ("b_in1", "b_out1"):       # (f,)
                    val = src.unsqueeze(0).expand(E, -1).gather(1, idx)
                elif name in ("w_in2", "w_out2"):       # (f, d) -> gather rows
                    val = src.unsqueeze(0).expand(E, *src.shape)
                    val = torch.gather(val, 1, idx.unsqueeze(2)
                                       .expand(E, idx.shape[1], src.shape[1]))
                else:
                    val = src.unsqueeze(0).expand(E, *src.shape)
            else:
                if src.shape != tgt.shape[1:]:
                    skipped.append(name)
                    continue
                val = src.unsqueeze(0).expand(E, *src.shape)
            val = val.contiguous().float()
            # std() of a 1-element tensor is NaN, which would poison member 0.
            scale = (src.float().std() if src.numel() > 1
                     else src.float().abs().squeeze()).clamp(min=1e-3)
            noise = torch.randn(val.shape, generator=g, device=device) * scale
            tgt.copy_(val + noise * sig.view(-1, *([1] * (val.dim() - 1))))
            copied.append(name)
    if verbose:
        print(f"[warm_start] copied {copied}", flush=True)
        print(f"[warm_start] random-init {skipped}", flush=True)
    return model
