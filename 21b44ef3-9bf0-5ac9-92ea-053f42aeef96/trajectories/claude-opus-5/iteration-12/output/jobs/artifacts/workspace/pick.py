"""Evaluate the saved members of a checkpoint at several place-counts."""

import argparse
import torch
from torch.func import functional_call, stack_module_state, vmap

import data
from adder import DigitPairAdder
from train import ARCHES, DiscoveryAdder


def load_stack(path, device="cuda"):
    ck = torch.load(path, map_location=device, weights_only=False)
    cls = DiscoveryAdder if ck["cfg"].get("discovery") else ARCHES[ck["cfg"].get("arch", "base")]
    params = {k: v.to(device) for k, v in ck["params"].items()}
    n = next(iter(params.values())).shape[0]
    base = cls().to("meta")
    bufs = {k: v.to(device).unsqueeze(0).expand(n, *v.shape).contiguous()
            for k, v in ck["buffers"].items()}

    def fwd(p, b, da, db):
        return functional_call(base, (p, b), (da, db))

    batched = vmap(fwd, in_dims=(0, 0, None, None))
    return ck, params, bufs, batched, n, cls


@torch.no_grad()
def acc_at(params, bufs, batched, n, batch=32768, seed=5, device="cuda", held_only=True):
    g = torch.Generator(device=device).manual_seed(seed)
    da, db, tgt, bucket = data.sample(batch, n, device, g, force_msb=(n > 1))
    if held_only:
        sel = bucket == 0
        da, db, tgt = da[sel], db[sel], tgt[sel]
    pred = batched(params, bufs, da, db).argmax(-1)
    return (pred[..., 1:] == tgt[..., 1:].unsqueeze(0)).all(-1).float().mean(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--places", default="1,2,3,5,8,11")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()
    ck, params, bufs, batched, n, cls = load_stack(args.ckpt)
    places = [int(p) for p in args.places.split(",")]
    accs = {p: acc_at(params, bufs, batched, p) for p in places}
    key = accs[max(places)] + accs[8] if 8 in accs else accs[max(places)]
    order = key.argsort(descending=True)[: args.top]
    print(f"{args.ckpt}: {n} members, cfg places={ck['cfg'].get('places')} "
          f"sched={ck['cfg'].get('sched')} ls={ck['cfg'].get('ls')}")
    hdr = "member  " + "  ".join(f"n={p:<7}" for p in places)
    print(hdr)
    for i in order.tolist():
        print(f"{i:6d}  " + "  ".join(f"{accs[p][i].item():<9.5f}" for p in places))
    return ck, params


if __name__ == "__main__":
    main()
