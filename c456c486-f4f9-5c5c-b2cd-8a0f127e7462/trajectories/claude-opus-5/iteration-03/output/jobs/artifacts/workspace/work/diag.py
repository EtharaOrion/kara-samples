"""Inspect a checkpoint: attention pattern, learned key values, failure modes."""
import sys, torch, tiny

ck = torch.load(sys.argv[1], map_location="cuda")
i = int(sys.argv[2]) if len(sys.argv) > 2 else ck["best_idx"]
cfg = tiny.Cfg(**ck["cfg"])
p = {k: v[i:i + 1] for k, v in ck["params"].items()}
tab = tiny.pair_row_table("cuda")
rel, causal = tiny.make_masks("cuda")
print(f"cfg={cfg} params={tiny.param_count(cfg)} member {i} step {ck['step']} "
      f"unif {ck['acc'][i]:.4f} stress {ck['sacc'][i]:.4f}")
print("slope", p["slope"].item(), "bq", p.get("bq", torch.zeros(1)).item(), "bk", p.get("bk", torch.zeros(1)).item())
print("U", [round(v, 3) for v in p["U"][0].tolist()])

# key value as a function of the digit pair
d = torch.arange(10, device="cuda")
A, Bd = torch.meshgrid(d, d, indexing="ij")
c = p["U"][0][A] + p["U"][0][Bd]
e = p["P"][0][tab[A, Bd]]
x = torch.stack([c, e] + [torch.zeros_like(c)] * (cfg.d_model - 2), -1)
k = (x * p["wk"][0]).sum(-1) + p.get("bk", torch.zeros(1, device="cuda"))[0]
q = (x * p["wq"][0]).sum(-1) + p.get("bq", torch.zeros(1, device="cuda"))[0]
v = (x * p["wv"][0]).sum(-1)
s = (A + Bd)
print("\nkey by digit-sum s (mean/min/max over pairs):")
for ss in range(19):
    m = s == ss
    tag = " <- PROPAGATE" if ss == 9 else (" (generate)" if ss >= 10 else "")
    print(f"  s={ss:2d}  k {k[m].mean():+8.3f} [{k[m].min():+8.3f},{k[m].max():+8.3f}]  "
          f"v {v[m].mean():+8.3f}  q {q[m].mean():+7.3f}{tag}")

# per-position accuracy and worst cases
g = torch.Generator(device="cuda").manual_seed(5)
a, b, t = tiny.sample_batch(200000, "cuda", g, regime_p=(1.0, 0, 0, 0))
with torch.no_grad():
    pred = tiny.forward(p, cfg, a, b, tab, rel, causal).argmax(-1)[0]
acc_pos = (pred == t).float().mean(0)
print("\nper-position acc:", [round(x_, 4) for x_ in acc_pos.tolist()])
print("exact:", (pred == t).all(1).float().mean().item())

# accuracy vs longest propagate run in the sample
run = torch.zeros(a.shape[0], dtype=torch.long, device="cuda")
cur = torch.zeros_like(run)
for j in range(1, 15):
    isp = (a[:, j] + b[:, j]) == 9
    cur = torch.where(isp, cur + 1, torch.zeros_like(cur))
    run = torch.maximum(run, cur)
okm = (pred == t).all(1)
print("\nexact-match by longest propagate run:")
for r in range(0, 7):
    m = run == r
    if m.sum() > 0:
        print(f"  run={r}: {okm[m].float().mean():.4f}  (n={m.sum().item()})")

# canonical carry-chain probes
def probe(x1, x2):
    da = [0] + [(x1 // 10 ** j) % 10 for j in range(15)]
    db = [0] + [(x2 // 10 ** j) % 10 for j in range(15)]
    aa = torch.tensor([da], device="cuda")
    bb = torch.tensor([db], device="cuda")
    with torch.no_grad():
        dg = tiny.forward(p, cfg, aa, bb, tab, rel, causal).argmax(-1)[0, 0].tolist()
    return sum(dv * 10 ** j for j, dv in enumerate(dg))


print("\nprobes:")
for x1, x2 in [(999, 1), (50, 50), (9999999, 1), (99999999999999, 1), (12345, 54321),
               (1, 999999999999), (999999999999999 // 10, 1)]:
    got = probe(x1, x2)
    print(f"  {x1} + {x2} = {got} {'OK' if got == x1 + x2 else 'WRONG (want %d)' % (x1 + x2)}")
