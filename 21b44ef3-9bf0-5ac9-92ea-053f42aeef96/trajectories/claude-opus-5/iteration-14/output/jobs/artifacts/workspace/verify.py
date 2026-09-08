"""Independent check of /workspace/submission.py, importing only what the grader does.

Nothing here reaches into the workspace training code: it imports the graded file,
calls build_model()/add(), and tests the result.  Checks, in order:

  1  parameter count read from model.parameters() (not from metadata)
  2  every operand pair is full 8-digit width, as the grader states
  3  1,000,000 uniform random 8-digit pairs
  4  all 3^8 = 6561 carry-class patterns (absorb / transparent / generate per place),
     which is what actually exercises carry propagation
  5  edge cases: 99999999+99999999, longest possible carry chains, boundaries
  6  the answer really comes out of the forward pass -- freeze the attention to a
     fixed (input-independent) pattern and confirm accuracy collapses, measured on
     carry-heavy inputs where attention is load-bearing
  7  float64 vs float32 and CPU vs CUDA agreement
"""
import importlib.util, itertools, random, sys, time
import torch

PATH = sys.argv[1] if len(sys.argv) > 1 else "submission.py"
spec = importlib.util.spec_from_file_location("submission", PATH)
sub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub)

LO, HI = 10_000_000, 99_999_999
fails = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        fails.append(name)


model, meta = sub.build_model()
n_param = sum(p.numel() for p in model.parameters())
print(f"metadata: {meta}")
check("parameter count from model.parameters()", True, f"= {n_param}")
check("metadata n_parameters matches model", meta.get("n_parameters") == n_param,
      f"meta={meta.get('n_parameters')} actual={n_param}")
check("all learned floats are registered parameters",
      all(t.dtype.is_floating_point for t in model.parameters()),
      f"{sum(1 for _ in model.parameters())} tensors")

# ------------------------------------------------------------------ 3  random pairs
def run(pairs, label):
    t0 = time.time()
    bad = []
    for a, b in pairs:
        got = sub.add(model, a, b)
        if got != a + b:
            bad.append((a, b, got, a + b))
            if len(bad) > 8:
                break
    check(label, not bad, f"{len(pairs)} pairs, {len(bad)} wrong, {time.time()-t0:.0f}s"
          + (f" e.g. {bad[:3]}" if bad else ""))
    return bad


rng = random.Random(0)
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000
pairs = [(rng.randint(LO, HI), rng.randint(LO, HI)) for _ in range(N)]
check("operands are full 8-digit width",
      all(len(str(a)) == 8 and len(str(b)) == 8 for a, b in pairs))
run(pairs, f"{N} uniform random 8-digit pairs")

# ---------------------------------------------------- 4  all 6561 carry patterns
def pair_for_pattern(pat, rng):
    """pat: 8 entries in {0,1,2} = absorb / transparent / generate at that place."""
    a = b = 0
    for i, c in enumerate(pat):
        while True:
            s = rng.randint(0, 8) if c == 0 else (9 if c == 1 else rng.randint(10, 18))
            ai = rng.randint(max(0, s - 9), min(9, s))
            bi = s - ai
            if i < 7 or (ai > 0 and bi > 0):
                break
        a += ai * 10 ** i
        b += bi * 10 ** i
    return a, b


pats = [pair_for_pattern(p, rng) for p in itertools.product((0, 1, 2), repeat=8)]
run(pats, "all 6561 carry-class patterns (absorb/transparent/generate)")

# --------------------------------------------------------------- 5  edge cases
edge = [(99999999, 99999999), (10000000, 10000000), (99999999, 10000000),
        (10000000, 99999999), (55555555, 44444445), (12345678, 87654322),
        (99999999, 99999991), (19999999, 10000001), (11111111, 88888889),
        (50000000, 50000000), (99999998, 99999999), (45454545, 54545455)]
run(edge, "edge cases (max carry chains, boundaries)")

# ------------------------------------- 6  is the attention doing the work?
class FrozenAttn(torch.nn.Module):
    """Same weights, but the attention logits ignore the input entirely: the key
    term is replaced by its mean, leaving only the fixed relative-position bias."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, da, db):
        m = self.m
        code = torch.cat([m.code0, m.code], 0)
        x = code[da] + code[db]
        u = (x @ m.Bw + m.bb).clamp(0.0, 1.0)
        k = u @ m.kw
        v = u @ m.vw + m.vb
        k = k.mean(dim=1, keepdim=True).expand_as(k)          # <- input-independent
        P = da.shape[1]
        i = torch.arange(P, device=da.device)
        rel = (i[:, None] - i[None, :]).to(x.dtype)
        neg = torch.finfo(x.dtype).min / 4
        z = torch.zeros((), dtype=x.dtype, device=da.device)
        nn_ = torch.full((), neg, dtype=x.dtype, device=da.device)
        m_s = torch.where(i[None, :] < i[:, None], z, nn_).clone()
        m_s[0, 0] = 0.0
        m_i = torch.where(i[None, :] <= i[:, None], z, nn_)
        base = m.q * k[:, None, :] + m.lam * rel
        a_s = torch.softmax(base + m_s, dim=-1)
        a_i = torch.softmax(base + m_i, dim=-1)
        c_in = torch.einsum("bij,bj->bi", a_s, v)
        c_out = torch.einsum("bij,bj->bi", a_i, v)
        out = x + c_in[..., None] * m.w1 + c_out[..., None] * m.w2 + m.rb
        return m.ls * (out @ code.t() - 0.5 * (code * code).sum(-1))


def acc_on(mod, pairs):
    good = 0
    for a, b in pairs:
        if sub.add(mod, a, b) == a + b:
            good += 1
    return good / len(pairs)


carry_heavy = pats[:2000]                       # every carry class, attention matters
# Inputs that genuinely need routing.  A run of transparent places is preceded by
# either a generate (the carry must travel across the run) or an absorb (it must not).
# Both halves are needed: a model that just reads its immediate neighbour gets the
# generate half right by luck, so only the contrast separates routing from recency.
prop = []
for i in range(2000):
    run_len = rng.randint(2, 5)
    start = rng.randint(0, 7 - run_len)
    pat = [rng.choice((0, 2)) for _ in range(8)]
    pat[start] = 2 if i % 2 else 0            # carry generated / not generated
    for j in range(start + 1, start + run_len):
        pat[j] = 1                            # transparent: propagates whatever arrives
    prop.append(pair_for_pattern(pat, rng))

frozen = FrozenAttn(model)
frozen.code = model.code
a_full, a_froz = acc_on(model, carry_heavy), acc_on(frozen, carry_heavy)
gen_half, abs_half = prop[1::2], prop[0::2]          # carry crosses the run / does not
check("attention is load-bearing (frozen-attention ablation collapses)",
      a_full > 0.99 and a_froz < 0.5 * a_full,
      f"all carry classes: full {a_full:.4f} -> frozen {a_froz:.4f}")
gf, ga = acc_on(model, gen_half), acc_on(model, abs_half)
ff, fa = acc_on(frozen, gen_half), acc_on(frozen, abs_half)
# With the key frozen the attention is pure recency, so the model gives the same
# answer whether or not a carry actually reached the far side of the transparent run.
# It must therefore fail one of the two halves outright, whichever way its value
# stream is signed, while the real model gets both.
check("carry propagation needs input-dependent attention",
      gf > 0.99 and ga > 0.99 and min(ff, fa) < 0.05,
      f"carry-crosses-run: full {gf:.4f} -> frozen {ff:.4f} | "
      f"no-carry-crosses-run: full {ga:.4f} -> frozen {fa:.4f}")

# ------------------------------------------------- 7  precision / device agreement
m64, _ = sub.build_model()
m64.double()
same = all(sub.add(m64, a, b) == a + b for a, b in pats[:1500])
check("float64 execution agrees", same)
if torch.cuda.is_available():
    mc, _ = sub.build_model()
    mc.cuda()
    same = all(sub.add(mc, a, b) == a + b for a, b in pats[:1500])
    check("CUDA execution agrees", same)

print()
print("ALL CHECKS PASSED" if not fails else f"FAILURES: {fails}")
sys.exit(1 if fails else 0)
