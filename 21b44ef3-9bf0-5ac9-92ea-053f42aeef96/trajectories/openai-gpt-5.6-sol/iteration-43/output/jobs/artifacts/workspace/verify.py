import random
import torch
import submission

m, metadata = submission.build_model()
assert sum(p.numel() for p in m.parameters()) == 2281
random.seed(91827)
values = {10_000_000, 10_000_001, 10_000_009, 10_000_099, 10_000_999,
          10_009_999, 10_099_999, 10_999_999, 11_111_111, 19_999_999,
          20_000_000, 20_000_001, 49_999_999, 50_000_000, 50_000_001,
          89_999_999, 90_000_000, 98_999_999, 99_000_001, 99_899_999,
          99_900_000, 99_990_000, 99_999_000, 99_999_900, 99_999_990,
          99_999_998, 99_999_999}
for k in range(1, 8):
    q = 10 ** k
    for lead in range(1, 10):
        base = lead * 10_000_000
        for delta in (-q-1, -q, -q+1, -1, 0, 1, q-1, q, q+1):
            if 10_000_000 <= base + delta <= 99_999_999:
                values.add(base + delta)
grid = [(a, b) for a in sorted(values) for b in sorted(values)]
random.shuffle(grid)
pairs = grid[:1500]
pairs += [(random.randint(10_000_000, 99_999_999), random.randint(10_000_000, 99_999_999)) for _ in range(1500)]
bad=[]
for a,b in pairs:
    got=submission.add(m,a,b)
    if got != a+b:
        bad.append((a,b,got,a+b))
        if len(bad)>=10: break
print('direct',len(pairs)-len(bad),'/',len(pairs),'bad',bad)

# Capture input-dependent attention probabilities by reproducing the first score map.
def amap(a,b):
    sa=str(a)[::-1];sb=str(b)[::-1];t=[]
    for i in range(8): t.extend((ord(sa[i])-48,ord(sb[i])-48))
    t.append(10); tokens=torch.tensor([t])
    x=torch.nn.functional.embedding(tokens,torch.nn.functional.pad(m.token_free,(0,1)))+m._position()[:17]
    z=torch.nn.functional.layer_norm(x,(20,),bias=m.attn_norm_bias)
    q=torch.nn.functional.linear(z,m.q[0]).view(1,17,4,5).transpose(1,2)
    k=torch.nn.functional.linear(z,m._projection(m.k_free,submission.K_FIXED))
    return torch.einsum('bhtd,bsd->bhts',q,k).softmax(-1)
print('attention_input_delta',float((amap(12345678,87654321)-amap(98765432,11111111)).abs().max()))

base=[submission.add(m,*p) for p in pairs[:40]]
with torch.no_grad():
    saved=[p.clone() for p in m.o]
    for p in m.o: p.zero_()
ablated=[submission.add(m,*p) for p in pairs[:40]]
with torch.no_grad():
    for p,s in zip(m.o,saved): p.copy_(s)
    head=m.output_free.clone();m.output_free.zero_()
corrupt=[submission.add(m,*p) for p in pairs[:40]]
with torch.no_grad(): m.output_free.copy_(head)
print('attention_ablation_changes',sum(x!=y for x,y in zip(base,ablated)),'/40')
print('output_corruption_changes',sum(x!=y for x,y in zip(base,corrupt)),'/40')
assert not bad
assert sum(x!=y for x,y in zip(base,ablated)) >= 35
assert sum(x!=y for x,y in zip(base,corrupt)) >= 35
