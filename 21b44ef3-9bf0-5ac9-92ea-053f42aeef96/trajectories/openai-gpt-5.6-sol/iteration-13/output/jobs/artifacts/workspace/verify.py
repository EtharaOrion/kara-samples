import copy, random, sys, torch
sys.path.insert(0,'/workspace')
import submission
m,meta=submission.build_model()
print('metadata',meta)
print('parameters',sum(p.numel() for p in m.parameters()))
random.seed(1313)
cases=[]
for _ in range(10000): cases.append((random.randint(10_000_000,99_999_999),random.randint(10_000_000,99_999_999)))
vals=[10_000_000,10_000_001,10_000_009,10_000_010,10_000_099,10_000_100,10_000_999,10_001_000,10_009_999,10_010_000,10_099_999,10_100_000,10_999_999,11_111_111,50_000_000,88_888_888,89_999_999,90_000_000,98_999_999,99_000_000,99_900_000,99_990_000,99_999_000,99_999_900,99_999_990,99_999_998,99_999_999]
edges=[(a,b) for a in vals for b in vals]
for x in vals:
    y=100_000_000-x
    if 10_000_000<=y<=99_999_999: edges.extend([(x,y),(y,x)])
def check(model,cs):
    bad=[]
    for a,b in cs:
        got=submission.add(model,a,b)
        if got!=a+b and len(bad)<20:bad.append((a,b,got,a+b))
    return len(cs)-len([1 for a,b in cs if submission.add(model,a,b)!=a+b]),bad
# Run once without duplicated calls for speed.
def run(model,cs):
    good=0; bad=[]
    for a,b in cs:
        got=submission.add(model,a,b); want=a+b
        good += got==want
        if got!=want and len(bad)<20:bad.append((a,b,got,want))
    return good,bad
print('random',run(m,cases))
print('edges',run(m,edges))
baseline=[submission.add(m,a,b) for a,b in cases[:100]]
abl=copy.deepcopy(m)
with torch.no_grad():
    for block in abl.blocks:
        block.attention.in_proj_weight.zero_(); block.attention.in_proj_bias.zero_(); block.attention.out_proj.weight.zero_(); block.attention.out_proj.bias.zero_()
print('attention_changed',sum(submission.add(abl,a,b)!=v for (a,b),v in zip(cases[:100],baseline)))
corrupt=copy.deepcopy(m)
with torch.no_grad():corrupt.output.weight.zero_()
print('output_changed',sum(submission.add(corrupt,a,b)!=v for (a,b),v in zip(cases[:100],baseline)))
# Record input dependence directly from attention output at the first normalized layer.
t1=[]
for a,b in cases[:2]:
    sa,sb=str(a)[::-1],str(b)[::-1]; t1.append([int(d) for p in zip(sa,sb) for d in p]+[10]*9)
t=torch.tensor(t1)
x=m.token(t)+m.position
y=m.blocks[0].norm1(x)
_,w=m.blocks[0].attention(y,y,y,need_weights=True,average_attn_weights=False)
print('attention_input_delta',float((w[0]-w[1]).abs().max()))
