import copy, random, sys, time
import torch
sys.path.insert(0, '/workspace')
import submission

torch.manual_seed(12345)
random.seed(12345)
model, metadata = submission.build_model()
model = model.cuda().eval()
parameters = sum(p.numel() for p in model.parameters())

@torch.inference_mode()
def decode(pairs, tested_model=model):
    device = next(tested_model.parameters()).device
    rows = []
    for a, b in pairs:
        ad = [ord(c)-48 for c in str(a)[::-1]]
        bd = [ord(c)-48 for c in str(b)[::-1]]
        row = []
        for x, y in zip(ad, bd): row.extend((x, y))
        rows.append(row + [10])
    prefix = torch.tensor(rows, dtype=torch.long, device=device)
    generated = []
    for _ in range(9):
        digit = tested_model(prefix)[:, -1].argmax(-1)
        generated.append(digit)
        prefix = torch.cat((prefix, digit[:, None]), 1)
    digits = torch.stack(generated, 1).cpu().tolist()
    return [int(''.join(map(str, row[::-1]))) for row in digits]

random_pairs = [(random.randrange(10_000_000, 100_000_000), random.randrange(10_000_000, 100_000_000)) for _ in range(100_000)]
edge = set()
values = [10_000_000,10_000_001,10_000_009,10_000_010,10_000_099,10_000_100,10_000_999,10_001_000,10_009_999,10_010_000,10_099_999,10_100_000,10_999_999,11_111_111,19_999_999,20_000_000,49_999_999,50_000_000,88_888_888,89_999_999,90_000_000,98_999_999,99_000_000,99_900_000,99_990_000,99_999_000,99_999_900,99_999_990,99_999_998,99_999_999]
for a in values:
    for b in values: edge.add((a,b))
for _ in range(10000):
    a=random.randrange(10_000_000,90_000_001); edge.add((a,100_000_000-a))
for k in range(1,8):
    p=10**k
    for _ in range(1000):
        a=random.randrange(10_000_000,100_000_000)
        a=(a//p)*p
        if a<10_000_000:a=10_000_000
        b=random.randrange(10_000_000,100_000_000)
        edge.add((a,b))
edge=list(edge)

def accuracy(pairs, batch=4096):
    good=0
    for i in range(0,len(pairs),batch):
        part=pairs[i:i+batch]
        predictions=decode(part)
        good += sum(pred == a+b for pred,(a,b) in zip(predictions,part))
    return good,len(pairs)

start=time.time()
r_good,r_total=accuracy(random_pairs)
e_good,e_total=accuracy(edge)
# Exercise the public scalar interface itself.
api_pairs=random_pairs[:250]
api_good=sum(submission.add(model,a,b)==a+b for a,b in api_pairs)

probe=random_pairs[200:400]
baseline=decode(probe)
ablated=copy.deepcopy(model)
def no_attention(query,key,value,**kwargs):
    return torch.zeros_like(query), None
ablated.attention.forward=no_attention
no_attn=decode(probe,ablated)
attention_changes=sum(x!=y for x,y in zip(baseline,no_attn))

head_changed=copy.deepcopy(model)
with torch.no_grad(): head_changed.output.weight.zero_()
no_head=decode(probe,head_changed)
head_changes=sum(x!=y for x,y in zip(baseline,no_head))

# Directly compare attention distributions for two distinct token sequences.
with torch.inference_mode():
    sample=[]
    for a,b in probe[:2]:
        ad=[ord(c)-48 for c in str(a)[::-1]]; bd=[ord(c)-48 for c in str(b)[::-1]]; row=[]
        for x,y in zip(ad,bd):row.extend((x,y))
        sample.append(row+[10])
    tokens=torch.tensor(sample,device='cuda')
    pos=torch.arange(17,device='cuda')
    hidden=model.token(tokens)+model.position(pos)+model.pass_embedding[0]
    normalized=model.norm1(hidden)
    _,weights=model.attention(normalized,normalized,normalized,attn_mask=model.causal_mask[:17,:17],need_weights=True,average_attn_weights=False)
    attention_input_delta=(weights[0]-weights[1]).abs().max().item()

print('metadata',metadata)
print('parameters',parameters)
print('random',r_good,r_total,r_good/r_total)
print('edge',e_good,e_total,e_good/e_total)
print('public_add',api_good,len(api_pairs),api_good/len(api_pairs))
print('attention_ablation_changes',attention_changes,len(probe))
print('output_head_changes',head_changes,len(probe))
print('attention_input_delta',attention_input_delta)
print('seconds',time.time()-start)
assert r_good/r_total >= .99
assert e_good/e_total >= .99
assert api_good == len(api_pairs)
assert attention_changes > 0
assert head_changes > 0
assert attention_input_delta > 1e-5
