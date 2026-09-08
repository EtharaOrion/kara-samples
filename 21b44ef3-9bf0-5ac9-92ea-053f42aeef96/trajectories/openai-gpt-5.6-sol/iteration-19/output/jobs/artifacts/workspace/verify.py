import ast
import random
import time
import torch
import submission


def encode(a, b):
    p = torch.tensor([1,10,100,1000,10000,100000,1000000,10000000], device=a.device)
    da, db = a[:,None] // p % 10, b[:,None] // p % 10
    x = torch.empty(a.numel(), 17, dtype=torch.long, device=a.device)
    x[:,:16:2], x[:,1:16:2], x[:,16] = da, db, 10
    return x

@torch.no_grad()
def test_pairs(model, a, b, chunk=8192):
    good = total = 0
    for start in range(0, a.numel(), chunk):
        aa, bb = a[start:start+chunk], b[start:start+chunk]
        seq = encode(aa, bb)
        result = torch.zeros_like(aa)
        place = 1
        for _ in range(9):
            digit = model(seq)[:,-1].argmax(-1)
            result += digit * place
            place *= 10
            seq = torch.cat((seq, digit[:,None]), 1)
        good += (result == aa + bb).sum().item()
        total += aa.numel()
    return good, total

m, meta = submission.build_model()
print('metadata', meta)
print('parameters', sum(p.numel() for p in m.parameters()))
print('state_none', submission._TRAINED_STATE is None)
print('imports', sorted({n.names[0].name for n in ast.walk(ast.parse(open('/workspace/submission.py').read())) if isinstance(n, ast.Import)} | {n.module for n in ast.walk(ast.parse(open('/workspace/submission.py').read())) if isinstance(n, ast.ImportFrom)}))

m = m.cuda().eval()
torch.manual_seed(9919)
n = 500_000
a = torch.randint(10_000_000,100_000_000,(n,),device='cuda')
b = torch.randint(10_000_000,100_000_000,(n,),device='cuda')
print('random', test_pairs(m,a,b))

# Disjoint deterministic stress set: extrema, decimal boundaries, asymmetric 0/9 runs,
# sparse/repeated digits, all leading-digit pairings, and exact/near complements.
values = {10_000_000, 10_000_001, 10_000_009, 10_000_010, 10_000_099,
          10_000_100, 10_000_999, 10_009_999, 10_099_999, 10_999_999,
          90_000_000, 99_000_000, 99_900_000, 99_990_000, 99_999_000,
          99_999_900, 99_999_990, 99_999_998, 99_999_999}
for lead in range(1,10):
    values.update({lead*10_000_000, lead*10_000_000+9, lead*10_000_000+999,
                   lead*10_000_000+999_999, lead*11_111_111})
for power in (10,100,1000,10000,100000,1000000,10000000):
    for lead in range(1,10):
        center = lead*10_000_000
        for off in (-power-1,-power,-power+1,-1,0,1,power-1,power,power+1):
            values.add(max(10_000_000,min(99_999_999,center+off)))
values=sorted(values)
pairs={(x,y) for x in values for y in values}
for x in values:
    for target in (100_000_000,100_000_001,109_999_999,110_000_000,150_000_000,199_999_998):
        y=target-x
        if 10_000_000 <= y <= 99_999_999: pairs.add((x,y))
pa=torch.tensor([x for x,y in pairs],device='cuda'); pb=torch.tensor([y for x,y in pairs],device='cuda')
print('curated_count', len(pairs), 'curated', test_pairs(m,pa,pb))

# Dependence checks on fixed samples.
sa=a[:64]; sb=b[:64]
def outputs(model):
    seq=encode(sa,sb); out=[]
    for _ in range(9):
        d=model(seq)[:,-1].argmax(-1); out.append(d); seq=torch.cat((seq,d[:,None]),1)
    return torch.stack(out,1)
base=outputs(m)
backup=[p.detach().clone() for layer in m.o for p in layer.parameters()]
with torch.no_grad():
    for layer in m.o:
        for p in layer.parameters(): p.zero_()
ablated=outputs(m)
with torch.no_grad():
    i=0
    for layer in m.o:
        for p in layer.parameters(): p.copy_(backup[i]); i+=1
print('attention_ablation_changed', (base != ablated).any(1).sum().item(), '/64')
head=m.output.weight.detach().clone()
with torch.no_grad(): m.output.weight.zero_()
corrupt=outputs(m)
with torch.no_grad(): m.output.weight.copy_(head)
print('output_corruption_changed', (base != corrupt).any(1).sum().item(), '/64')

# Compare actual first-layer attention distributions for two operand pairs.
def attention(tokens):
    length=tokens.shape[1]
    x=m.token(tokens)+(m.pos_left[:length]@m.pos_right)
    z=m.attn_norm(x); q=m.q[0](z).view(tokens.shape[0],length,4,5).transpose(1,2)
    k=m.k(z).unsqueeze(1)
    scores=(q@k.transpose(-2,-1))*(5**-.5)
    return torch.softmax(scores.masked_fill(m.causal[:length,:length],-torch.inf),-1)
w=attention(encode(a[:2],b[:2]))
print('attention_input_delta', (w[0]-w[1]).abs().max().item())

m=m.cpu()
checks=[(10_000_000,10_000_000),(99_999_999,99_999_999),(40_000_009,89_999_999),(12_345_678,87_654_321)]
print('cpu_add', [(x,y,submission.add(m,x,y)) for x,y in checks])
