import copy, random, sys
sys.path.append('/usr/local/lib/python3.11/dist-packages')
import torch
import submission
from train import TrainModel, digits, uniform, structured

model, meta = submission.build_model()
print('parameters', sum(p.numel() for p in model.parameters()), 'weight_tensors', len(submission._WEIGHTS), meta)
assert sum(p.numel() for p in model.parameters()) == 2385
assert all(isinstance(p, torch.nn.Parameter) for p in model.parameters())

# Compare exact exported model to its pre-gauge trained source.
source=TrainModel(2); source.load_state_dict(torch.load('/workspace/final.pt',map_location='cpu',weights_only=True)); source.eval()
tok=torch.randint(0,11,(256,25)); tok[:,16]=10
with torch.no_grad(): diff=(source(tok)-model(tok)).abs()
print('export logit difference max/mean',float(diff.max()),float(diff.mean()))

# Large GPU autoregressive validation of exact submission model.
model.cuda()
def check(total, generator):
    good=seen=0
    with torch.no_grad():
        while seen<total:
            n=min(8192,total-seen); a,b=generator(n,'cuda'); y=digits(a+b)
            ad,bd=digits(a)[:,:8],digits(b)[:,:8]
            x=torch.empty(n,17,dtype=torch.long,device='cuda');x[:,0:16:2]=ad;x[:,1:16:2]=bd;x[:,16]=10
            out=[]
            for j in range(9):
                d=model(x)[:,-1].argmax(1);out.append(d)
                if j<8:x=torch.cat((x,d[:,None]),1)
            good+=int((torch.stack(out,1)==y).all(1).sum());seen+=n
    return good,total
print('export random',check(500000,uniform))
print('export structured',check(500000,structured))
model.cpu()

cases=[(10_000_000,10_000_000),(99_999_999,99_999_999),(40_000_009,89_999_999),(99_000_001,99_899_999),(50_000_000,50_000_000)]
rng=random.Random(33033); cases += [(rng.randrange(10_000_000,100_000_000),rng.randrange(10_000_000,100_000_000)) for _ in range(2000)]
base=[submission.add(model,a,b) for a,b in cases]
print('direct API',sum(x==a+b for x,(a,b) in zip(base,cases)),len(cases))

abl=copy.deepcopy(model)
with torch.no_grad():
    for layer in abl.o: layer.weight.zero_()
changed=sum(submission.add(abl,a,b)!=x for (a,b),x in zip(cases[:50],base[:50]))
corrupt=copy.deepcopy(model)
with torch.no_grad(): corrupt.head.weight.zero_(); corrupt.head.bias.zero_()
changed_head=sum(submission.add(corrupt,a,b)!=x for (a,b),x in zip(cases[:50],base[:50]))

# Directly measure variation of first-layer attention distributions.
def amap(a,b):
    left=[ord(c)-48 for c in f'{a:08d}'[::-1]];right=[ord(c)-48 for c in f'{b:08d}'[::-1]]
    x=torch.tensor([[d for pair in zip(left,right) for d in pair]+[10]])
    h=torch.nn.functional.embedding(x,model.token_weight())+model.positions()[:17];n=model.norm(h);q=model.q[0](n).view(1,17,4,5);k=torch.nn.functional.linear(n,model.shared_projection(model.k_free))
    s=torch.einsum('bthd,bsd->bhts',q,k)*.4472135954999579
    return torch.softmax(s+model.causal[:17,:17],-1)
a1=amap(12_345_678,87_654_321);a2=amap(91_234_567,18_765_432)
print('attention ablation changed',changed,'/ 50; head corruption changed',changed_head,'/ 50; attention map max delta',float((a1-a2).abs().max()))
