import torch
from train import Adder, outputs, random_batch, carry_batch

ckpt = torch.load('/workspace/best.pt', weights_only=True)
w, h, r = ckpt['config']
model = Adder(w, h, r).cuda().eval()
model.load_state_dict(ckpt['model'])

@torch.no_grad()
def check(left, right, label):
    target = outputs(left, right)
    pred = model(left, right).argmax(-1)
    errors = pred.ne(target).any(1)
    print(label, int(errors.sum()), '/', len(left))
    if errors.any():
        ids = errors.nonzero().flatten()[:5]
        for i in ids:
            print(' ', left[i].tolist(), right[i].tolist(), target[i].tolist(), pred[i].tolist())

# Two million fresh ordinary cases.
for i in range(4):
    l, r, _ = random_batch(524288)
    check(l, r, f'random-{i}')

# Every homogeneous repeated digit pair and all single-boundary carry chains.
pairs = []
M = 10**14 - 1
for da in range(10):
    for db in range(10):
        pairs.append((int(str(da)*14), int(str(db)*14)))
for start in range(14):
    for length in range(1, 15-start):
        base = 10**start
        chain = (10**length - 1) * base
        pairs.extend([(chain, base), (M-chain, chain), (10**(start+length)-base, base)])
pairs += [(0,0),(M,0),(M,M),(M,1),(1,M),(99999999999990,9),(50000000000000,50000000000000)]
left=torch.zeros(len(pairs),15,dtype=torch.long,device='cuda')
right=torch.zeros_like(left)
for j,(a,b) in enumerate(pairs):
    for p in range(14):
        left[j,p]=a%10; right[j,p]=b%10; a//=10; b//=10
check(left,right,'explicit')

# Input-dependence of computed attention scores.
with torch.no_grad():
    x1=model.digit(left[:1])+model.digit(right[:1])+model.position
    x2=model.digit(left[-1:])+model.digit(right[-1:])+model.position
    q1,k1,_=model.qkv(model.norm1(x1)).chunk(3,-1)
    q2,k2,_=model.qkv(model.norm1(x2)).chunk(3,-1)
    print('qk difference', float(((q1@k1.transpose(-2,-1))-(q2@k2.transpose(-2,-1))).abs().max()))
