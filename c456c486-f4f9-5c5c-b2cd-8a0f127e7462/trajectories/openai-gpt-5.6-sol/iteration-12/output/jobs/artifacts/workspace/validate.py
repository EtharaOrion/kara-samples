import torch
from train import AdditionTransformer, batch_random, batch_structured, targets


def check(model, generator, total, batch=16384):
    failures = 0
    digit_failures = 0
    examples = []
    for _ in range(total // batch):
        a, b, y = generator(batch, "cuda")
        pred = model(a, b).argmax(-1)
        bad = ~(pred == y).all(1)
        failures += bad.sum().item()
        digit_failures += (pred != y).sum().item()
        if bad.any() and len(examples) < 5:
            ix = bad.nonzero()[0].item()
            examples.append((a[ix].tolist(), b[ix].tolist(), y[ix].tolist(), pred[ix].tolist()))
    return failures, digit_failures, examples


def deterministic_edges(device):
    pairs = []
    maxv = 10**14 - 1
    seeds = [0, 1, 2, 5, 8, 9, 10, 11, 19, 90, 99, maxv, maxv-1, 10**13, 10**13-1]
    pairs += [(a,b) for a in seeds for b in seeds]
    for start in range(14):
        for length in range(1, 15-start):
            p = 10**start
            chain = (10**length - 1) * p
            pairs += [(chain, p), (p, chain), (maxv-chain, chain), (chain, maxv-chain)]
    for d1 in range(10):
        for d2 in range(10):
            pairs.append((int(str(d1)*14), int(str(d2)*14)))
    patterns = ["90", "09", "99", "00", "19", "91", "89", "98", "1234567", "9876543", "909", "099"]
    for x in patterns:
        a = int((x * 14)[:14])
        for y in patterns:
            b = int((y * 14)[:14])
            pairs += [(a,b),(b,a)]
    left=[]; right=[]
    for av,bv in pairs:
        left.append([int(c) for c in f"{av:014d}"[::-1]]+[0])
        right.append([int(c) for c in f"{bv:014d}"[::-1]]+[0])
    a=torch.tensor(left,device=device); b=torch.tensor(right,device=device)
    return a,b,targets(a,b),pairs


model=AdditionTransformer().cuda().eval()
model.load_state_dict(torch.load('/workspace/model.pt',weights_only=True))
with torch.no_grad():
    print('random',check(model,batch_random,1048576))
    print('structured',check(model,batch_structured,524288))
    a,b,y,pairs=deterministic_edges('cuda')
    pred=model(a,b).argmax(-1)
    bad=~(pred==y).all(1)
    print('deterministic',len(pairs),bad.sum().item())
    for i in bad.nonzero()[:20]:
        j=i.item(); print(pairs[j],y[j].tolist(),pred[j].tolist())
    # Demonstrate content-sensitive attention scores.
    x1=model.digit(a[:2])+model.digit(b[:2])+model.position
    q1,k1,_=model.qkv(model.norm1(x1)).chunk(3,-1)
    print('qk input delta',(q1[0]@k1[0].T-q1[1]@k1[1].T).abs().max().item())
