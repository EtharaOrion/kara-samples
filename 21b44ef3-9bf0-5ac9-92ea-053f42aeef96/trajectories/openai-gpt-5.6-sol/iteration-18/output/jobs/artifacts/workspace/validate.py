import random
import torch
from submission import AdderTransformer
from train import autoregressive_accuracy, batch_data

DEVICE = "cuda"
model = AdderTransformer().to(DEVICE)
model.load_state_dict(torch.load("/workspace/checkpoint.pt", weights_only=True))
print("parameters", sum(p.numel() for p in model.parameters()))
print("random", autoregressive_accuracy(model, 500_000, 0.0))
print("structured", autoregressive_accuracy(model, 500_000, 1.0))

pairs = set()
anchors = [10_000_000, 10_000_001, 10_000_009, 10_000_010, 10_000_099, 10_000_100,
           10_000_999, 10_001_000, 10_009_999, 10_010_000, 10_099_999, 10_100_000,
           10_999_999, 11_000_000, 19_999_999, 20_000_000, 40_000_009, 49_999_999,
           50_000_000, 50_000_001, 89_999_999, 90_000_000, 98_999_999, 99_000_000,
           99_900_000, 99_990_000, 99_999_000, 99_999_900, 99_999_990, 99_999_998, 99_999_999]
for a in anchors:
    for b in anchors:
        pairs.add((a, b))
for k in range(1, 8):
    p = 10 ** k
    tails = [0, 1, p // 10 if p >= 10 else 1, p - 2, p - 1]
    for da in range(1, 10):
        for db in range(1, 10):
            for ta in tails:
                for tb in tails:
                    a = da * 10_000_000 + ta
                    b = db * 10_000_000 + tb
                    if a <= 99_999_999 and b <= 99_999_999:
                        pairs.add((a, b))
            target = 100_000_000 + (da - 5) * p
            lo = max(10_000_000, target - 99_999_999)
            hi = min(99_999_999, target - 10_000_000)
            if lo <= hi:
                a = lo + ((da * 13 + db * 7) % (hi - lo + 1))
                pairs.add((a, target - a))
pairs = list(pairs)
correct = 0
failures = []
model.eval()
with torch.no_grad():
    for start in range(0, len(pairs), 2048):
        part = pairs[start:start+2048]
        a = torch.tensor([x for x, _ in part], device=DEVICE)
        b = torch.tensor([y for _, y in part], device=DEVICE)
        powers = torch.tensor([10**i for i in range(8)], device=DEVICE)
        ad, bd = (a[:,None]//powers)%10, (b[:,None]//powers)%10
        seq = torch.cat((torch.stack((ad,bd),2).reshape(-1,16), torch.full((len(part),1),10,device=DEVICE)),1)
        ds=[]
        for _ in range(9):
            d=model(seq)[:,-1].argmax(1); ds.append(d); seq=torch.cat((seq,d[:,None]),1)
        pred=(torch.stack(ds,1)*torch.tensor([10**i for i in range(9)],device=DEVICE)).sum(1)
        good=pred == a+b
        correct += int(good.sum())
        bad=(~good).nonzero().flatten().tolist()
        for i in bad[:20-len(failures)]: failures.append((part[i],int(pred[i])))
print("curated", correct, len(pairs), failures)

x,_=batch_data(64,0.0); seq=x[:,:17]
base=[]
for _ in range(9):
    d=model(seq)[:,-1].argmax(1);base.append(d);seq=torch.cat((seq,d[:,None]),1)
base=torch.stack(base,1)
for attn in model.attn:
    attn.out.weight.zero_()
seq=x[:,:17]; altered=[]
for _ in range(9):
    d=model(seq)[:,-1].argmax(1);altered.append(d);seq=torch.cat((seq,d[:,None]),1)
altered=torch.stack(altered,1)
print("attention_ablation_changed", int((base != altered).any(1).sum()), "/64")
