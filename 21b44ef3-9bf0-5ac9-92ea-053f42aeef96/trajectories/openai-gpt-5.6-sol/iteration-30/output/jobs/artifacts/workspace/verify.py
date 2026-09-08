import sys, random, torch
sys.path.insert(0,'/workspace')
import submission, train
m,meta=submission.build_model(); print('params',sum(p.numel() for p in m.parameters()),meta,'trained',submission._TRAINED is not None)
m=m.cuda(); print('random',train.eval_model(m,1000000,0)); print('structured',train.eval_model(m,1000000,.75))
# Cartesian boundaries, asymmetric zero/nine suffixes, extrema and repeated digits.
vals={10_000_000,10_000_001,10_000_009,10_000_010,10_000_099,10_000_100,10_000_999,10_001_000,10_009_999,10_010_000,10_099_999,10_100_000,10_999_999,11_111_111,20_000_000,22_222_222,40_000_009,49_999_999,50_000_000,50_000_001,55_555_555,80_000_000,88_888_888,89_999_999,90_000_000,90_000_001,98_999_999,99_000_000,99_000_001,99_899_999,99_900_000,99_990_000,99_999_000,99_999_900,99_999_990,99_999_998,99_999_999}
for p in [10,100,1000,10000,100000,1000000,10000000]:
 for lead in range(1,10):
  for off in [-p,-p+1,-1,0,1,p-1]:
   v=lead*10_000_000+off
   if 10_000_000<=v<=99_999_999: vals.add(v)
vals=sorted(vals); pairs=[(a,b) for a in vals for b in vals]
good=0
with torch.no_grad():
 for st in range(0,len(pairs),10000):
  q=pairs[st:st+10000]; a=torch.tensor([x for x,y in q],device='cuda'); b=torch.tensor([y for x,y in q],device='cuda')
  seq=train.encode(a,b)[0][:,:17]
  for _ in range(9): seq=torch.cat((seq,m(seq)[:,-1].argmax(1)[:,None]),1)
  good+=int(((seq[:,17:]*train.POW10).sum(1)==a+b).sum())
print('edges',good,len(pairs),'values',len(vals))
# API checks on CPU.
m=m.cpu(); random.seed(7); cases=[(random.randint(train.LO,train.HI),random.randint(train.LO,train.HI)) for _ in range(2000)]
base=[submission.add(m,a,b) for a,b in cases]; print('api',sum(x==a+b for x,(a,b) in zip(base,cases)),len(cases))
# Dependence on attention and classifier.
with torch.no_grad():
 for o in m.outputs: o.weight.zero_()
abl=[submission.add(m,a,b) for a,b in cases[:40]]
print('attention_changed',sum(x!=y for x,y in zip(base,abl)),'of',len(abl))
m,_=submission.build_model()
with torch.no_grad(): m.classifier.weight.zero_(); m.classifier.bias.zero_()
out=[submission.add(m,a,b) for a,b in cases[:40]]
print('classifier_changed',sum(x!=y for x,y in zip(base,out)),'of',len(out))
