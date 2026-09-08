import sys,random,copy,torch
sys.path.insert(0,'/workspace')
import submission
m,meta=submission.build_model()
print('PARAMS',sum(p.numel() for p in m.parameters()),'META',meta)
rng=random.Random(9922)
pairs=[(rng.randint(10_000_000,99_999_999),rng.randint(10_000_000,99_999_999)) for _ in range(3000)]
edge=[(10_000_000,10_000_000),(99_999_999,99_999_999),(40_000_009,89_999_999),(10_000_999,30_000_099),(99_000_001,99_899_999),(50_000_000,50_000_000)]
pairs+=edge
good=sum(submission.add(m,a,b)==a+b for a,b in pairs)
print('CPU_API',good,len(pairs))
base=[submission.add(m,a,b) for a,b in pairs[:40]]
abl=copy.deepcopy(m)
with torch.no_grad():
 for layer in abl.outputs: layer.weight.zero_()
changed=sum(submission.add(abl,a,b)!=x for (a,b),x in zip(pairs[:40],base))
corrupt=copy.deepcopy(m)
with torch.no_grad(): corrupt.head.weight.zero_()
headchanged=sum(submission.add(corrupt,a,b)!=x for (a,b),x in zip(pairs[:40],base))
print('ATTN_ABLATION_CHANGED',changed,'HEAD_CORRUPTION_CHANGED',headchanged)
# Directly verify attention distributions vary with input at the first generated position.
def weights(model,a,b):
 seq=[]
 for x,y in zip(str(a)[::-1],str(b)[::-1]):seq.extend((ord(x)-48,ord(y)-48))
 seq.append(10);t=torch.tensor(seq).unsqueeze(0);L=t.shape[1]
 x=model.token(t)+(model.pos_left[:L]@model.pos_right);z=model.norm_attn(x)
 q=model.queries[0](z).view(1,L,4,5).transpose(1,2);k=model.keys(z)
 s=torch.einsum('bhtd,bsd->bhts',q,k)*(5**-.5);s=s.masked_fill(torch.ones(L,L,dtype=torch.bool).triu(1),float('-inf'))
 return s.softmax(-1)
w1=weights(m,*pairs[0]);w2=weights(m,*pairs[1]);print('ATTENTION_INPUT_DELTA',float((w1-w2).abs().max()))
