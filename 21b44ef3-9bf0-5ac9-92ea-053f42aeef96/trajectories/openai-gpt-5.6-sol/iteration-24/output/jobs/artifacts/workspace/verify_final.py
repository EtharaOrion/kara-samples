import importlib.util
import random
from pathlib import Path
import torch

spec=importlib.util.spec_from_file_location('submission_final',Path('/workspace/submission.py'))
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
model,meta=s.build_model(); model.cuda().eval()

@torch.no_grad()
def decode(model,a,b):
 n=len(a); div=torch.tensor([10**i for i in range(8)],device=a.device)
 ad=(a[:,None]//div)%10;bd=(b[:,None]//div)%10
 x=torch.empty((n,17),dtype=torch.long,device=a.device)
 x[:,0:16:2]=ad;x[:,1:16:2]=bd;x[:,16]=10
 out=torch.zeros_like(a);place=1
 for _ in range(9):
  d=model(x)[:,-1].argmax(-1);out+=d*place;place*=10;x=torch.cat((x,d[:,None]),1)
 return out

def test_pairs(pairs,label):
 good=0; failures=[]
 for start in range(0,len(pairs),8192):
  q=pairs[start:start+8192]
  a=torch.tensor([x for x,_ in q],device='cuda');b=torch.tensor([y for _,y in q],device='cuda')
  out=decode(model,a,b);ok=out==a+b;good+=int(ok.sum())
  for i in (~ok).nonzero()[:5]:
   j=int(i);failures.append((int(a[j]),int(b[j]),int(out[j])))
 print(label,good,len(pairs),good/len(pairs),failures[:10])

# One million uniform unseen pairs.
g=torch.Generator().manual_seed(240099)
pairs=[]
for _ in range(1_000_000): pairs.append((random.randrange(10_000_000,100_000_000),random.randrange(10_000_000,100_000_000)))
test_pairs(pairs,'random')

# Broad decimal boundaries, asymmetric 0/9 runs, sparse and repeated digits.
values={10_000_000,99_999_999,11_111_111,22_222_222,55_555_555,88_888_888,89_999_999,99_000_001,99_899_999}
for p in range(1,8):
 u=10**p
 for base in (10_000_000,20_000_000,40_000_000,50_000_000,80_000_000,90_000_000,100_000_000):
  for d in (-u-1,-u,-u+1,-99,-10,-9,-1,0,1,9,10,99,u-1,u,u+1):
   if 10_000_000<=base+d<100_000_000:values.add(base+d)
values=sorted(values)
test_pairs([(a,b) for a in values for b in values],'boundaries')

# Exact public API on CPU.
cpu,_=s.build_model(); rng=random.Random(240100); cases=[(rng.randrange(10_000_000,100_000_000),rng.randrange(10_000_000,100_000_000)) for _ in range(3000)]
api=sum(s.add(cpu,a,b)==a+b for a,b in cases)
print('cpu_api',api,len(cases),api/len(cases))

# Attention is input-dependent: explicitly compare computed causal attention matrices.
def weights(m,tokens,layer=0):
 length=tokens.shape[1];x=m.token(tokens)+(m.pos_left[:length]@m.pos_right);z=m.attention_norm(x)
 q=m.queries[layer](z).view(len(tokens),length,4,5).transpose(1,2);k=m.key(z).unsqueeze(1)
 score=(q@k.transpose(-2,-1))/(5**.5);mask=torch.ones(length,length,device=tokens.device).triu(1).bool()
 return score.masked_fill(mask,float('-inf')).softmax(-1)
a=torch.tensor([[1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,10]],device='cuda')
b=torch.tensor([[9,8,7,6,5,4,3,2,1,0,9,8,7,6,5,4,10]],device='cuda')
print('attention_input_delta',float((weights(model,a)-weights(model,b)).abs().max()))

sample_a=torch.tensor([x for x,_ in pairs[:64]],device='cuda');sample_b=torch.tensor([y for _,y in pairs[:64]],device='cuda')
baseline=decode(model,sample_a,sample_b)
original=model.attention
model.attention=lambda x,layer:torch.zeros_like(x)
ablated=decode(model,sample_a,sample_b)
model.attention=original
print('attention_ablation_changed',int((baseline!=ablated).sum()),64)
with torch.no_grad():model.classifier.weight.zero_()
corrupt=decode(model,sample_a,sample_b)
print('output_corruption_changed',int((baseline!=corrupt).sum()),64)
print(meta,'actual_parameters',sum(p.numel() for p in model.parameters()))
