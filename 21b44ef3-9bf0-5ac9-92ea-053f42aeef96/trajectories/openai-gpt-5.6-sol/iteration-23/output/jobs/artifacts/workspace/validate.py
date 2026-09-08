import torch
from submission import AdditionTransformer
from train import digits, LOW, HIGH, make_batch

model=AdditionTransformer().cuda()
model.load_state_dict(torch.load('/workspace/trained.pt',weights_only=True)['model'])
model.eval()

@torch.no_grad()
def test_pairs(a,b,chunk=32768):
    total=correct=0; failures=[]
    for start in range(0,len(a),chunk):
        aa=a[start:start+chunk].cuda(); bb=b[start:start+chunk].cuda(); n=len(aa)
        da,db=digits(aa,8),digits(bb,8)
        x=torch.empty(n,26,dtype=torch.long,device='cuda')
        x[:,:16:2],x[:,1:16:2]=da,db; x[:,16]=10
        for j in range(9): x[:,17+j]=model(x[:,:17+j])[:,-1].argmax(-1)
        target=digits(aa+bb,9)
        ok=(x[:,17:26] == target).all(1)
        correct+=ok.sum().item(); total+=n
        bad=(~ok).nonzero().flatten()[:10]
        for z in bad:
            failures.append((int(aa[z]),int(bb[z]),x[z,17:26].tolist(),target[z].tolist()))
    return correct,total,failures[:20]

torch.manual_seed(9917)
a=torch.randint(LOW,HIGH+1,(500000,),device='cpu'); b=torch.randint(LOW,HIGH+1,(500000,),device='cpu')
print('random',test_pairs(a,b)[:2],test_pairs(a[:1000],b[:1000])[2])
vals=set([LOW,HIGH,10000001,10000009,10000010,10000099,10000100,10000999,10001000,10009999,10010000,10099999,10100000,10999999,11000000,19999999,20000000,40000009,50000000,89999999,90000000,98999999,99000000,99000001,99899999,99900000,99900001,99989999,99990000,99990001,99999000,99999001,99999900,99999901,99999990,99999998])
for lead in range(1,10):
 for run in range(1,8):
  p=10**run
  for tail in [0,1,9,p-1,max(0,p-9)]:
   for base in [lead*10_000_000, (lead+1)*10_000_000-p]:
    v=base+tail
    if LOW<=v<=HIGH: vals.add(v)
v=torch.tensor(sorted(vals)); aa=v.repeat_interleave(len(v)); bb=v.repeat(len(v))
r=test_pairs(aa,bb,8192); print('cartesian',r[:2], 'fails',r[2])
# Structured generator, autoregressive.
aall=[]; ball=[]
for _ in range(20):
 x,y=make_batch(25000,0.9)
 # decode original input token pairs into integers only for validation setup
 p=(10**torch.arange(8,device='cuda')).view(1,-1)
 aall.append((x[:,:16:2]*p).sum(1).cpu()); ball.append((x[:,1:16:2]*p).sum(1).cpu())
a=torch.cat(aall);b=torch.cat(ball); r=test_pairs(a,b); print('structured',r[:2],r[2])
