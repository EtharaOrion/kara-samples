import sys, time
sys.path.insert(0, '/workspace')
import torch
import submission
from train import random_batch, mixed_batch, sums_from_digits

torch.manual_seed(140014)
torch.set_float32_matmul_precision('high')
device = torch.device('cuda')
model, metadata = submission.build_model()
model = model.to(device).eval()
print('parameters', sum(p.numel() for p in model.parameters()), metadata)

def evaluate(generator, total, batch=8192):
    errors = digit_errors = seen = 0
    first = None
    with torch.no_grad():
        while seen < total:
            n = min(batch, total-seen)
            a,b,y = generator(n, device)
            pred = model(a,b).argmax(-1)
            wrong = pred != y
            errors += int(wrong.any(1).sum())
            digit_errors += int(wrong.sum())
            if first is None and wrong.any():
                i = int(wrong.any(1).nonzero()[0])
                first = (a[i].tolist(),b[i].tolist(),y[i].tolist(),pred[i].tolist())
            seen += n
    return errors, digit_errors, first

start=time.time()
print('uniform', evaluate(random_batch, 1048576), 'seconds', time.time()-start, flush=True)
start=time.time()
structured=lambda n,d: mixed_batch(n,d,1.0,0.8)
print('structured', evaluate(structured, 524288), 'seconds', time.time()-start, flush=True)

# Systematically combine carry start/length, surrounding digits, both orders,
# all repeated pairs, near-max patterns, and decimal boundaries.
cases={(0,0),(10**14-1,1),(1,10**14-1),(10**14-1,10**14-1)}
for start_pos in range(14):
 for length in range(1,15-start_pos):
  run=(10**length-1)*10**start_pos; inc=10**start_pos
  for prefix_digit in [0,1,5,7,9]:
   prefix=prefix_digit*10**(start_pos+length) if start_pos+length<14 else 0
   for suffix in [0, (10**start_pos-1) if start_pos else 0]:
    x=prefix+run+suffix
    if x<10**14:
     cases.add((x,inc)); cases.add((inc,x))
for d in range(10):
 for e in range(10):
  cases.add((int(str(d)*14),int(str(e)*14)))
for power in range(15):
 boundary=10**power
 for delta in range(-10,11):
  x=boundary+delta
  if 0<=x<10**14:
   for y in [0,1,9,10**14-1-x]:
    if 0<=y<10**14: cases.add((x,y)); cases.add((y,x))
pairs=sorted(cases)
a=torch.tensor([[int(c) for c in f'{x:015d}'[::-1]] for x,y in pairs],device=device)
b=torch.tensor([[int(c) for c in f'{y:015d}'[::-1]] for x,y in pairs],device=device)
y=sums_from_digits(a,b)
with torch.no_grad(): pred=model(a,b).argmax(-1)
wrong=(pred!=y).any(1)
print('systematic edge',int(wrong.sum()),'/',len(pairs))
if wrong.any():
 for i in wrong.nonzero()[:20]: print('FAIL',pairs[int(i)])

# Input dependence in the first-round attention logits.
with torch.no_grad():
 def qk(left,right):
  x=model.digit(left)+model.digit(right)+model.position
  z=model.norm1(x)
  qkv=model.qkv(z).view(-1,15,3,2,6).permute(2,0,3,1,4)
  return qkv[0]@qkv[1].transpose(-2,-1)
 x1=torch.zeros(1,15,dtype=torch.long,device=device)
 x2=torch.arange(15,device=device).remainder(10).view(1,15)
 qk_delta=float((qk(x1,x1)-qk(x2,9-x2)).abs().max())
 print('qk max input delta',qk_delta)

# CPU tests use only the final exported submission interface.
cpu_model,_=submission.build_model(); cpu_model.eval()
examples=[(0,0),(2,3),(99999999999999,1),(99999999999999,99999999999999),(12345678901234,87654321098765),(50000000000000,50000000000000)]
for left,right in examples:
 got=submission.add(cpu_model,left,right)
 print('add',left,right,got,'ok',got==left+right)
