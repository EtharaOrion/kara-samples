import ast, copy, random, sys, time
sys.path.extend(['/usr/local/lib/python3.11/dist-packages','/usr/local/lib/python3.11/site-packages'])
import torch, submission
src=open('/workspace/submission.py').read(); tree=ast.parse(src)
print('imports',[ast.unparse(n) for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))])
print('file_bytes',len(src),'weight_literals',src.count('_WEIGHTS ='))
m,meta=submission.build_model(); m.eval()
print('params',sum(p.numel() for p in m.parameters()),'training',m.training,meta)
cases=[(0,0),(1,9),(99999999999999,1),(99999999999999,99999999999999),(5,10000000000005),(12345678901234,87654321098765),(55555555555555,44444444444445)]
for a,b in cases:
 y=submission.add(m,a,b); print(a,b,y,y==a+b)
# End-to-end CPU random smoke.
r=random.Random(271828); bad=[]; t=time.time()
for _ in range(1000):
 a=r.randrange(100_000_000_000_000); b=r.randrange(100_000_000_000_000); y=submission.add(m,a,b)
 if y!=a+b: bad.append((a,b,y));
print('cpu_random_errors',len(bad),'seconds',time.time()-t,'examples',bad[:3])
# Replacing trained parameters must materially break outputs.
broken=copy.deepcopy(m)
with torch.no_grad():
 for p in broken.parameters(): p.zero_()
changed=sum(submission.add(broken,a,b)!=submission.add(m,a,b) for a,b in cases)
print('zero_weight_outputs_changed',changed,'of',len(cases))
