import ast, sys, random
sys.path.insert(0,'/workspace')
import torch
import submission
from train import sums_from_digits

torch.manual_seed(141414)
model,_=submission.build_model(); model.cuda().eval()
errors=seen=0
first=None
with torch.no_grad():
 for _ in range(128):
  n=8192; a=torch.randint(10,(n,15),device='cuda'); b=torch.randint(10,(n,15),device='cuda'); a[:,-1]=b[:,-1]=0
  start=torch.randint(0,14,(n,1),device='cuda'); max_len=14-start
  long=torch.rand((n,1),device='cuda') < .7
  length=torch.where(long,torch.clamp(max_len-torch.randint(0,3,(n,1),device='cuda'),min=1),1+(torch.rand((n,1),device='cuda')*max_len).long())
  pos=torch.arange(14,device='cuda').view(1,-1); continuation=(pos>start)&(pos<start+length)
  da=torch.randint(10,(n,14),device='cuda'); a[:,:14]=torch.where(continuation,da,a[:,:14]); b[:,:14]=torch.where(continuation,9-da,b[:,:14])
  rows=torch.arange(n,device='cuda'); s=start[:,0]; av=torch.randint(1,10,(n,),device='cuda'); extra=torch.minimum(torch.randint(0,9,(n,),device='cuda'),av-1)
  a[rows,s]=av; b[rows,s]=10-av+extra
  swap=torch.rand(n,device='cuda')<.5; temp=a[swap].clone(); a[swap]=b[swap]; b[swap]=temp
  y=sums_from_digits(a,b); pred=model(a,b).argmax(-1); wrong=(pred!=y).any(1); errors+=int(wrong.sum()); seen+=n
  if first is None and wrong.any():
   i=int(wrong.nonzero()[0]); first=(a[i].tolist(),b[i].tolist(),y[i].tolist(),pred[i].tolist())
print('explicit carry',errors,'/',seen,'first',first)

# Export fidelity and registration.
checkpoint=torch.load('/workspace/final.pt',map_location='cpu',weights_only=True)
cpu,_=submission.build_model(); state=cpu.state_dict()
print('state keys equal',state.keys()==checkpoint.keys())
print('max export delta',max(float((state[k]-checkpoint[k]).abs().max()) for k in state))
print('registered parameters',sum(p.numel() for p in cpu.parameters()),'floating buffers',[(n,b.shape) for n,b in cpu.named_buffers() if b.is_floating_point()])

source=open('/workspace/submission.py').read(); tree=ast.parse(source)
imports=[]
for node in ast.walk(tree):
 if isinstance(node,ast.Import): imports += [x.name for x in node.names]
 if isinstance(node,ast.ImportFrom): imports.append(node.module)
print('imports',imports,'syntax ok',compile(tree,'submission.py','exec') is not None)

random.seed(1400); failures=[]
for _ in range(200):
 left=random.randrange(10**14); right=random.randrange(10**14); got=submission.add(cpu,left,right)
 if got!=left+right: failures.append((left,right,got))
print('CPU add random',len(failures),'/200 failures',failures[:3])
