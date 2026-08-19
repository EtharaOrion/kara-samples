import ast,random,sys,time,torch
sys.path.insert(0,'/workspace'); import submission
m,meta=submission.build_model(); assert sum(p.numel() for p in m.parameters())==4826
r=random.Random(135792468); pairs=[(r.randrange(10**14),r.randrange(10**14)) for _ in range(1000)]; t=time.time(); ok=sum(submission.add(m,a,b)==a+b for a,b in pairs)
edges=[(0,0),(0,99999999999999),(99999999999999,0),(99999999999999,1),(1,99999999999999),(99999999999999,99999999999999),(55555555555555,44444444444445),(99999999999990,9)]; edge=sum(submission.add(m,a,b)==a+b for a,b in edges)
b=m.blocks[0]; pos=torch.arange(5); xs=[m.digit(x)+m.role(pos%3)[None] for x in (torch.tensor([[1,2,3,4,5]]),torch.tensor([[9,0,8,7,6]]))]
def qk(x):
 q,k,_=b.qkv(b.ln1(x)).chunk(3,-1); return q.view(1,5,2,8).transpose(1,2)@k.view(1,5,2,8).transpose(1,2).transpose(-2,-1)
imports=[]
for n in ast.walk(ast.parse(open('/workspace/submission.py').read())):
 if isinstance(n,ast.Import): imports.extend(a.name for a in n.names)
 if isinstance(n,ast.ImportFrom): imports.append(n.module)
print({'accuracy':ok/1000,'edges':edge/len(edges),'params':sum(p.numel() for p in m.parameters()),'ms_per_call':(time.time()-t)*1000/1008,'qk_delta':(qk(xs[0])-qk(xs[1])).abs().max().item(),'imports':imports,'metadata':meta})
