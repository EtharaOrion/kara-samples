import ast, random, sys, time, torch
sys.path.insert(0,'/workspace'); import submission
model,meta=submission.build_model(); assert isinstance(model,torch.nn.Module); assert sum(p.numel() for p in model.parameters())==4826
rng=random.Random(246813579); pairs=[(rng.randrange(10**14),rng.randrange(10**14)) for _ in range(2000)]
edges=[(0,0),(0,10**14-1),(10**14-1,0),(10**14-1,1),(1,10**14-1),(10**14-1,10**14-1),(55555555555555,44444444444445),(99999999999990,9)]
t=time.time(); correct=sum(submission.add(model,a,b)==a+b for a,b in pairs); edge=sum(submission.add(model,a,b)==a+b for a,b in edges)
# Explicitly inspect first-block content-dependent QK scores.
b=model.blocks[0]; x1=model.digit(torch.tensor([[1,2,3,4,5]]))+model.role(torch.arange(5).remainder(3))[None]; x2=model.digit(torch.tensor([[9,0,8,7,6]]))+model.role(torch.arange(5).remainder(3))[None]
def scores(x):
 q,k,_=b.qkv(b.ln1(x)).chunk(3,-1); q=q.view(1,5,2,8).transpose(1,2); k=k.view(1,5,2,8).transpose(1,2); return q@k.transpose(-2,-1)
delta=(scores(x1)-scores(x2)).abs().max().item()
source=open('/workspace/submission.py').read(); tree=ast.parse(source); imports=[n.names[0].name for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
print({'random_accuracy':correct/len(pairs),'correct':correct,'total':len(pairs),'edge_accuracy':edge/len(edges),'seconds':time.time()-t,'per_call_ms':1000*(time.time()-t)/(len(pairs)+len(edges)),'qk_max_delta':delta,'params':sum(p.numel() for p in model.parameters()),'imports':imports})
