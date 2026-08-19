import ast, importlib.util, random, sys, time, torch
spec=importlib.util.spec_from_file_location('graded','/workspace/submission.py'); s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
m,meta=s.build_model()
print('params',sum(p.numel() for p in m.parameters()),'registered',len(list(m.parameters())),'meta',meta)
# Compare every exported value to checkpoint model parameter vector.
sys.path.insert(0,'/workspace'); from train import Model
ref=Model();ref.load_state_dict(torch.load('/workspace/final.pt',map_location='cpu',weights_only=True))
v1=torch.nn.utils.parameters_to_vector(m.parameters());v2=torch.nn.utils.parameters_to_vector(ref.parameters())
print('weight_max_diff',(v1-v2).abs().max().item(),'finite',torch.isfinite(v1).all().item())
cases=[(0,0),(1,2),(99999999999999,1),(99999999999999,99999999999999),(12345678901234,87654321098765),(9999994,1)]
for a,b in cases: print(a,b,s.add(m,a,b),a+b)
random.seed(24); t=time.time(); errors=[]
for _ in range(1000):
 a=random.randrange(100_000_000_000_000);b=random.randrange(100_000_000_000_000);g=s.add(m,a,b)
 if g!=a+b:errors.append((a,b,g,a+b))
print('cpu_random_errors',len(errors),errors[:3],'seconds',time.time()-t)
# Parameter-use perturbation: zeroing learned parameters must break predictions.
probe=(12345678901234,87654321098765); good=s.add(m,*probe)
with torch.no_grad():
 for p in m.parameters(): p.zero_()
broken=s.add(m,*probe)
print('perturbation',probe,good,broken,'changed',good!=broken)
# Dependency/static summary.
tree=ast.parse(open('/workspace/submission.py').read());print('imports',[ast.unparse(x) for x in tree.body if isinstance(x,(ast.Import,ast.ImportFrom))])
