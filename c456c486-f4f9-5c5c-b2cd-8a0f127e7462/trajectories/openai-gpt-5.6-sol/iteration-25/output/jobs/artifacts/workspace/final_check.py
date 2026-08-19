import ast, importlib.util, random, sys, time, torch
path='/workspace/submission.py'
ast.parse(open(path).read())
spec=importlib.util.spec_from_file_location('graded_submission',path)
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
model,metadata=module.build_model()
print('type',isinstance(model,torch.nn.Module),'parameters',sum(p.numel() for p in model.parameters()),metadata)
cases=[(0,0),(1,1),(99999999999999,1),(99999999999999,99999999999999),(50000000000000,50000000000000),(12345678901234,87654321098765)]
random.seed(25)
cases += [(random.randrange(10**14),random.randrange(10**14)) for _ in range(100)]
t=time.time(); errors=[]
for a,b in cases:
    got=module.add(model,a,b)
    if got != a+b: errors.append((a,b,got,a+b))
print('cpu_errors',errors[:5],len(errors),'calls',len(cases),'seconds',time.time()-t)
# Verify outputs depend on trained parameters by replacing the output head.
reference=module.add(model,12345678901234,87654321098765)
with torch.no_grad(): model.head.weight.zero_(); model.head.bias.zero_()
changed=module.add(model,12345678901234,87654321098765)
print('parameter_perturbation',reference,changed,reference!=changed)
