import sys,torch,pprint
sys.path.insert(0,'/workspace');import submission
sd=torch.load('/workspace/model.pt',weights_only=True)
weights={k:v.flatten().tolist() for k,v in sd.items()}
path='/workspace/submission.py';s=open(path).read()
s=s.replace("def build_model():\n    model = AdditionTransformer()\n    return model, {'architecture': 'aligned autoregressive transformer', 'parameters': 1487}\n", "WEIGHTS = "+repr(weights)+"\n\ndef build_model():\n    model = AdditionTransformer()\n    with torch.no_grad():\n        for name, parameter in model.named_parameters():\n            parameter.copy_(torch.tensor(WEIGHTS[name]).view_as(parameter))\n    return model, {'architecture': 'aligned autoregressive transformer', 'parameters': 1487}\n")
open(path,'w').write(s)
