import torch
path='/workspace/submission.py'
state=torch.load('/workspace/trained.pt',map_location='cpu',weights_only=True)['model']
parts=[]
for name,tensor in state.items():
    parts.append(repr(name)+': torch.tensor('+repr(tensor.tolist())+', dtype=torch.float32)')
blob='_TRAINED_STATE = {\n    '+',\n    '.join(parts)+'\n}'
text=open(path).read()
start=text.index('_TRAINED_STATE =')
end=text.index('\n\n\ndef build_model',start)
text=text[:start]+blob+text[end:]
open(path,'w').write(text)
