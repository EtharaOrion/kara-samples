import torch
p='/workspace/submission.py'
state=torch.load('/workspace/model.pt',map_location='cpu',weights_only=True)
state={k:v for k,v in state.items() if k!='mask'}
lines=['_WEIGHTS = {']
for k,v in state.items():
    vals=','.join(format(float(x),'.9g') for x in v.flatten())
    lines.append(f"    {k!r}: ({list(v.shape)!r}, [{vals}]),")
lines.append('}\n')
text=open(p).read()
marker='\n_WEIGHTS = {'
pos=text.find(marker)
if pos < 0:
    pos=text.index('\ndef build_model():')
head=text[:pos]
tail='''
def build_model():
    model = AdditionTransformer()
    state = {name: torch.tensor(values).reshape(shape) for name, (shape, values) in _WEIGHTS.items()}
    model.load_state_dict(state, strict=False)
    return model.eval(), {"architecture": "recurrent parallel causal transformer", "rounds": 6}


def add(model, a: int, b: int) -> int:
    left = torch.tensor([[int(c) for c in f"{a:014d}"[::-1]] + [0]], dtype=torch.long)
    right = torch.tensor([[int(c) for c in f"{b:014d}"[::-1]] + [0]], dtype=torch.long)
    with torch.no_grad():
        digits = model(left, right).argmax(dim=-1)[0].tolist()
    return int("".join(str(d) for d in digits[::-1]))
'''
open(p,'w').write(head+'\n'+'\n'.join(lines)+tail)
