import torch
src=open('/workspace/train.py').read().split('model=Model().cuda()')[0]
exec(src)
m=Model().cpu();m.load_state_dict(torch.load('/workspace/checkpoint.pt',map_location='cpu',weights_only=True));m.eval()
state={k:v.tolist() for k,v in m.state_dict().items()}
classes=src[src.index('class Attn'):src.index('def digits')]
header="\"\"\"Trained simultaneous-output addition transformer.\"\"\"\nimport math\nimport torch\nfrom torch import nn\nimport torch.nn.functional as F\nD=10; H=2; FF=4; N=14; OUT=15; ROUNDS=7\n"
tail="""
def build_model():
    model=Model()
    model.load_state_dict({name:torch.tensor(value) for name,value in _WEIGHTS.items()})
    model.eval()
    return model,{"architecture":"encoder-query cross-attention transformer","parameters":sum(p.numel() for p in model.parameters())}

def add(model,a:int,b:int)->int:
    ad=[]; bd=[]
    for _ in range(14):
        ad.append(a%10); a//=10
        bd.append(b%10); b//=10
    with torch.no_grad():
        pred=model(torch.tensor([ad]),torch.tensor([bd])).argmax(-1)[0]
    value=0
    place=1
    for digit in pred:
        value += int(digit)*place
        place *= 10
    return value
"""
text=header+classes+'\n_WEIGHTS='+repr(state)+tail
open('/workspace/submission.py','w').write(text)
print(len(text))
