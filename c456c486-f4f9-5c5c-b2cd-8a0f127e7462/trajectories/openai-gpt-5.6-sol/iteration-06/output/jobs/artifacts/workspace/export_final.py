from pathlib import Path
import torch
s=torch.load('/workspace/unshared.pt',weights_only=True)
vals={k:v.flatten().tolist() for k,v in s.items()}
shapes={k:list(v.shape) for k,v in s.items()}
t=Path('/workspace/final_template.py').read_text()
Path('/workspace/submission.py').write_text(t.replace('__VALUES__',repr(vals)).replace('__SHAPES__',repr(shapes)))
