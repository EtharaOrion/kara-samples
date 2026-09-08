from pathlib import Path
import torch

path=Path('/workspace/submission.py')
text=path.read_text()
state=torch.load('/workspace/final_best.pt',map_location='cpu',weights_only=True)
def literal(t):
    return 'torch.tensor(' + repr(t.tolist()) + ', dtype=torch.float32)'
body='_TRAINED_STATE = {\n' + ''.join(f'    {name!r}: {literal(value)},\n' for name,value in state.items()) + '}\n'
start=text.index('_TRAINED_STATE =')
end=text.index('\n\ndef build_model', start)
path.write_text(text[:start]+body.rstrip()+text[end:])
print(path.stat().st_size, len(state), sum(v.numel() for v in state.values()))
