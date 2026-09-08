from pathlib import Path
import torch
p = Path('/workspace')
s = torch.load(p / 'rank8_working.pt', map_location='cpu', weights_only=True)
pos = s['pos_left'] @ s['pos_right']
u, v, vh = torch.linalg.svd(pos, full_matrices=False)
s['pos_left'] = u[:, :7] * v[:7].sqrt()
s['pos_right'] = v[:7].sqrt()[:, None] * vh[:7]
torch.save(s, p / 'checkpoint.pt')
source = (p / 'submission.py').read_text()
source = source.replace('torch.empty(25, 8)', 'torch.empty(25, 7)').replace('torch.empty(8, d)', 'torch.empty(7, d)').replace('rank8-recurrent-adder', 'rank7-recurrent-adder')
(p / 'submission.py').write_text(source)
print(s['pos_left'].shape, s['pos_right'].shape)
