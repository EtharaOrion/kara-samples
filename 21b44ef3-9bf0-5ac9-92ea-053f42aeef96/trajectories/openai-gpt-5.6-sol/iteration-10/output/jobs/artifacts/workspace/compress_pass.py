from pathlib import Path
import torch
p = Path('/workspace')
s = torch.load(p / 'rank7_working.pt', map_location='cpu', weights_only=True)
u, v, vh = torch.linalg.svd(s.pop('pass_embedding'), full_matrices=False)
s['pass_left'] = u[:, :1] * v[:1].sqrt()
s['pass_right'] = v[:1].sqrt()[:, None] * vh[:1]
torch.save(s, p / 'checkpoint.pt')
source = (p / 'submission.py').read_text()
source = source.replace('self.pass_embedding = nn.Parameter(torch.empty(2, d))', 'self.pass_left = nn.Parameter(torch.empty(2, 1))\n        self.pass_right = nn.Parameter(torch.empty(1, d))')
source = source.replace('nn.init.normal_(self.pass_embedding, std=0.02)', 'nn.init.normal_(self.pass_left, std=0.02)\n        nn.init.normal_(self.pass_right, std=0.02)')
source = source.replace('x = x + self.pass_embedding[pass_index]', 'x = x + (self.pass_left @ self.pass_right)[pass_index]')
source = source.replace('rank7-recurrent-adder', 'rank7-passrank1-adder')
# Strip the embedded state before the next exporter regenerates it.
a = source.index('_STATE = '); b = source.index('\n\n\ndef build_model', a)
source = source[:a] + '_STATE = None' + source[b:]
(p / 'submission.py').write_text(source)
