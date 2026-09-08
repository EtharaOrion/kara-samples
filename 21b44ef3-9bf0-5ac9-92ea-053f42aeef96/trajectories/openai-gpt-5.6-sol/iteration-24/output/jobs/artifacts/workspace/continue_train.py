import importlib.util
from pathlib import Path
import torch

ROOT = Path('/workspace')
spec = importlib.util.spec_from_file_location('trainer', ROOT / 'train.py')
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
spec2 = importlib.util.spec_from_file_location('sub', ROOT / 'submission.py')
s = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(s)
torch.set_float32_matmul_precision('high')
model = s.AdditionTransformer(2).cuda()
model.load_state_dict(torch.load(ROOT / 'final.pt', weights_only=True))
t.train_phase(model, 18000, 1e-5, .35, 'width2_general')
t.train_phase(model, 8000, 3e-6, .50, 'width2_finish')
ar, mr = t.accuracy(model, 250, 0.0)
ae, me = t.accuracy(model, 250, .85)
print(f'CONTINUED random={ar:.8f}/{mr:.4f} structured={ae:.8f}/{me:.4f}', flush=True)
torch.save(model.state_dict(), ROOT / 'final2.pt')
t.export(model)
