"""Exact gauge conversion used after train.py produced width2b.pt."""
from pathlib import Path
import torch

state = torch.load('/workspace/width2b.pt', map_location='cpu', weights_only=True)
a, b = state['pos_a'], state['pos_b']
best = max(
    ((abs(float(torch.linalg.det(a[[i, j]]))) / (float(torch.linalg.norm(a[[i, j]])) ** 2 + 1e-30), i, j)
     for i in range(25) for j in range(i + 1, 25)),
)
_, i, j = best
change = a[[i, j]]
a = a @ torch.linalg.inv(change)
state['pos_a'] = a[[r for r in range(25) if r not in (i, j)]]
state['pos_b'] = change @ b
print(f'Fixed positional rows {i},{j}; maximum product error:',
      float((a @ state['pos_b'] - torch.load('/workspace/width2b.pt', map_location='cpu', weights_only=True)['pos_a'] @ b).abs().max()))
print('The transformed tensors were exported as source literals in submission.py.')
