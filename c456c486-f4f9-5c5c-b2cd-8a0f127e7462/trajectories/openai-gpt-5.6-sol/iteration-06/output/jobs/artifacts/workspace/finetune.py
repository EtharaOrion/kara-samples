import sys
import time
import torch
import torch.nn.functional as F
from train import Adder, batch, exact_accuracy

kind = sys.argv[1]
path = sys.argv[2]
steps = int(sys.argv[3])
lr = float(sys.argv[4])
batch_size = int(sys.argv[5]) if len(sys.argv) > 5 else 1024
model = Adder(kind == 'shared')
model.load_state_dict(torch.load(path, weights_only=True))
torch.set_num_threads(16)
opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.001)
start = time.time()
for step in range(1, steps + 1):
    tok = batch(batch_size)
    logits = model(tok[:, :-1])
    positions = torch.arange(1, 44, 3)
    loss = F.cross_entropy(logits[:, positions].reshape(-1, 10), tok[:, positions + 1].reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 250 == 0:
        print(step, loss.item(), exact_accuracy(model, 2000, 910000+step), time.time()-start, flush=True)
        torch.save(model.state_dict(), path)
print('final', exact_accuracy(model, 20000, 314159), flush=True)
torch.save(model.state_dict(), path)
