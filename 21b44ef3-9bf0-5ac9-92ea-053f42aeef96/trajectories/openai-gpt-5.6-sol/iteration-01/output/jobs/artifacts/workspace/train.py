import json
import math
import sys

sys.path = [p for p in sys.path if "openhands-venv" not in p]
sys.path.extend(["/usr/local/lib/python3.11/dist-packages", "/usr/lib/python3/dist-packages"])

import torch
from torch import nn
import torch.nn.functional as F


class ColumnTransformer(nn.Module):
    def __init__(self, width=8, hidden=32):
        super().__init__()
        self.embedding = nn.Embedding(12, width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.projection = nn.Linear(width, width, bias=False)
        self.norm1 = nn.LayerNorm(width)
        self.ff1 = nn.Linear(width, hidden)
        self.ff2 = nn.Linear(hidden, width)
        self.norm2 = nn.LayerNorm(width)
        self.output = nn.Linear(width, 20)

    def forward(self, tokens):
        x = self.embedding(tokens)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        attention = F.softmax(q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1]), dim=-1)
        x = self.norm1(x + self.projection(attention @ v))
        x = self.norm2(x + self.ff2(F.relu(self.ff1(x))))
        return self.output(x[:, 2])


def training_set(device):
    rows, labels = [], []
    for carry in range(2):
        for a in range(10):
            for b in range(10):
                rows.append((a, b, 10 + carry))
                labels.append(a + b + carry)
    return torch.tensor(rows, device=device), torch.tensor(labels, device=device)


def train(width=8, hidden=32, seed=1, steps=20000):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    model = ColumnTransformer(width, hidden).to(device)
    x, y = training_set(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        if step % 500 == 0 or step == steps - 1:
            accuracy = (logits.argmax(1) == y).float().mean().item()
            print(step, f"loss={loss.item():.6f}", f"accuracy={accuracy:.4f}")
        if step >= 1000 and (model(x).argmax(1) == y).all():
            print("perfect at", step)
            break
    model.eval()
    torch.save(model.cpu().state_dict(), "/workspace/model.pt")
    print("parameters", sum(p.numel() for p in model.parameters()))


if __name__ == "__main__":
    train(*(map(int, sys.argv[1:]) if len(sys.argv) > 1 else ()))
