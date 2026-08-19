"""Train the 17-parameter digit-transition transformer used by submission.py."""
import torch
from torch import nn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STATES = torch.tensor(
    [(a, b, carry) for a in range(10) for b in range(10) for carry in range(2)],
    device=DEVICE,
)
TARGETS = STATES.sum(dim=1).float()


class ScalarAdder(nn.Module):
    def __init__(self):
        super().__init__()
        self.digit = nn.Embedding(10, 1)
        self.cls = nn.Parameter(torch.randn(1))
        self.carry_role = nn.Parameter(torch.randn(1))
        self.qkv = nn.Linear(1, 3, bias=False)
        self.out = nn.Linear(1, 1)

    def forward(self, triples):
        tokens = self.digit(triples)
        carry = tokens[:, 2:3] + self.carry_role
        tokens = torch.cat((self.cls.expand(len(triples), 1, 1), tokens[:, :2], carry), dim=1)
        query, key, value = self.qkv(tokens).chunk(3, dim=-1)
        attention = (query[:, :1] @ key.transpose(1, 2)).softmax(dim=-1)
        return self.out((attention @ value)[:, 0])[:, 0]


def train(seed):
    torch.manual_seed(seed)
    model = ScalarAdder().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 30_000, eta_min=1e-4)
    for step in range(30_000):
        prediction = model(STATES)
        loss = ((prediction - TARGETS) ** 2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % 100 == 0 and (prediction - TARGETS).abs().max() < 0.15:
            break
    with torch.no_grad():
        error = (model(STATES) - TARGETS).abs().max().item()
        accuracy = model(STATES).round().eq(TARGETS).float().mean().item()
    return model, accuracy, error, step + 1


if __name__ == "__main__":
    for seed in range(30):
        model, accuracy, error, steps = train(seed)
        print(seed, accuracy, error, steps, flush=True)
        if accuracy == 1.0 and error < 0.2:
            state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
            torch.save({"seed": seed, "state_dict": state}, "/workspace/model.pt")
            break
