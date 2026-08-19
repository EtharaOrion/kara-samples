import torch
from torch import nn


class AdditionTransformer(nn.Module):
    """Placeholder replaced with trained weights by /workspace/train.py."""

    def __init__(self):
        super().__init__()
        self.untrained = nn.Parameter(torch.zeros(1))

    def forward(self, a_digits, b_digits):
        return torch.zeros((*a_digits.shape, 10), device=a_digits.device)


def build_model():
    return AdditionTransformer().eval(), {"status": "training in progress"}


def add(model, a: int, b: int) -> int:
    raise RuntimeError("Training has not completed")
