"""Sanity check ONLY: hand-set weights to confirm arch.forward can express the
carry mechanism, and to validate the token layout / masks.  These weights are a
test fixture; nothing here is trained and nothing here is ever shipped."""
import torch, arch, data

dev = "cuda"
cfg = arch.default_cfg(C=1, U=2)
p = {
    "code": torch.arange(1, 10, device=dev, dtype=torch.float32).reshape(1, 9, 1),
    "Bw":   torch.tensor([[[8.0, 8.0]]], device=dev),
    "bb":   torch.tensor([[-8 * 8.5, -8 * 9.5]], device=dev),
    "kw":   torch.tensor([[-400.0, 400.0]], device=dev),
    "vw":   torch.tensor([[0.0, 1.0]], device=dev),
    "q":    torch.tensor([1.0], device=dev),
    "lam":  torch.tensor([-8.0], device=dev),
    "w1":   torch.tensor([[1.0]], device=dev),
    "w2":   torch.tensor([[-10.0]], device=dev),
    "ls":   torch.tensor([1.0], device=dev),
}
g = torch.Generator(device=dev).manual_seed(7)
for n in (3, 8, 12):
    ok = tot = 0
    for _ in range(20):
        ta, tb, tgt, _, _ = data.batch(4096, n, dev, g, heldout=True)
        pred = arch.forward(p, ta, tb, cfg).argmax(-1)
        ok += int((pred[:, :, 1:] == tgt[None, :, 1:]).all(-1).sum())
        tot += ta.shape[0]
    print(f"n={n:3d}  exact {ok}/{tot}")
