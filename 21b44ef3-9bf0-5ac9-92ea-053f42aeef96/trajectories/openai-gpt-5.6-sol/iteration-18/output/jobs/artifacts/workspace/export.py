import torch

path = "/workspace/submission.py"
state = torch.load("/workspace/checkpoint.pt", map_location="cpu", weights_only=True)
entries = []
for name, value in state.items():
    data = repr(value.float().tolist())
    entries.append(f"    {name!r}: torch.tensor({data}),")
replacement = "_TRAINED_STATE = {\n" + "\n".join(entries) + "\n}"
source = open(path).read()
source = source.replace("_TRAINED_STATE = None", replacement)
open(path, "w").write(source)
print("exported", len(state), "tensors", sum(x.numel() for x in state.values()))
