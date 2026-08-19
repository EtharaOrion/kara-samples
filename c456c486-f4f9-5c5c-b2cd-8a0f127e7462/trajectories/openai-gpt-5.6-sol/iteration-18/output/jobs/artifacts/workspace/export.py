import torch

path = "/workspace/submission.py"
state = torch.load("/workspace/model.pt", map_location="cpu", weights_only=True)
with open(path) as f:
    source = f.read()
marker = "_WEIGHTS = "
if marker in source:
    source = source[:source.index(marker)]
items = []
for name, value in state.items():
    values = ",".join(repr(float(x)) for x in value.flatten())
    items.append(f"{name!r}:({tuple(value.shape)!r},[{values}])")
source += "\n\n_WEIGHTS = {" + ",".join(items) + "}\n"
source += '''

def build_model():
    model = Adder()
    state = {}
    for name, (shape, values) in _WEIGHTS.items():
        state[name] = torch.tensor(values, dtype=torch.float32).reshape(shape)
    model.load_state_dict(state)
    return model, {"architecture": "two-block causal digit transformer", "parameters": 5034}
'''
# Remove the earlier untrained build_model while preserving add().
start = source.index("\ndef build_model():")
end = source.index("\ndef add(", start)
old = source[start:end]
# The appended trained builder must remain; remove only first occurrence.
source = source[:start] + source[end:]
with open(path, "w") as f:
    f.write(source)
print(path, len(source), "chars", sum(v.numel() for v in state.values()), "weights")
