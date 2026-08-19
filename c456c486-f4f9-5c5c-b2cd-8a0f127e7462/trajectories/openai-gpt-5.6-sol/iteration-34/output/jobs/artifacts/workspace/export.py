from pathlib import Path
import torch
import train

model = train.Adder()
model.load_state_dict(torch.load("/workspace/final.pt", map_location="cpu", weights_only=True))
weights = []
for parameter in model.parameters():
    values = parameter.detach().float().reshape(-1).tolist()
    weights.append("[" + ",".join(format(value, ".9g") for value in values) + "]")
literal = "[\n" + ",\n".join(weights) + "\n]"
path = Path("/workspace/submission.py")
text = path.read_text()
text = text.replace("_WEIGHTS = None", "_WEIGHTS = " + literal)
path.write_text(text)
print("exported", sum(p.numel() for p in model.parameters()), "values", path.stat().st_size, "bytes")
