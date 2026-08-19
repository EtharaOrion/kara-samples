import torch
from submission import AdditionTransformer
from train import batch_data
D='cuda'
for width,path in [(10,'model_w10.pt'),(12,'model_w12.pt')]:
 for rounds in (4,6,8,10,14):
  m=AdditionTransformer(width=width,rounds=rounds).to(D);m.load_state_dict(torch.load('/workspace/'+path,weights_only=True));m.eval();good=tot=0
  with torch.no_grad():
   for _ in range(20):
    x,y=batch_data(5000);good+=m(x).argmax(-1).eq(y).all(1).sum().item();tot+=len(x)
  print(width,rounds,good/tot)
