import sys,torch
sys.path.insert(0,'/workspace');import submission
m,_=submission.build_model();m.cuda()
def attention_scores(tokens,layer=0):
 with torch.no_grad():
  x=m.token(tokens)+m.position[:tokens.shape[1]];z=m.norm_attn[layer](x)
  q=m.query[layer](z).view(z.shape[0],tokens.shape[1],4,5).transpose(1,2)
  k=m.key[layer](z).view(z.shape[0],tokens.shape[1],4,5).transpose(1,2)
  return torch.softmax(q@k.transpose(-2,-1)/(5**.5),-1)
def tok(a,b):
 out=[]
 for x,y in zip(reversed(f'{a:08d}'),reversed(f'{b:08d}')):out += [int(x),int(y)]
 return out+[10]
x=torch.tensor([tok(12345678,87654321),tok(99999999,10000000)],device='cuda')
a=attention_scores(x)
print('attention_delta',float((a[0]-a[1]).abs().max()))
print('state_arrays',len(submission._TRAINED_STATE),'parameters',len(list(m.parameters())))
