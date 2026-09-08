import os,sys,torch
sys.path.insert(0,'/workspace')
import submission
old=torch.load('/workspace/final.pt',map_location='cpu',weights_only=True)
m=submission.AdditionTransformer(2)
new=m.state_dict()
for k in new:
    if k in old and k not in ('ff_in.weight','classifier.weight'):
        new[k]=old[k]
new['ff_in.weight']=old['ff_in.weight']*old['ff_norm.weight'].unsqueeze(0)
new['ff_in.bias']=old['ff_in.weight']@old['ff_norm.bias']
new['classifier.weight']=old['classifier.weight']*old['final_norm.weight'].unsqueeze(0)
new['classifier.bias']=old['classifier.weight']@old['final_norm.bias']
m.load_state_dict(new); m.eval()
vec=torch.nn.utils.parameters_to_vector(m.parameters()).tolist()
path='/workspace/submission.py'; text=open(path).read(); a=text.index('# TRAINED_WEIGHTS_START'); b=text.index('# TRAINED_WEIGHTS_END')+len('# TRAINED_WEIGHTS_END')
block='# TRAINED_WEIGHTS_START\n_TRAINED = ['+','.join(format(x,'.9g') for x in vec)+']\n# TRAINED_WEIGHTS_END'
open(path+'.new','w').write(text[:a]+block+text[b:]); os.replace(path+'.new',path)
print(len(vec))
