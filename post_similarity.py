
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from transformers import AutoModelForCausalLM,AutoTokenizer

##vocab=['upward downward trend slope increase decrease rise fall spike dip fluctuate oscillate seasonality cycle periodicity volatility stability plateau peak trough']
vocab=['upward downward trend slope increase decrease rise fall spike dip fluctuate oscillate seasonality cycle periodicity volatility stability plateau peak trough']

modelpath="D:/hf_cache/hub/models--microsoft--Phi-4-mini-reasoning/snapshots/0e3b1e2d02ee478a3743abe3f629e9c0cb722e0a"
##model_name="/home/mmk/projects/def-zonata/mmk/hf_cache/hub/models--microsoft--Phi-4-mini-reasoning/snapshots/7a8c4e2e81eae20a606d811f475d7dc316dd916a"
tokenizer=AutoTokenizer.from_pretrained(modelpath,local_files_only=True)
token_ids=tokenizer(vocab,return_tensors='pt',add_special_tokens=False,padding=True)['input_ids']

##check to get the name
model=AutoModelForCausalLM.from_pretrained(modelpath,local_files_only=True)
"""for name, param in model.named_parameters():
    print(f"name:{name}, parameter:{param.shape})"""
    
    ###to chechthe linear layers
import torch.nn as nn
"""for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        print(f"Found a Linear Layer: {name}")
    else:
        print(f"a module: {name},{type(module)}")"""
        
embed_layer=model.get_input_embeddings()
lm_head = model.get_output_embeddings()

print(type(embed_layer.weight),type(lm_head.weight))
"""
for name,params in embed_layer.named_parameters():
    if params.requires_grad==True:
        print(params.shape)
    else:
        pass"""
    
        
"""              
labels=[tokenizer.decode(t) for t in token_ids[0]]
##load the embedding_npy
embed_1=np.load("./stage_2_input_embed.npy")
##cmap = plt.cm.viridis

sim_np=cosine_similarity(embed_1)
plt.figure(figsize=(10, 8))
plt.imshow(sim_np, cmap='viridis',vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(labels)), labels, rotation=45)
plt.yticks(range(len(labels)), labels)

plt.title("Pairwise Token Similarity")
plt.show()"""

##print(sim.shape)


