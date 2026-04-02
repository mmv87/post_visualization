
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from transformers import AutoModelForCausalLM,AutoTokenizer
import torch.nn as nn
import math
import os

##vocab=['upward downward trend slope increase decrease rise fall spike dip fluctuate oscillate seasonality cycle periodicity volatility stability plateau peak trough']
vocab=['upward downward trend slope increase decrease rise fall spike dip fluctuate oscillate seasonality cycle periodicity volatility stability plateau peak trough']

modelpath="D:/hf_cache/hub/models--microsoft--Phi-4-mini-reasoning/snapshots/0e3b1e2d02ee478a3743abe3f629e9c0cb722e0a"
##model_name="/home/mmk/projects/def-zonata/mmk/hf_cache/hub/models--microsoft--Phi-4-mini-reasoning/snapshots/7a8c4e2e81eae20a606d811f475d7dc316dd916a"
tokenizer=AutoTokenizer.from_pretrained(modelpath,local_files_only=True)
special_token_dict={'pad_token':"<|pad|>","additional_special_tokens":['<ts>','<ts/>']}
tokenizer.add_special_tokens(special_token_dict)
token_ids=tokenizer(vocab,return_tensors='pt',add_special_tokens=False,padding=True)['input_ids']

##check to get the name
##model=AutoModelForCausalLM.from_pretrained(modelpath,local_files_only=True)

"""for name, param in model.named_parameters():
    print(f"name:{name}, parameter:{param.shape}")"""
"""
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        print(f"Found a Linear Layer: {name}")
    else:
        print(f"a module: {name},{type(module)}")"""
###to chechthe linear layers
       
"""    
embed_layer=model.get_input_embeddings()
lm_head = model.get_output_embeddings()
print(type(embed_layer.weight),type(lm_head.weight))
for name,params in embed_layer.named_parameters():
    if params.requires_grad==True:
        print(params.shape)
    else:
        pass"""
path="D:/Doctoral_research/code_implementation/"
labels=[tokenizer.decode(t) for t in token_ids[0]]
##load the embedding_npy
stage_1_ft=np.load(os.path.join(path,"stage_2_input_embed_upd2.npy"))
base=np.load("./base_model_input_embedding.npy")
tast_vector_matrix=stage_1_ft-base
indices = [tokenizer.convert_tokens_to_ids(token) for token in vocab]
# Extract the Deltas for just these tokens
###domain_deltas = tast_vector_matrix[indices]
task_sim_matrix = cosine_similarity(tast_vector_matrix.astype(np.float64))
"""##cmap = plt.cm.viridis
sim_np=cosine_similarity(domain_deltas)
mask = np.eye(sim_np.shape[0], dtype=bool)
off_diag_values = sim_np[~mask]
# 2. Basic Min/Max
actual_min = off_diag_values.min()
actual_max = off_diag_values.max()"""
###print(f"Absolute Range (excluding diagonal): {actual_min:.4f} to {actual_max:.4f}")
plt.figure(figsize=(10, 8))
sns.heatmap(
    task_sim_matrix, 
    xticklabels=task_sim_matrix, 
    yticklabels=task_sim_matrix, 
    annot=False,       # Shows the raw numbers in the cells
    fmt=".3f", 
    vmin=0.,
    vmax=1,#Format to 3 decimal places
            #The upper bound of your sensitive range
    cmap="YlGnBu",    # Good for showing "depth" of similarity       # Optional: you can visually hide the diagonal entirely
)
##plt.imshow(sim_np, cmap='viridis',vmin=0.3704, vmax=0.9340)
##plt.colorbar()
plt.xticks(range(len(labels)), labels, rotation=45)
plt.yticks(range(len(labels)), labels,rotation=45)
plt.title("Pairwise Token Similarity")
plt.show()
##print(sim.shape)