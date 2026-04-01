from torch.utils.data import Dataset,DataLoader
import torch
import json
from transformers import AutoModelForCausalLM,AutoTokenizer
import numpy as np
import os
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt

### to get the vocab for plotting
vocab=['upward downward trend slope increase decrease rise fall spike dip fluctuate oscillate seasonality cycle periodicity volatility stability plateau peak trough']
near_ts_vocab=['value signal level reading measurement observation metric indicator variable system process mechanism response output input feedback control state condition analysis estimation evaluation model prediction inference correlation regression parameter coefficient distribution function sample dataset feature dimension']

words = vocab[0].split()
print(f'total words:{len(words)}')

###print(len(vocab))
device ='cuda' if torch.cuda.is_available() else 'cpu'
modelpath="D:/hf_cache/hub/models--microsoft--Phi-4-mini-reasoning/snapshots/0e3b1e2d02ee478a3743abe3f629e9c0cb722e0a"
##print('path_read')
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
model_name='./hub/microsoft/phi-4-mini-reasoning'
model=AutoModelForCausalLM.from_pretrained(modelpath,local_files_only=True)
##model.to(device)
tokenizer=AutoTokenizer.from_pretrained(modelpath,local_file_only=True)
###input_text='The following timeseries in the model'
token_ids = tokenizer(vocab,return_tensors='pt',add_special_tokens=False,padding=True)['input_ids']

###print(token_ids.shape)
##print(token_ids)
##print(tokenizer.decode(token_ids[0]))

##special_token_dict={'pad_token':"<|pad|>","additional_special_tokens":['<ts>','<ts/>']}
##tokenizer.add_special_tokens(special_token_dict)

embeddings_layer=model.get_input_embeddings()

with torch.no_grad():
    embeddings=embeddings_layer(token_ids[0])

print(f'token_len :{embeddings.shape}')

embeddings = embeddings.view(-1, embeddings.shape[-1])
_file_model_embedding="./base_model_input_embedding.npy"

import torch.nn.functional as F
embeddings = F.normalize(embeddings, p=2, dim=1)
np_embedding=embeddings.detach().cpu().to(torch.float32).numpy()
np.save(_file_model_embedding,np_embedding)
 
"""pca = PCA(n_components=24)
embeddings_pca = pca.fit_transform(embeddings.cpu().numpy())
print(f'shape_after_pca:{embeddings_pca.shape}')
##import umap
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric="cosine"
)
embeddings_2d = reducer.fit_transform(embeddings_pca)
umap_emb_file="./embed_n_neighbors_15.npy"

np.save(umap_emb_file,embeddings_2d)
print(type(embeddings_2d))
plt.figure(figsize=(8, 6))"""
###plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
"""N = embeddings_2d.shape[0]
# Create a colormap
cmap = plt.cm.viridis
# Normalize indices to [0,1]
colors = cmap(np.linspace(0, 1, N))

for i, token in enumerate(token_ids[0]):
    label=tokenizer.decode(token,skip_special_tokens=True)
    plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], color=colors[i])
    plt.annotate(label,(embeddings_2d[i, 0], embeddings_2d[i, 1]),rotation=45,)

plt.title("Token Embedding Visualization")
plt.show()"""


###plotting phase
"""
import numpy as np
import matplotlib.pyplot as plt
_file_embed_1="./embed_1.npy"
_file_embed_2="./embed_n_neighbors_15.npy"

embed_1=np.load(_file_embed_1)
embed_2=np.load(_file_embed_2)
cmap = plt.cm.viridis
N = embed_1.shape[0]
colors = cmap(np.linspace(0, 1, N))
plt.figure(figsize=(10,15))

for i, token in enumerate(token_ids[0]):
    label=tokenizer.decode(token,skip_special_tokens=True)
    plt.scatter(embed_1[i, 0], embed_1[i, 1], color=colors[i])
    plt.annotate(label,(embed_1[i, 0], embed_1[i, 1]),rotation=45,)
    
    plt.scatter(embed_2[i, 0], embed_2[i, 1], color=colors[i])
    plt.annotate(label,(embed_2[i, 0], embed_2[i, 1]),rotation=45,)
    
plt.show()

##step-1 build a graph on the projected dimension
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import numpy as np

sim = cosine_similarity(np_embedding)
G = nx.Graph()
labels =[]

# add nodes
for i, token in enumerate(token_ids[0]):
    labels.append(tokenizer.decode(token))
    G.add_node(labels[i], pos=embed_1[i])

# add edges (kNN or threshold)
k = 1
for i in range(len(labels)):
    neighbors = np.argsort(sim[i])[-k-1:-1]  # top-k neighbors

    for j in neighbors:
        G.add_edge(labels[i], labels[j], weight=sim[i, j])
        
##step -2 Plot the graph structure
pos = {token: embed_1[i] for i, token in enumerate(labels)}
edges = G.edges(data=True)
weights = [d['weight'] for (_, _, d) in edges]
nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=300,
    width=weights  # edge thickness = similarity
)

plt.show()"""



