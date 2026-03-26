
###using graph_ to plot the umap's p_ij
import umap
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
import json
from transformers import AutoModelForCausalLM,AutoTokenizer
import numpy as np
import os
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import networkx as nx


vocab=['upward downward trend slope increase decrease rise fall spike dip fluctuate oscillate seasonality cycle periodicity volatility stability plateau peak trough']

###print(len(vocab))
device ='cuda' if torch.cuda.is_available() else 'cpu'
modelpath="D:/hf_cache/hub/models--microsoft--Phi-4-mini-reasoning/snapshots/0e3b1e2d02ee478a3743abe3f629e9c0cb722e0a"
##print('path_read')
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
model_name='./hub/microsoft/phi-4-mini-reasoning'
##model=AutoModelForCausalLM.from_pretrained(modelpath,local_files_only=True)
##model.to(device)
tokenizer=AutoTokenizer.from_pretrained(modelpath,local_file_only=True)
###input_text='The following timeseries in the model'
token_ids = tokenizer(vocab,return_tensors='pt',add_special_tokens=False,padding=True)['input_ids']

###update the file
embed_file="./stage_1_input_embed.npy"
base_input_embed="./base_model_input_embedding.npy"

embeddings =np.load(base_input_embed)
print(embeddings.shape)
##pca to reduce the dimension to 24 
pca = PCA(n_components=24)
embeddings_pca = pca.fit_transform(embeddings)

umap_model = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric="cosine"
)
embeddings_2d = umap_model.fit_transform(embeddings_pca)
print(embeddings_2d.shape)
labels=[]
##loop to decode the tokenizer
for i,token in enumerate(token_ids[0]):
    labels.append(tokenizer.decode(token))
    
###graph_ is a sparse matrix storing fuzzy edge weights
graph = umap_model.graph_

G = nx.Graph()
###plot only strong edges
G_filtered = nx.Graph()
k=2  # top neighbors per node

###threshold = np.percentile(weights, 75)  # keep top 25% strongest edges
"""G_filtered = nx.Graph(
    (u, v, d) for u, v, d in G.edges(data=True) if d['weight'] > threshold
)"""
coo = graph.tocoo()  # convert to COO for iteration
for i, j, v in zip(coo.row, coo.col, coo.data):
    if i != j:  # ignore self-loops
        G.add_edge(labels[i], labels[j], weight=v)

pos = {labels[i]: embeddings_2d[i] for i in range(len(labels))}

##nx.draw_networkx_edges(G,pos,alpha=0.3)
for node in G.nodes():
    edges = list(G.edges(node, data=True))
    # sort by weight descending
    edges = sorted(edges, key=lambda x: x[2]['weight'], reverse=True)
    for u, v, d in edges[:k]:
        G_filtered.add_edge(u, v, weight=d['weight'])
        
edges = G_filtered.edges(data=True)
weights = np.array([d['weight'] for (_, _, d) in edges])
print(f"weights:{weights}")
# scale to width between 1 and 5
scaled_weights = 1 + 4 * (weights - weights.min()) / (weights.max() - weights.min() + 1e-5)

import matplotlib.pyplot as plt
node_colors = np.arange(len(labels))  # unique color per token
plt.figure(figsize=(10, 8))

nx.draw(
    G_filtered,
    pos,
    with_labels=True,
    node_size=100,
    node_color=node_colors,
    cmap=plt.cm.tab20,
    width=weights,         # edge thickness = fuzzy weight
    edge_color=weights,           # edge color = fuzzy weight
    edge_cmap=plt.cm.viridis      # colormap for edges
)

plt.show()