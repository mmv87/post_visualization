
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

labels=[tokenizer.decode(t) for t in token_ids[0]]
##load the embedding_npy
embed_1=np.load("./base_model_input_embedding.npy")
##cmap = plt.cm.viridis

sim_np=cosine_similarity(embed_1)
plt.figure(figsize=(10, 8))
plt.imshow(sim_np, cmap='viridis',vmin=-1, vmax=1)
plt.colorbar()

plt.xticks(range(len(labels)), labels, rotation=45)
plt.yticks(range(len(labels)), labels)

plt.title("Pairwise Token Similarity")
plt.show()

##print(sim.shape)


