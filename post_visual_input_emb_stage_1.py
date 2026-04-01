###to load the embedding layer and the corresponsding .pt file
###import umap
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
import json
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer
import numpy as np
import os
from sklearn.decomposition import PCA
##import umap
import matplotlib.pyplot as plt

#vocabulary
vocab=['upward downward trend slope increase decrease rise fall spike dip fluctuate oscillate seasonality cycle periodicity volatility stability plateau peak trough']

non_ts_tokens=[
    "table", "chair", "window", "bottle", "phone", "keyboard", "screen",
    "friend", "family", "teacher", "conversation", "emotion", "memory",
    "river", "mountain", "forest", "ocean", "cloud", "wind",
    "freedom", "justice", "belief", "idea", "knowledge",
    "build", "create", "design", "imagine", "explore"
]

##model location in the login node
model_name="/home/mmk/projects/def-zonata/mmk/hf_cache/hub/models--microsoft--Phi-4-mini-reasoning/snapshots/7a8c4e2e81eae20a606d811f475d7dc316dd916a"

_input_embed_layer=os.path.join(os.environ["SLURM_TMPDIR"],'aligned_embeddings_ver2.pt')
embedding_file=os.path.join(os.environ["SLURM_TMPDIR"],'stage_1_input_embed_upd.npy')

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
device ='cuda' if torch.cuda.is_available() else 'cpu'

model = AutoModelForCausalLM.from_pretrained(model_name,local_files_only=True)
tokenizer=AutoTokenizer.from_pretrained(model_name,local_files_only=True)
###tokenizer_path =os.path.join(os.environ["SLURM_TMPDIR"],'llm_tokenizer')

special_token_dict={'pad_token':"<|pad|>","additional_special_tokens":['<ts>','<ts/>']}
tokenizer.add_special_tokens(special_token_dict)
model.resize_token_embeddings(len(tokenizer))
token_ids = tokenizer(vocab,return_tensors='pt',add_special_tokens=False,padding=True)['input_ids']

##trained_input_embed=torch.load(_input_embed_layer,map_location=device)
"""print(f"input_embed_keys:{trained_input_embed.keys()}")
input_embed_weights=trained_input_embed['weight']"""

print('loaded_embeddings')
embedding_layer=model.get_input_embeddings()
embed_state_dict = torch.load(_input_embed_layer,map_location=device)
embedding_layer.load_state_dict(embed_state_dict)
### without calculating the gradients
with torch.no_grad():
    ##input_embeddings.weight.copy_(input_embed_weights)
    vocab_embedding=embedding_layer(token_ids[0])
    
vocab_embedding = vocab_embedding.view(-1, vocab_embedding.shape[-1])
##embeddings = F.normalize(vocab_embedding, p=2, dim=1)
vocab_embedding_npy=vocab_embedding.cpu().to(torch.float32).numpy()

np.save(embedding_file,vocab_embedding_npy)
print('file_saved')

    