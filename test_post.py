
import torch
import torch.nn as nn
import torch.nn.functional as F
##from TS_encoder import PatchTSTEncoder
from  transformers import AutoModelForCausalLM,AutoTokenizer
from ts_dataloader import ts_textual,collate_func
import os
import sys
import numpy as np
from torch.utils.data import Dataset,DataLoader
from peft import get_peft_model 
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
###modules for hybrid ts_encoder building
"""from modules.conv_module import ConvFeatureExtraction
from modules.ts_encoder_rel_bias import PatchTSTEncoder
from modules.ts_encoder import llm_projection"""

device ='cuda' if torch.cuda.is_available() else 'cpu'

##location of llm_base model
model_name="/home/mmk/projects/def-zonata/mmk/hf_cache/hub/models--microsoft--Phi-4-mini-reasoning/snapshots/7a8c4e2e81eae20a606d811f475d7dc316dd916a"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
model = AutoModelForCausalLM.from_pretrained(model_name,local_files_only=True)
##expanded tokenizer path
tokenizer_path =os.path.join(os.environ["SLURM_TMPDIR"],'llm_tokenizer')
tokenizer_modified =AutoTokenizer.from_pretrained(tokenizer_path)

model_dtype=next(model.parameters()).dtype
model.resize_token_embeddings(len(tokenizer_modified))

## to expand the tokenizer to add the special tokens <ts> <ts/>
"""special_token_dict={'pad_token':"<|pad|>","additional_special_tokens":['<ts>','<ts/>']}
tokenizer.add_special_tokens(special_token_dict)"""
##dataset fetching
import json
_json_file = os.path.join(os.environ["SLURM_TMPDIR"],"processed_dataset.jsonl")

###datapipeline
dataset=ts_textual(128,128,tokenizer_modified,_json_file,device=device)
dataloader=DataLoader(dataset,batch_size=1,shuffle=True,collate_fn=lambda b:collate_func(b,tokenizer=tokenizer_modified))
"""
dataset= ts_multimodal_text(128,128,_json_file,tokenizer,device=device,model_dtype=None)
dataloader=DataLoader(dataset,batch_size=1,shuffle=True,collate_fn=lambda b:collate_func(b,tokenizer=tokenizer,device=device))"""

##Lora_config defintion based on best practices
peft_config = LoraConfig(
            r=16, lora_alpha=32,
            target_modules=["o_proj",'qkv_proj','gate_up_proj','down_proj'],
            modules_to_save=["embed_tokens"],lora_dropout=0.1, # important for Stage-2  as to keep th ties
            task_type="CAUSAL_LM",ensure_weight_tying=True)


peft_model=get_peft_model(model,peft_config)

print(model.print_trainable_parameters())
