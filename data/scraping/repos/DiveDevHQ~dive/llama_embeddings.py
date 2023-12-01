from langchain.embeddings.base import Embeddings
from typing import Optional, List
from langchain import HuggingFaceHub
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModel
import time
import torch
import os
import ssl
from torch import cuda, bfloat16,no_grad
import transformers
import environ
env = environ.Env()
environ.Env.read_env()  # reading .env file


class LlamaEmbeddings(Embeddings):

    def embed_documents(
            self, texts: List[str], chunk_size: Optional[int] = 0
    ) -> List[List[float]]:
        return self.compute_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        embeddings=self.compute_embeddings([text])
        if len(embeddings)>0:
            return embeddings[0]
        else:
            return []


    def compute_embeddings(self, texts: List[str]):
        #Llama v2 7B model
        model_id = 'meta-llama/Llama-2-7b-chat-hf'
        device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
        # set quantization configuration to load large model with less GPU memory
        # this requires the `bitsandbytes` library
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16
        )
        # begin initializing HF items, need auth token for these
        hf_auth = env.str('HUGGING_FACE_AUTH', default='') or os.environ.get('use_auth_token', '')
        model_config = transformers.AutoConfig.from_pretrained(
            model_id,
            use_auth_token=hf_auth
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map='auto',
            use_auth_token=hf_auth
        )
        model.eval()
        print(f"Model loaded on {device}")
        tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model = transformers.AutoModel.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
        model.resize_token_embeddings(len(tokenizer))
        encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings

    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)



