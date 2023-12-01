from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer
from langchain.embeddings import LlamaCppEmbeddings, GPT4AllEmbeddings
import os


class DocumentationEmbedder(object):
    docs_dir: str = None
    db = None
    embedding_tokenizer = None
    embedding_model = None
    
    def __init__(self,
                 embedding_model_name: str = 'distilbert',
                 device: str = 'cpu'):
        
        self.embedding_model_name = embedding_model_name
        self.device = device
        self.get_embeddings_object()
    
    def get_embeddings_object(self):
        """
        Chooses the embedding model depending on the provided name
        """
        
        if self.embedding_model_name.startswith('openai'):
            self.embedding_model = OpenAIEmbeddings(
                model = 'text-embedding-ada-002'
            )
        
        elif self.embedding_model_name == 'llamacpp':
            self.embedding_model = LlamaCppEmbeddings(
                model_path = r'C:\Users\ilija\llama.cpp\models\7B\ggml-model-q4_0.gguf',
                verbose = True,
                n_ctx = 1024,
                n_gpu_layers = 40,
                n_batch = 512
            )
        
        elif self.embedding_model_name == 'llamacpppython':
            self.embedding_model = LlamaCppEmbeddings(
                model_path = r'C:\Users\ilija\llama.cpp\models\7B\ggml-model-q4_0.gguf',
                verbose = True,
                n_ctx = 1024,
                n_gpu_layers = 40,
                n_batch = 512
            )
        
        elif self.embedding_model_name == 'sbert':
            self.embedding_model = GPT4AllEmbeddings(
                model_path = r"ggml-all-MiniLM-L6-v2-f16.bin"
            )
        
        elif self.embedding_model_name == 'ggml-falcon':
            print("Using falcon model")
            self.embedding_model = GPT4AllEmbeddings(
                model = r"D:\python_projects\loka_final\models\ggml-model-gpt4all-falcon-q4_0.bin"
            )
        
        elif self.embedding_model_name.startswith('flan'):
            self.embedding_model = GPT4AllEmbeddings(
                model_path = r"ggml-all-MiniLM-L6-v2-f16.bin"
            )
            
        else:
            model_name = None
            if self.embedding_model_name.startswith('distilbert'):
                model_name = "sentence-transformers/distilbert-base-nli-stsb-mean-tokens"
            elif self.embedding_model_name.startswith('bert'):
                model_name = "sentence-transformers/bert-base-nli-stsb-mean-tokens",
            elif self.embedding_model_name.startswith('roberta'):
                model_name = "symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli"
            elif self.embedding_model_name.startswith('bge-large'):
                model_name = "BAAI/bge-large-en-v1.5"
            elif self.embedding_model_name.startswith('bge-llm'):
                model_name = "BAAI/bge-large-en-v1.5"

            self.embedding_model = HuggingFaceEmbeddings(
                model_name = model_name,
                model_kwargs = {'device': 'cuda:0'} if self.device.startswith('cuda') else {},
                # encode_kwargs = {'normalize_embeddings': False}`
            )
            self.embedding_tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if not self.embedding_model:
            raise NameError("The model_name for embeddings that you entered is not supported")
        
        

