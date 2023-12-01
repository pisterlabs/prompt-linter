import pickle
import openai
import pandas as pd
import numpy as np
from datasets import Dataset

openai.api_key = open("key.txt", "r").read()


class BuildFaissIndex:
    def __init__(self, config):
        self.model = config['embedding_model']
        self.chunks_path = config['chunks_path']
        self.hf_dataset = config['hf_dataset']


    def read_chunks(self, path):
        with open(path, "rb") as f:
            chunks = pickle.load(f)

        return chunks
    
    def get_embeddings(self, text):
        embedding = openai.Embedding.create(input=text,model=self.model)['data'][0]['embedding']
        return np.array(embedding)

    
    def build_index(self):
        # path = config[chunks_path]
        # model = config[embedding_model]
        
        data = self.read_chunks(self.chunks_path)
        data_df = pd.DataFrame(data, columns=['text'])
        data_df['embedding'] = data_df['text'].apply(self.get_embeddings)
        hf_dataset = Dataset.from_pandas(data_df)
        hf_dataset.add_faiss_index(column='embedding')
        hf_dataset.save_faiss_index('embedding','faiss_index.faiss')
        hf_dataset.drop_index('embedding')
        hf_dataset.save_to_disk(self.hf_dataset)


if __name__ == "__main__":
    config = { 'chunks_path': "chunks.pkl",
               'embedding_model':'text-embedding-ada-002',
               'hf_dataset': './hf_dataset'}
    
    build_faiss = BuildFaissIndex(config)
    build_faiss.build_index()