from openai.embeddings_utils import get_embedding
from dotenv import load_dotenv
from ast import literal_eval
from tqdm import tqdm

import pinecone
import webvtt
import openai
import pandas as pd
import tiktoken
import random
import itertools
import os
import argparse

load_dotenv()

# create exception class for embedding class
class EmbeddingException(Exception):
    pass

class Embeddings:
    def __init__(self, df):
        # embedding dataframe
        self._df = df
        # openai parameters
        self._embedding_encoding = "cl100k_base"
        self._embedding_model = "text-embedding-ada-002"
        # load keys
        try:
            self._openai_key = os.getenv('OPENAI_KEY')
            self._pinecone_key = os.getenv('PINECONE_KEY')
            self._pinecone_environment = os.getenv('PINECONE_ENVIRONMENT', 'us-west1-gcp')
            openai.api_key = self._openai_key
        except:
            raise EmbeddingException("No API keys found")
        # init pinecone
        pinecone.init(api_key=self._pinecone_key, environment=self._pinecone_environment)
        # use first index in list
        self._pinecone_index = pinecone.Index(pinecone.list_indexes()[0])
            
    @classmethod
    def read_vtt_file(cls, file_path):
        captions = webvtt.read(file_path)
        # TODO: so inefficient, fix this
        df = pd.DataFrame()
        text, start, end = [], [], []
        for caption in captions:
            text.append(caption.text)
            start.append(str(caption.start))
            end.append(str(caption.end))
        df["captions"] = text
        df["start"] = start
        df["end"] = end
        return df

    @classmethod
    def read_csv_embedding_file(cls, file_path):
        df = pd.read_csv(file_path)
        # literal_eval needed to parse string to list
        df["embedding"] = df["embedding"].apply(lambda x: literal_eval(x))
        return df

    def get_encodings(self):
        encoding = tiktoken.get_encoding(self._embedding_encoding)
        self._df["n_tokens"] = self._df["captions"].apply(lambda x: len(encoding.encode(x)))

    def get_embeddings(self):
        tqdm.pandas(desc="generating embeddings")
        self._df["embedding"] = self._df["captions"].progress_apply(
            lambda x: get_embedding(x, engine=self._embedding_model))
        
    def save_embeddings_to_csv(self, file_name):
        print(f"Saving embeddings to {file_name}")
        self._df.to_csv(file_name)

    def upload_embeddings(self):
        # A helper function to break an iterable into chunks of size batch_size
        def chunks(iterable, batch_size=100):
            it = iter(iterable)
            chunk = tuple(itertools.islice(it, batch_size))
            while chunk:
                yield chunk
                chunk = tuple(itertools.islice(it, batch_size))

        df = self._df

        example_data_generator = map(lambda i: (
            f'id-{i}',
            df["embedding"][i],
            {"captions": df["captions"][i], "time": f'{df["start"][i]}-{df["end"][i]}'}
        ), range(len(df["embedding"])))

        for ids_vectors_chunk in tqdm(chunks(example_data_generator, batch_size=100)):
            self._pinecone_index.upsert(vectors=ids_vectors_chunk)

    def query_embeddings(self, query, top_k=5):
        embed = openai.Embedding.create(input=query, engine=self._embedding_model)['data'][0]['embedding']
        # query, returning the top 5 most similar results
        res = self._pinecone_index.query([embed], top_k=top_k, include_metadata=True)
        for match in res['matches']:
            print(f"{match['score']:.2f}: {match['metadata']['captions']} {match['metadata']['time']}")

def main():
    parser = argparse.ArgumentParser('embed_vtt')
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser('generate', help='generate embeddings')
    generate.add_argument('--vtt-file', type=str, help='vtt file path')

    generate = subparsers.add_parser('upload', help='upload embeddings to pinecone')
    generate.add_argument('--csv-embedding-file', type=str, help='csv embeddings file path')

    query = subparsers.add_parser('query', help='query pinecone')
    query.add_argument('--text', type=str, help='text to query')

    args = parser.parse_args()
    if args.command == "generate":
        df = Embeddings.read_vtt_file(args.vtt_file)
        embed = Embeddings(df)
        embed.get_embeddings()
        embed.save_embeddings_to_csv(f"{args.vtt_file.split('.')[0]}_embeddings.csv")
    elif args.command == "upload":
        df = Embeddings.read_csv_embedding_file(args.csv_embedding_file)
        embed = Embeddings(df)
        embed.upload_embeddings()
    elif args.command == "query":
        embed = Embeddings(None)
        embed.query_embeddings(args.text)

if __name__ == '__main__':
    main()
