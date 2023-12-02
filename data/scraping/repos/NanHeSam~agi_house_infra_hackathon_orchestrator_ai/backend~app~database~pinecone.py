import openai
import pinecone
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file


class PineconeClient:
    def __init__(self) -> None:
        pinecone.init(
            api_key=os.environ['PINECONE_API_KEY'],
            environment=os.environ['PINECONE_ENVIRONMENT']
        )
        openai.api_key = os.environ['OPENAI_API_KEY']

    def get_embeddings(self, strings, model="text-embedding-ada-002"):
        if isinstance(strings, str):
            strings = [strings]

        for i in range(len(strings)):
            strings[i] = strings[i].strip().replace("\n", " ")

        results = openai.Embedding.create(input=strings, model=model)

        embeddings = []
        for result in results['data']:
            embeddings.append(result['embedding'])
        return embeddings


    def query_pinecone(self, string, top_k=1, index_name = 'example-index'):
        index = pinecone.Index(index_name)
        response = index.query(
            top_k=top_k,
            include_values=False,
            include_metadata=True,
            vector=self.get_embeddings(string)[0]
        )
        return response.to_dict()['matches']

