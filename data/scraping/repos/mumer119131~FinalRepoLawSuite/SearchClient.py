import openai
import cohere
from qdrant_client import QdrantClient
import Creds
from typing import Dict, List


class SearchClient:

    def __init__(
            self, 
            query,
            qdrant_host=Creds.QDRANT_HOST, 
            qdrant_api_key=Creds.QDRANT_API_KEY,
            cohere_api_key=Creds.COHERE_API_KEY,
            openai_api_key=Creds.OPENAI_API_KEY, 
            collection_name = 'legalcompanion', 
        ):

        self.query = query
        self.qdrant_client = QdrantClient(host=qdrant_host, api_key=qdrant_api_key)
        self.collection_name = collection_name
        self.co_client = cohere.Client(api_key=cohere_api_key)
        openai.api_key = openai_api_key

    # Qdrant requires data in float format
    def _float_vector(self, vector: List[float]):
        return list(map(float, vector))

    # Search using text query
    def search(self, limit: int = 3):

        try:
            query_vector = self.co_client.embed(texts=[self.query], model="large", truncate="RIGHT").embeddings[0]

            return self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=self._float_vector(query_vector),
                limit=limit,
            )
        except Exception as e:
            print(e)

    # Answer the user query based on the context of most relevant result of semantic search of user query
    def answerQuery(self):

        try:
            search_result = self.search()
                
            context = search_result[0].payload["text"]

            prompt = f"Answer the Query based on the contexts, if it's not in the contexts say 'I don't know the answer'. Answer should be in the language of query. \n\nContext:\n{context}\n\nQuery:\n{self.query}\nAnswer:\n"

            res = openai.Completion.create(
                    engine='text-davinci-003',
                    prompt=prompt,
                    temperature=0,
                    max_tokens=1000,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None
            )

            return res['choices'][0]['text'].strip()  
        except Exception as e:
            print(e)
            return "Error"
            