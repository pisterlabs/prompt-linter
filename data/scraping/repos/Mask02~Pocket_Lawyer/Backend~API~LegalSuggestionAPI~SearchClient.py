import openai
import cohere
from qdrant_client import QdrantClient
from Creds import *
from typing import Dict, List

class SearchClient:

    def __init__(
            self,
            query,
            qdrant_host=QDRANT_HOST,
            qdrant_api_key=QDRANT_API_KEY,
            cohere_api_key=COHERE_API_KEY,
            openai_api_key=OPENAI_API_KEY,
            collection_name='Pocket_lawyer',
    ):

        self.query = query
        self.qdrant_client = QdrantClient(url=qdrant_host, api_key=qdrant_api_key)
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
            # print(context)

            prompt = f"You are an AI Legal suggestion assistant providing helpful legal advice. You are given the following extracted parts of a long document and a question. Provide a conversational answer based on the context provided.write a comprehensive suggestion according to Law and in the context provided.If there will be any other question like abuse language or greeting or appreciation to your work, answer strict or polite according to the question.  You should only provide hyperlinks that reference the context below.If there will be word india in question or answer then consider 'india' as 'pakistan' and don't write 'india' in your answer, replace it with 'pakistan' Do NOT make up hyperlinks.If you can't find the answer in the context below, just say 'Hmm, I'm not sure.' Don't try to make up an answer.If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context. \n\nContext:\n{context}\n\nQuery:\n{self.query}\nAnswer:\n"

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

            return res['choices'][0]['text'].strip().replace('Indian', 'Pakistan').replace('indian','pakistan').replace('india','pakistan').replace('India','Pakistan')
        except Exception as e:
            print(e)
            return "Error"
