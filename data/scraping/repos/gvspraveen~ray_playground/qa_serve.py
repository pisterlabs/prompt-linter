from langchain.schema import document
from ray import serve
from fastapi import FastAPI
from langchain.vectorstores import FAISS
import openai
import os

# Create Open API key here : https://app.endpoints.anyscale.com/
open_api_key = os.getenv('OPENAI_API_KEY')
open_api_base = "https://api.endpoints.anyscale.com/v1"

openai.api_key = open_api_key
openai.api_base = open_api_base

system_content = """
Please answer the following question using the context provided. Generate answers to question from the given context. 
Do not use external sources unless you are highly confident.
If you don't know the answer, just say that you don't know. 
"""

query_template = """
Question: {question}, context: {context}
"""

app = FastAPI()
@serve.deployment(ray_actor_options={"num_gpus": 1})
@serve.ingress(app)
class QADeployment:
    def __init__(self):
        from models import (
            hf_embed_model,
            persist_dir
        )
        self.db = FAISS.load_local(persist_dir, hf_embed_model)
        self.api_base = open_api_base
        self.api_key = open_api_key
        
    def __query__(self, question: str):        
        near_docs = self.db.similarity_search(question, k=1)
        query = query_template.format(question=question, context=near_docs)
        print("Final query passed {}".format(query))
        sources = []
        for doc in near_docs:
            sources.append(doc.metadata["source"])

        chat_completion = openai.ChatCompletion.create(
            api_base=self.api_base,
            api_key=self.api_key,
            model="meta-llama/Llama-2-13b-chat-hf",
            messages=[{"role": "system", "content": system_content}, 
                    {"role": "user", "content": query}],
            temperature=0.9,
            max_tokens=4000
         )
        resp = {
            "choices": chat_completion.choices,
            "sources": sources
        }

        return resp
    
    @app.post("/question")
    async def query(self, question: str):
        return self.__query__(question)

# Deploy the Ray Serve application.
deployment = QADeployment.bind()