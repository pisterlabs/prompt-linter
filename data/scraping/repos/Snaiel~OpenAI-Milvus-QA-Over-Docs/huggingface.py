from typing import List
from qa_over_docs.apis.base import BaseAPI, ChatResponse
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import requests, os
from time import time
from transformers import AutoTokenizer
from dotenv import load_dotenv
load_dotenv()

API_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
API_INFERENCE_URL = os.getenv("HUGGINGFACE_INFERENCE_ENDPOINT")
API_EMBEDDINGS_URL = os.getenv("HUGGINGFACE_EMBEDDINGS_ENDPOINT")

HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

SYSTEM_INSTRUCTIONS = """
Answer my question using the context below.
"""

REMINDER = """
Only respond with information from the context.
Do not mention anything outside of the provided context.
"""

MAX_TOKEN_LENGTH = 1000


class HuggingFace(BaseAPI):

    tokenizer = AutoTokenizer.from_pretrained("h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3")
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    def num_tokens_from_string(self, string: str) -> int:
        encoding = self.tokenizer(string)
        return len(encoding.tokens())
    

    def retrieve_response(self, question: str, relevant_docs: list[dict]) -> ChatResponse:
        base_token_length = self.num_tokens_from_string(SYSTEM_INSTRUCTIONS + f"\n\n{question}" + REMINDER) + 5

        context = ""

        for doc in relevant_docs:
            new_context = context
            new_context += "\n\n\n" + "{SOURCE ID}: " +  str(doc["pk"]) + "\n" + doc["text"]

            if self.num_tokens_from_string(new_context) > MAX_TOKEN_LENGTH - base_token_length:
                break

            context = new_context

        input = "<|prompt|>" + SYSTEM_INSTRUCTIONS + f"\n\nCONTEXT:{context}" + f"\n\n\nQUESTION:\n{question}" + f"\n\n{REMINDER}" + "<|endoftext|><|answer|>"

        print(input, self.num_tokens_from_string(input))

        payload = {
            "inputs": input,
            "parameters": {
                "max_new_tokens": 1024,
                "do_sample": True, 
                "temperature": 0.1, 
                "repetition_penalty": 3.2}
        }

        response = requests.post(API_INFERENCE_URL, headers=HEADERS, json=payload)

        response_string = response.json()[0]["generated_text"]
        
        response: ChatResponse  = {
            "relevant_source_ids": [],
            "answer": response_string
        }

        return response
    

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)
    

    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)