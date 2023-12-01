from typing import List
from httpx import stream

from langchain.schema import Document
from llama_cpp import ChatCompletion, ChatCompletionMessage
import openai

class HostedLlm():
    def __init__(self, 
                url: str = "http://localhost:8000/v1", 
                api_key: str = "password",
                temperature: float = 0.2) -> None:
        self.url = url
        openai.api_key = api_key
        openai.api_base = url
        self.temperature = temperature
    
    system_message = """You are a helpful assistant. You are helping a user with a question.
        Answer in a concise way in a few sentences.
        Use the following context to answer the user's question.
        If the given given context does not have the information to answer the question, you should answer "I don't know" and don't say anything else."""

    def RAG_QA_chain(self, retrieved_docs: List[Document], query: str) -> str:

        context: str = "\n".join(doc.page_content for doc in retrieved_docs)
        # We ignore the type because it is entirely dependent on streaming
        result: ChatCompletion = openai.ChatCompletion.create(
            messages = [
                {
                    "role": "system",
                    "content": self.system_message + "\n" + context
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            temperature=self.temperature,
            stream=False
        ) # type: ignore

        # return generated text
        return result["choices"][0]["message"]["content"] or ""