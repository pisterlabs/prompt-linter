import os

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.indexes import VectorstoreIndexCreator

from gpt_module import get_openai_api_key

class RAG_GPT:
    MODEL_NAME = "gpt-4"

    def __init__(self):
        os.environ["OPENAI_API_KEY"] = get_openai_api_key()
        # Load KnowledgeBase from data files
        loader = TextLoader("./database/live.json")
        # Internal source
        self.index = VectorstoreIndexCreator().from_loaders([loader])
        # choose llm model use
        self.openai = ChatOpenAI(model_name=self.MODEL_NAME)

    def query(self, prompt):
        # generate complete prompt
        response = self.index.query(prompt, llm=self.openai, retriever_kwargs={"k": 1})
        return response