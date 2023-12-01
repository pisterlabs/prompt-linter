import openai
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.vectorstores import Chroma

openai.api_key = "PASTE YOUR KEY"


class AIHandler:
    def __init__(self) -> None:
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
        self.chat = ChatOpenAI(openai_api_key=openai.api_key, temperature=0)

    def init_vectordb(self, chunks):
        persist_directory = "data/docs/chroma"
        self.vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_directory,
        )
        return self.vectordb

    def get_model_output(self, input_prompt):
        model_output = self.chat([HumanMessage(content=input_prompt)])
        return model_output.content
