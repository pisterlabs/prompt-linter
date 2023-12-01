import lancedb
from langchain.document_loaders import DirectoryLoader
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import LanceDB
from langchain.tools import tool
from pydantic import BaseModel, Field
from langchain.embeddings import OpenAIEmbeddings
import langchain

#langchain.debug = True

path_when_using_as_tool = "audio/structured_chat/knowledge_base/"
path_when_using_directly = "./"

path = path_when_using_as_tool

class KnowledgeBase:
    def __init__(self, uri: str, table_name: str = "restaurants_table") -> None:
        self.connection = lancedb.connect(uri)
        embeddings = OpenAIEmbeddings()
        try:
            self.table = self.connection.open_table(table_name)
            self.docsearch = LanceDB(connection=self.table, embedding=embeddings)
        except FileNotFoundError as e:
            embeddings = OpenAIEmbeddings()
            documents = self.get_documents(f"{path}/raw_data/")
            self.table = self.connection.create_table(table_name, data=[
                {"vector": embeddings.embed_query("Hello World"), "text": "Hello World", "id": "1"}
            ], mode="create")
            self.docsearch = LanceDB.from_documents(documents, embeddings, connection=self.table)
        self.qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0), chain_type="stuff", retriever=self.docsearch.as_retriever())

    def embeddings_func(self, batch: List[str]):
        return [self.model.encode(doc) for doc in batch]
    
    def get_documents(self, dir_path: str) -> List[Document]:
        loader = DirectoryLoader(dir_path, glob="**/*.txt")
        documents = loader.load()
        text_spitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = text_spitter.split_documents(documents)
        return split_docs
    
    def search(self, query: str) -> List[str]:
        return self.docsearch.similarity_search(query, k=3)
    
    def search_chain(self, query: str) -> str:
        return self.qa.run(query)


kb = KnowledgeBase(uri=f"{path}/data/restaurant-db")

class KnowledgeBaseSchema(BaseModel):
    query: str = Field(description = "information you want to find about the restaurant")

@tool("knowledge_base", args_schema=KnowledgeBaseSchema)
def knowledge_base(query: str) -> str:
    """ Use this whenever you want to search for restaurant services. Be precise it what you're looking for. """
    result = kb.search_chain(query)
    return result

# For testing the knowledge base
"""
while True:
    question = input("User: ")
    answer = kb.search_chain(question)
    print(answer)
"""

