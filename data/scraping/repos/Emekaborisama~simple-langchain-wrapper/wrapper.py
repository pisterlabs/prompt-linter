from pydantic import BaseModel,validator
import os
from typing import List, Optional
from langchain.document_loaders import DirectoryLoader
import pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain



class ConnectionException(ConnectionError):
  pass



class langchain_docs_wrapper():
    def __init__(self, directory: str,
                pinecone_index_name: str ='', 
                chunk_size: int = 1000, 
                chunk_overlap: int = 20,
                k: int = 2,
                score: bool = False,
                model_name: Optional[str] = None,
                chain_type: str ="stuff"
                ) -> None:
      self.directory=directory
      self.score = score
      self.k = k
      self.pinecone_index_name = pinecone_index_name
      self.embedding = OpenAIEmbeddings()
      self.llm = OpenAI()
      self.chain = load_qa_chain(self.llm, chain_type=chain_type)
      self.chunk_size = chunk_size
      self.chunk_overlap =chunk_overlap

    @classmethod
    def env_auth_variables(cls):
      cls.open_ai = os.environ['OPENAI_API_KEY']
      assert len(cls.open_ai)>1, """Invalid API key --- use os.environ["OPENAI_API_KEY"] = xxxxx ---- to set your OpenAI api key"""

      cls.pinecone_api_key = os.environ['pinecone_api_key']
      assert len(cls.pinecone_api_key) >1, """Invalid API key --- use os.environ["pinecone_api_key"] = xxxxx ---- to set your OpenAI api key"""

      cls.pinecone_environment = os.environ['pinecone_environment']
      assert len(cls.pinecone_environment)>1, """Invalid API key --- use os.environ["pinecone_api_key"] = xxxxx ---- to set your OpenAI api key"""
      cls.pinecone_index_name =os.environ['pinecone_index_name'] 
      assert len(cls.pinecone_index_name )>1, """Invalid API key --- use os.environ["pinecone_index_name"] = xxxxx ---- to set your OpenAI api key"""
      print (f'authorized pinecone project {cls.pinecone_index_name}')
      global pinecone_index_name
      pinecone_index_name = cls.pinecone_index_name


    @classmethod
    def pinecone_auth_wrapper(cls):
      pinecone.init(
          api_key=cls.pinecone_api_key,
          environment=cls.pinecone_environment
          )
      try:
        pinecone.list_indexes()
        pinecone.Index(cls.pinecone_index_name)
      except Exception as e:
        raise ConnectionException(f"Error connecting to API at {e}")
      
    @classmethod
    def openai_api_auth_wrapper(cls):
      open_Ai_auth = OpenAIEmbeddings()
      try:
        open_Ai_auth.embed_query("Hello world")
        pass
      except Exception as e:
        raise ConnectionException(f"Error connecting to API at {e}")


    
    @classmethod
    def auth_wrapper(cls):
      cls.pinecone_auth_wrapper()
      cls.openai_api_auth_wrapper()


    def load_and_split_docs(self)-> List[str]:
      loader = DirectoryLoader(self.directory)
      document = loader.load()
      text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
      docs = text_splitter.split_documents(document)
      return docs

    def get_similiar_docs(self,query):
      index = self.generate_store_embeddings()
      if self.score:
        return index.similarity_search_with_score(query, k=self.k)
      else:
        return index.similarity_search(query=query, k=self.k)


    def generate_store_embeddings(self):
      docs = self.load_and_split_docs()
      index = Pinecone.from_documents(documents=docs, embedding=self.embedding, index_name=pinecone_index_name)
      return index


    def get_answer(self, query):
      similiar_docs = self.get_similiar_docs(query =query)
      answer = self.chain.run(input_documents=similiar_docs, question=query)
      return answer





