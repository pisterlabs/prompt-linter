#(scrape->load->split->[dwnld_instructor_e]->create_vec_graph->save(optional)->[download_model->]initialise_chain->query->)->api
from langchain.document_loaders import UnstructuredURLLoader #to load the source text 
from langchain.text_splitter import CharacterTextSplitter #chunking the load data

import faiss#vector graph knowledge base
from langchain.vectorstores import FAISS

from InstructorEmbedding import INSTRUCTOR# InstructorEmbedding
from langchain.embeddings import HuggingFaceInstructEmbeddings

from pydantic import BaseModel
from typing import Optional, Any


# Python code to illustrate the Modules
class SSV(BaseModel):
        urls: Any = ['https://example.com/','https://example.com/']
        chunk_size: int = 500
        chunk_overlap: int = 100 
        model_name: str = "hkunlp/instructor-xl"
        kwargs: int = 3
        instructor_embeddings: Optional[Any] = None
        data: Optional[Any] = None

        # def __init__(self, urls = ['https://it.pccoepune.com/','https://it.pccoepune.com/hod'], chunk_size = 500, chunk_overlap = 100, model_name = "hkunlp/instructor-xl", kwargs = 3):
        #         self.urls = urls
        #         self.chunk_size = chunk_size
        #         self.chunk_overlap = chunk_overlap
        #         self.model_name = model_name
        #         self.kwargs = kwargs
        

        # A normal print function
        def scrap(self):
                try:
                        if self.data:
                                print("data already exists")
       
                        else:
                                print("scraping...")                                 
                                loaders = UnstructuredURLLoader(urls=self.urls)
                                self.data = loaders.load()                        
                except:                            
                        print("scraping...")        
                        loaders = UnstructuredURLLoader(urls=self.urls)
                        self.data = loaders.load()

        def split(self):
                #initialised text splitter
                text_splitter = CharacterTextSplitter(separator=' ',
                                      chunk_size=self.chunk_size,
                                      chunk_overlap=self.chunk_overlap)
                #split the text                              
                docs = text_splitter.split_documents(self.data)                              
                return docs                              

        def vec(self, docs):
                #downloading instructor embedding
                try:
                        if self.instructor_embeddings:
                                print("Embedding model already exists")
                        else:
                                self.instructor_embeddings = HuggingFaceInstructEmbeddings(model_name=self.model_name,
                                                model_kwargs={"device": "cuda"})                        
                except:                            
                        self.instructor_embeddings = HuggingFaceInstructEmbeddings(model_name=self.model_name,
                                                      model_kwargs={"device": "cuda"})
            
                #creating vector graph
                db_instructEmbedd = FAISS.from_documents(docs, self.instructor_embeddings)
                retriever = db_instructEmbedd.as_retriever(search_kwargs={"k": self.kwargs})
                return retriever
        
        def ret(self):
                self.scrap()
                retriever = self.vec(self.split())
                return retriever
#urls, chunk size, overlap, search kwargrs, instructor model