from langchain.vectorstores import chroma
from backendLLM.dataReading import *
from backendLLM.utils import *


class GetDataBase:
  def __init__(self , db_path , file_paths , enrich_metadata=False):
    self.db_path = db_path
    self.file_paths = file_paths
    self.enrich_metadata = enrich_metadata
    
    self.db = chroma.Chroma(embedding_function=embedder , persist_directory=db_path)

  def add_data(self):
    data = DataReading(self.file_paths)
    docs:List[Document] = data.read_all()
    splitter = DataSplitting(docs , enrich_metadata=self.enrich_metadata)

    pages:List[Document] = splitter.split()
    self.idxs = [i for i in range(len(pages))]
    self.db.add_documents(pages, id = self.idxs)
    return
  
  def delete_db(self):
    self.db.delete(self.idxs)

  


    


    