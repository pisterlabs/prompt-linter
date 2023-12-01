import openai


import os

os.environ['OPENAI_API_KEY'] = "to be replaced"
from langchain.indexes import VectorstoreIndexCreator

from langchain.document_loaders.csv_loader import CSVLoader
csvloader = CSVLoader(file_path="C:\Prj\LLM\Hackathon\health_log\media\measurements.txt", source_column="measurement type")

def ai_answer(query):

  index = VectorstoreIndexCreator().from_loaders([csvloader])
  answer = index.query(query).strip()
  
  return answer
  

