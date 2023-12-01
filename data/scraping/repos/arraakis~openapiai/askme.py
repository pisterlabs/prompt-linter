import os, sys
import constants
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator

os.environ["OPENAI_API_KEY"] = constants.APIKEY

query = sys.argv[1]
loader = TextLoader('data.txt')
index = VectorstoreIndexCreator().from_loaders([loader])
print(index.query(query))