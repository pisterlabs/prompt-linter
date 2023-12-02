

from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.indexes import VectorstoreIndexCreator
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

query ="Please list all your shirts with sun protection \
in a table in markdown and summarize each one."

print(query)