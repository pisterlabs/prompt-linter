from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator
import os


# Set your OpenAI API key
os.environ['OPENAI_API_KEY'] = "your open ai api key"

# Create the PDF loader and index
loaders = [UnstructuredPDFLoader(file_path="path of a file")]
index = VectorstoreIndexCreator().from_loaders(loaders)

# Query the index
print(index.query_with_sources("your query"))
