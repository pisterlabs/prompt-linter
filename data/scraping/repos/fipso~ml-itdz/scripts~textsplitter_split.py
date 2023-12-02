from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import TokenTextSplitter

directory = "./textsplitter/data"

def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

documents = load_docs(directory)
len(documents)

def split_docs(documents, chunk_size=1500, chunk_overlap=20):
  text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

docs = split_docs(documents)
#print(docs)

for i in docs:
    print(i.page_content)
    print(len(i.page_content.split(" ")))
    print("=" * 80)
