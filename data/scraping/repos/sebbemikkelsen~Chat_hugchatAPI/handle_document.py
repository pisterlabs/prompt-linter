from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

directory = './docs'

def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

documents = load_docs(directory)



def split_docs(documents,chunk_size=1000,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

docs = split_docs(documents)




embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


db = Chroma.from_documents(docs, embeddings)





#query = "How many books is there in the library?"
query = "How many nations are there?"
#matching_docs = db.similarity_search_with_score(query, k=4)
matching_docs = db.similarity_search(query)

print(matching_docs[0]["page_content"])
