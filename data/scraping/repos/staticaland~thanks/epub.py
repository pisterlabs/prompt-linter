from langchain.document_loaders import UnstructuredEPubLoader

loader = UnstructuredEPubLoader("captain-charles-johnson_a-general-history-of-the-pirates.epub")

documents = loader.load()

from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()

from langchain.vectorstores import FAISS

db = FAISS.from_documents(docs, embeddings)

query = "Who was the captain of the ship?"
docs = db.similarity_search(query)

print(docs[0].page_content)