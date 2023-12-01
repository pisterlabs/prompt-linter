# https://python.langchain.com/docs/integrations/vectorstores/scann

from langchain.embeddings       import HuggingFaceEmbeddings
from langchain.text_splitter    import CharacterTextSplitter
from langchain.vectorstores     import ScaNN
from langchain.document_loaders import TextLoader

loader    = TextLoader("state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

from langchain.embeddings import TensorflowHubEmbeddings
embeddings = HuggingFaceEmbeddings()

db = ScaNN.from_documents(docs, embeddings)
query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)

docs[0]
