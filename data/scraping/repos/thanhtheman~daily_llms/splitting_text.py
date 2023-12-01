from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

loader = TextLoader("sample.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
print(f"you have {len(documents)} documents")
embedding_list = embeddings.embed_documents([i.page_content for i in documents])
print(f"you have {len(embedding_list)} embeddings")
print(f"here is the the first embedding: {embedding_list[0][:3]}...")
