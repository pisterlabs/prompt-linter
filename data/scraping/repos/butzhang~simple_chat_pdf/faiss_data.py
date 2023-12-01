from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ChatVectorDBChain
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from simple_chat_pdf.constant import PINECONE_API_KEY, PINECONE_ENVIRONMENT, OPENAI_API_KEY, PINE_CONE_INDEX_NAME


text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

loader = PyPDFLoader('./case_studies.pdf')
documents = loader.load_and_split(text_splitter)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
faiss_vector_store = FAISS.from_documents(documents, embeddings)