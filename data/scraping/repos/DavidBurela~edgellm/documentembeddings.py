# pip install tiktoken faiss-cpu
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import LlamaCppEmbeddings

### Cloud
embeddings = OpenAIEmbeddings()

### Edge
# embeddings = LlamaCppEmbeddings(model_path="./models/gpt4all-lora-quantized-new.bin")
# embeddings = LlamaCppEmbeddings(model_path="./models/ggml-vicuna-7b-4bit-rev1.bin", n_threads=16)
# embeddings = LlamaCppEmbeddings(model_path="./models/ggml-vicuna-13b-4bit-rev1.bin", n_threads=16)

# Load the document and split to fit in token context
loader = TextLoader('data/satya-openai-announcement.txt')
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
texts = text_splitter.split_documents(documents)
print(f"{len(texts)} chunks")

# Embedd your texts
db = FAISS.from_documents(texts, embeddings)
retriever = db.as_retriever()

# Retrieve relevant embeddings (Could also use a vector database here)
docs = retriever.get_relevant_documents("what years are mentioned")
for doc in docs:
    print("###")
    print(doc.page_content)