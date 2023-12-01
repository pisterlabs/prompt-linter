#%%

PROJECT_ID = "cloud-llm-preview1"  # @param {type:"string"}
LOCATION = "us-central1" # @param {type:"string"}

import langchain
from google.cloud import aiplatform
import vertexai
vertexai.init(project=PROJECT_ID, location=LOCATION)

from genai import VertexLLM, VertexChat, VertexMultiTurnChat, VertexEmbeddings

REQUESTS_PER_MINUTE = 100

llm = VertexLLM(
    model_name='text-bison@001',
    max_output_tokens=256,
    temperature=0.1,
    top_p=0.8,
    top_k=40,
    verbose=True,
)

chat = VertexChat()
mchat = VertexMultiTurnChat(max_output_tokens=1024)
embedding = VertexEmbeddings(requests_per_minute=REQUESTS_PER_MINUTE)

## Q/A Chain
#%%
# Ingest PDF files
from langchain.document_loaders import PyPDFLoader

# Load GOOG's 10K annual report (92 pages).
url = "https://abc.xyz/investor/static/pdf/20230203_alphabet_10K.pdf"
loader = PyPDFLoader(url)
documents = loader.load()

# %%
# split the documents into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
print(f"# of documents = {len(docs)}")

# %%
# Store docs in local vectorstore as index
# it may take a while since API is rate limited
from langchain.vectorstores import Chroma

db = Chroma.from_documents(docs, embedding)

# %%
# Expose index to the retriever
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k":2})

#%%
# Create chain to answer questions
from langchain.chains import RetrievalQA

# Uses LLM to synthesize results from the search index.
# We use Vertex PaLM Text API for LLM
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True)

query = "What was Alphabet's net income in 2022?"
result = qa({"query": query})
print(result)
# %%
