from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

n_gpu_layers = 32  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Load Model
llm = LlamaCpp(model_path="./models/llama-2-7b-chat.ggmlv3.q2_K.bin", n_gpu_layers=n_gpu_layers, n_batch=n_batch, n_ctx=2048, f16_kv=True, callback_manager=callback_manager, verbose=False)

# Embedding Model Details
embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
device = 'cpu'

embed_model = HuggingFaceEmbeddings(model_name=embed_model_id, model_kwargs={'device': device}, encode_kwargs={'device': device, 'batch_size': 32})

arr = ['./india.txt', './china.txt']

# Load Data
data = []
for d in arr:
    loader = UnstructuredFileLoader(d)
    data.append(loader.load()[0])

# Split Data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
all_splits = text_splitter.split_documents(data)

print(len(all_splits))

# Define Question
question = "difference between indian and chinese geography"

# Define Store and Retrieve
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embed_model)
# docs = vectorstore.similarity_search(question)

print(len(docs))

rag_pipeline = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=vectorstore.as_retriever())
result = rag_pipeline(question)

print(result)