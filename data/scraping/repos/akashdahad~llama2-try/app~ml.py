# Installation:
# CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir

from flask import Flask, request, jsonify
from langchain.llms import LlamaCpp
from langchain.llms import CTransformers
from langchain import PromptTemplate, LLMChain
import torch
from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import LlamaCppEmbeddings
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.document_loaders import WebBaseLoader

app = Flask(__name__)

embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
embed_model = HuggingFaceEmbeddings(
model_name=embed_model_id,
model_kwargs={'device': device},
encode_kwargs={'device': device, 'batch_size': 32})


##loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")

n_gpu_layers = 32  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

def load_llm():
    llm = LlamaCpp(
    model="TheBloke/Llama-2-7B-Chat-GGML",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=False)
    return llm

def load_pipeline(all_splits):
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=embed_model)
    docs = vectorstore.similarity_search(question)
    rag_pipeline = RetrievalQA.from_chain_type(
    llm=llm, chain_type='stuff',
    retriever=vectorstore.as_retriever())
    return rag_pipeline

@app.route("/")
def home():
    return "Hello, World!"

@app.route('/llm', methods = ['POST'])
def answer():
    body = request.get_json()
    # result = body['prompt'], content=body['content']
    loader = WebBaseLoader("https://www.quadratics.com/MLOPSimplified.html")
    data = loader.load()
    print(data)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    all_splits = text_splitter.split_documents(data)
    rag_pipeline = load_pipeline(all_splits)
    result = rag_pipeline("what accelerators did quadratic build") #Add body['prompt'] here
    return result

def create_app():
   return app


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        print(f"GPU Detected: {device}")
    else:
        print("No GPU detected. Using CPU.")
    llm = load_llm()

    
    app.run(host="0.0.0.0", port=5005)
