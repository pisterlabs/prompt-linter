# Using the combination of LlamaIndex and LangChain provides improvements in the following ways:
# Simplification: LangChain simplifies the process of working with large language models (LLMs) and related technologies. It provides a uniform wrapper around several related APIs, making it quick and powerful for experimentation
#
# Modularity: LangChain introduces key concepts like chain prompting, making it more accessible for users. It also allows for easy swapping of components (models, vector stores, etc.)
#
# Efficiency: LlamaIndex is more efficient than LangChain, making it a better choice for certain tasks
#
# However, there are some limitations to using LlamaIndex and LangChain compared to the previous code:
# Unnecessary Abstraction: LangChain has been criticized for introducing unnecessary abstraction and indirection, making simple LLM tasks more complex than just using Python and APIs directly
#
# Poor Documentation: LangChain has been criticized for its poor documentation, lack of customizability, and difficulty debugging

# Performance Issues: Some users have reported that LangChain is inefficient and can be slower than alternative implementations

# In contrast, the previous code provided a more direct approach to implementing a question-answering system using RAG and OpenAI API. It might be easier to understand and customize for specific use cases. However, it lacks the modularity and abstraction provided by LangChain and LlamaIndex, which can be beneficial for certain applications



import os
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader
from langchain import LangChain
from flask import Flask, request, render_template

# Document Preparation
def extract_text_from_pdfs(pdf_dir):
    loader = PyPDFLoader()
    documents = loader.load_data(pdf_dir)
    return documents

# Document Embedding
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cuda"}
)
embeddings = HuggingFaceEmbeddings(text_splitter)

# Setting up LLama Index
def create_llama_index(documents):
    index = GPTSimpleVectorIndex(documents)
    return index

# Setting up LangChain
def create_langchain(index):
    langchain = LangChain(index)
    return langchain

# Flask Application
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        question = request.form['question']
        answer = langchain.ask(question)
        return render_template('index.html', answer=answer)
    return render_template('index.html')

if __name__ == '__main__':
    pdf_dir = '/path/to/pdf/dir'
    documents = extract_text_from_pdfs(pdf_dir)
    llama_index = create_llama_index(documents)
    langchain = create_langchain(llama_index)
    app.run(debug=True)