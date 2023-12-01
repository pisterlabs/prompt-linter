from typing import List
import os
import urllib
from bs4 import BeautifulSoup

from langchain import PromptTemplate, LLMChain
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import GPT4All
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Constants
local_path = '../../models/ggml-gpt4all-j.bin'
index_path = "../data/pci_dss_v4/embedding_store/pci_dss_v4_index"
html_path = "../data/pci_dss_v4/pci_dss_docs.html"

# Functions
def initialize_embeddings() -> HuggingFaceInstructEmbeddings:
    return HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                         model_kwargs={"device": "cpu"})


def parse_html():
    html_page = urllib.request.urlopen("file://" + os.path.abspath(html_path))
    soup = BeautifulSoup(html_page, "html.parser")
    urls = []
    for link in soup.findAll('a'):
        url = link.get('href')
        if not url in urls and url.lower().endswith('.pdf') and url.startswith('http'):
            urls.append(url)

    urls.sort(key=lambda x: x.split('/')[-1])

    for url in urls:
        print(url)

    print(len(urls))
    return urls


def load_documents() -> List:
    # loader = DirectoryLoader('../data/pci_dss_v4/pdfs/', glob="./*.pdf", loader_cls=PyPDFLoader)
    loader = DirectoryLoader('../data/pci_dss_v4/pdfs/',
                             glob="./PCI-DSS-v3-2-1-to-v4-0-Summary-of-Changes-r2.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    urls = parse_html()

    for text in documents:
        source = text.metadata['source']
        filename = source.split('/')[-1]
        for url in urls:
            if url.endswith(filename):
                text.metadata['url'] = url
                break

    return documents


def split_chunks(sources: List) -> List:
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=32)
    return splitter.split_documents(sources)


def generate_index(chunks: List, embeddings: HuggingFaceInstructEmbeddings) -> FAISS:
    vectorStore = FAISS.from_documents(chunks, embeddings)
    vectorStore.save_local(index_path, "wb")
    return vectorStore


# Main execution
# Callbacks support token-wise streaming
callbacks = [StreamingStdOutCallbackHandler()]
# Verbose is required to pass to the callback manager
# llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True)
# If you want to use GPT4ALL_J model add the backend parameter
llm = GPT4All(model=local_path, n_ctx=2048, backend='gptj',
              callbacks=callbacks, verbose=True)

embeddings = initialize_embeddings()
# sources = load_documents()
# chunks = split_chunks(sources)
# index = generate_index(chunks, embeddings)

index = FAISS.load_local(index_path, embeddings)

qa = ConversationalRetrievalChain.from_llm(
    llm, index.as_retriever(), max_tokens_limit=400)

# Chatbot loop
chat_history = []
print("Welcome to the State of the Union chatbot! Type 'exit' to stop.")
while True:
    query = input("Please enter your question: ")

    if query.lower() == 'exit':
        break
    result = qa({"question": query, "chat_history": chat_history})

    print("Answer:", result['answer'])
