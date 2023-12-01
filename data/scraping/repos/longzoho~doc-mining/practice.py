import json

import torch
from huggingface_hub import hf_hub_download
from langchain import LlamaCpp, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PDFMinerLoader


def read_file(file_path: str) -> str:
    # read file content
    with open(file_path, 'rt') as f:
        return f.read()


chunk_size = 1000
chunk_overlap = 200
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

max_ctx_size = 2048
model_id = "TheBloke/Llama-2-7B-Chat-GGML"
model_basename = "llama-2-7b-chat.ggmlv3.q4_0.bin"
hf_hub_download(repo_id=model_id, filename=model_basename)


def load_document():
    # load document
    loader = PDFMinerLoader(file_path='./resource-test/bb5dd3ac3401ef5cc8aa67b4e27a3c7f.pdf')
    document = loader.load()[0]
    doc_json = document.json()
    # save document as json file
    with open('./resource-test/bb5dd3ac3401ef5cc8aa67b4e27a3c7f.pdf.json', 'w') as f:
        f.write(doc_json)


def embedding_transformer():
    json_content = read_file(file_path='./resource-test/bb5dd3ac3401ef5cc8aa67b4e27a3c7f.pdf.json')
    # create document object from json
    document = Document(**json.loads(json_content))

    # sllit document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splited_documents = text_splitter.split_documents([document])

    # create embeddings
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large",
        model_kwargs={"device": 'cuda'},
    )

    # embedd document into chroma db
    db = Chroma.from_documents(
        splited_documents,
        embeddings,
        persist_directory='./resource-test/chroma')
    db.persist()


def query_document():
    model_path = hf_hub_download(repo_id=model_id, filename=model_basename)
    kwargs = {
        "model_path": model_path,
        "n_ctx": max_ctx_size,
        "max_tokens": max_ctx_size,
        # "n_threads": 16,
        "n_gpu_layers": 1000 if device_type.lower() in ["mps", "cuda"] else None,
        "n_batch": max_ctx_size if device_type.lower() == "cuda" else None,
        "verbose": False
    }

    # create lager language model
    llm = LlamaCpp(**kwargs)

    # create chroma retriever
    retriever = Chroma(
        persist_directory='./resource-test/chroma',
        embedding_function=HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-large",
            model_kwargs={"device": device_type},
        ),
    ).as_retriever()

    # create memory
    memory = ConversationBufferMemory(input_key="question", memory_key="history")

    # create prompt
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer,\
    just say that you don't know, don't try to make up an answer.

    {context}

    {history}
    Question: {question}
    Helpful Answer:"""

    prompt = PromptTemplate(input_variables=["history", "context", "question"], template=template)

    # create retriever question answer
    retrievalQA = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "memory": memory},
    )

    result = retrievalQA("What is key point of eCommerce")
    print(result.get("result"))


# main function
if __name__ == "__main__":
    # load_document()
    # embedding_transformer()
    query_document()
