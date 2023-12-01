from langchain.vectorstores import FAISS 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
from tqdm import tqdm
import hashlib
load_dotenv()
from langchain.document_loaders import PDFMinerLoader
from utils import config_loader
from langchain.chains.question_answering import load_qa_chain

class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata
    
    def __call__(self):
         return f"Document(page_content={self.page_content},metadata={self.metadata})"
    
    def get_pagecontent(self):
         return self.page_content
    
    def get_metadata(self):
         return self.get_metadata
    
def pdf_to_json_and_insert(filepath):
        documents=[]
        loader = PDFMinerLoader(filepath)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function = len,
            separators=["\n\n", "\n", " ", ""],

        )
        m = hashlib.md5()  # this will convert URL into unique ID
        for doc in tqdm(docs):
            url = doc.metadata["source"].split("/")[-1]
            m.update(url.encode("utf-8"))
            chunks = text_splitter.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                doc = Document(page_content=chunk, metadata={"source":url})

                documents.append(
                    doc
                )
        embedding = openai_embedding()
        global knowledge_base
        knowledge_base = FAISS.from_documents(documents=documents, 
                                    embedding=embedding)
        return knowledge_base

def openai_embedding():
    model_name = config_loader["openai_embedding_model"]
    embed = OpenAIEmbeddings(
        model=model_name,
        openai_api_key="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",
    )
    return embed




def retriever_faiss(query):
    retrieve=knowledge_base.similarity_search(query=query)
    llm = OpenAI(openai_api_key="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")
    chain = load_qa_chain(llm, chain_type='map_rerank')
    return chain.run(input_documents=retrieve, question=query)
