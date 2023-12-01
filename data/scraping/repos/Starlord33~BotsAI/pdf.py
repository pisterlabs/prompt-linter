import os
import config
os.environ["OPENAI_API_KEY"] = config.openAI

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from PyPDF2 import *
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, FAISS, Weaviate

chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)

def pdf_to_bot(filename):
    reader = PdfReader(filename)

    raw_text = ''
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
            
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function = len
    )

    text_chunk = text_splitter.split_text(raw_text)

    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(text_chunk, embeddings)

    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    def chat():
        while True:
            message = input("Enter a message: ")
            if message == "quit":
                break
            else:
                docs = docsearch.similarity_search(message)
                res = chain.run(input_documents=docs, question=message)
                print(res)
    chat()

pdf_to_bot("/home/bhargav/Hadoop/hadoopinstallation.pdf")