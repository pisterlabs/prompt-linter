import os
import config
os.environ["OPENAI_API_KEY"] = config.openAI

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI, VectorDBQA



def docx(filename):

    text_splitter = RecursiveCharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function = len
    )

    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(text_splitter.split_text(filename), embeddings)

    chain = VectorDBQA(OpenAI(), docsearch)

    def chat():
        while True:
            message = input("Enter a message: ")
            if message == "quit":
                break
            else:
                res = chain.run(question=message)
                print(res)
    chat()

docx("/home/bhargav/Downloads/AI_bots.docx")